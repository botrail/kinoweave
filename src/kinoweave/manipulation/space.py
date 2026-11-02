"""Mode-aware candidate generation and atomic task-progress transitions."""

from __future__ import annotations

from collections import OrderedDict, deque
from dataclasses import dataclass
from types import SimpleNamespace

from .._contracts import ContractError, value_as
from ..backends._description import UnsupportedModelError
from ..collision import Motion
from ..search.reference import JointGraph
from ..search.space import SearchLimit
from .conflicts import ConflictBuilder
from .models import HeldObject
from .task import (
    ActionClock,
    RunningSkill,
    TaskAction,
    TaskState,
    canonical_mode,
    object_owners,
    skill_clock,
)
from .timeline import Activity
from .validation import _rates


@dataclass(frozen=True, slots=True)
class PreparedStep:
    table: object
    options: tuple[tuple[int, ...], ...]
    holds: tuple[int, ...]


class TaskSpace:
    def __init__(self, evaluator, graphs, config, budget):
        self.evaluator, self.problem = evaluator, evaluator.manipulation
        self.config, self.budget, self.stats = config, budget, budget.stats
        self.dt = self.problem.scene.config.time_step
        self.names = tuple(r.name for r in self.problem.scene.robots)
        self.skills = self.problem.skeleton.skills
        by_robot = {g.robot: g for g in value_as(graphs, tuple[JointGraph, ...], "graphs")}
        if len(by_robot) != len(graphs) or by_robot.keys() != set(self.names):
            raise ContractError("graphs", "must cover every robot exactly once")
        self.graphs = tuple(by_robot[name] for name in self.names)
        self.models = {r.name: evaluator.cpu.models[r.model] for r in self.problem.scene.robots}
        self.adjacency, self.indices = [], []
        for robot, graph in zip(self.problem.scene.robots, self.graphs, strict=True):
            self.check("graph_preparation")
            if (
                graph.joint_names != self.models[robot.name].joint_order
                or graph.vertices[graph.start] != robot.start
                or graph.vertices[graph.goal] != robot.goal
            ):
                raise ContractError("graphs", "joint order and exact start/goal must match")
            if len(set(graph.vertices)) != len(graph.vertices):
                raise ContractError(
                    "graphs", "duplicate configurations are ambiguous skill anchors"
                )
            for q in graph.vertices:
                if evaluator.cpu.position_issues(robot.name, q):
                    raise ContractError("graphs", "position-invalid vertex")
            rows = [{i} for i in range(len(graph.vertices))]
            for a, b in graph.edges:
                rows[a].add(b)
                rows[b].add(a)
            keys = [(a, b) for a, row in enumerate(rows) for b in sorted(row) if a != b]
            flags = evaluator.motion_validity(
                robot.name,
                tuple(Motion(start=graph.vertices[a], end=graph.vertices[b]) for a, b in keys),
            )
            for (a, b), flag in zip(keys, flags, strict=True):
                if not flag:
                    rows[a].remove(b)
            self.adjacency.append(tuple(tuple(sorted(row)) for row in rows))
            self.indices.append({q: i for i, q in enumerate(graph.vertices)})
        order = self.problem.skeleton.ancestors()
        index = {s.name: i for i, s in enumerate(self.skills)}
        self.topological = tuple(index[name] for name in order)
        self.ancestors = tuple(frozenset(index[n] for n in order[s.name]) for s in self.skills)
        self.clocks, self.entries, self.exits, self.enabled = {}, {}, {}, {}
        for i, skill in enumerate(self.skills):
            for j, option in enumerate(skill.options):
                self.check("skill_preparation")
                duration, ticks = skill_clock(skill, option, self.dt, config.max_skill_ticks)
                self.clocks[i, j] = duration, ticks
                tracks = {t.robot: t for t in option.trace.tracks}
                participants = tuple(self.names.index(name) for name in skill.robots)
                entries, exits = [], []
                available = True
                for r in participants:
                    track = tracks[self.names[r]]
                    available &= (
                        track.positions[0] in self.indices[r]
                        and track.positions[-1] in self.indices[r]
                    )
                    entries.append(self.indices[r].get(track.positions[0], -1))
                    exits.append(self.indices[r].get(track.positions[-1], -1))
                self.entries[i, j], self.exits[i, j] = tuple(entries), tuple(exits)
                activity = Activity(skill.name, 0, duration, option.trace)
                dummy = SimpleNamespace(
                    problem=self.problem,
                    by_robot={
                        name: ([activity] if name in skill.robots else []) for name in self.names
                    },
                )
                self.enabled[i, j] = available and not _rates(dummy)
        self.start = TaskState(
            tuple(g.start for g in self.graphs),
            (None,) * len(self.names),
            frozenset(),
            canonical_mode(self.problem, {s.object: s for s in self.problem.initial_mode.states}),
        )
        self.builder = ConflictBuilder(self)
        self._tables, self._table_cells, self._distances = OrderedDict(), 0, {}
        for key in (
            "prefetch_calls",
            "prefetched_pairs",
            "gpu_pair_batches",
            "cpu_pair_batches",
            "peak_state_geometry_bytes",
            "max_pair_tile_bytes",
            "table_cache_hits",
            "peak_cached_table_pairs",
        ):
            self.stats[key] = 0

    def check(self, stage):
        self.budget.check(stage)

    def goal(self, state):
        return (
            len(state.completed) == len(self.skills)
            and all(r is None for r in state.running)
            and state.vertices == tuple(g.goal for g in self.graphs)
            and {s.object: s for s in state.mode.states}
            == {s.object: s for s in self.problem.goal_mode.states}
        )

    def ready(self, state, skill):
        return skill not in state.completed and self.ancestors[skill] <= state.completed

    def distances(self, robot, targets):
        key = (robot, tuple(sorted(targets)))
        if key not in self._distances:
            rows = [float("inf")] * len(self.graphs[robot].vertices)
            queue = deque(key[1])
            for i in queue:
                rows[i] = 0
            while queue:
                i = queue.popleft()
                for j in self.adjacency[robot][i]:
                    if rows[j] == float("inf"):
                        rows[j] = rows[i] + 1
                        queue.append(j)
            self._distances[key] = tuple(rows)
        return self._distances[key]

    def distance(self, state, robot, vertex=None):
        source = state.vertices[robot] if vertex is None else vertex
        pending = [
            i
            for i in self.topological
            if i not in state.completed and self.names[robot] in self.skills[i].robots
        ]
        if not pending:
            return self.distances(robot, (self.graphs[robot].goal,))[source]
        skill = pending[0]
        definition = self.skills[skill]
        if not self.ready(state, skill):
            return self.distances(robot, (self.graphs[robot].start,))[source]
        position = definition.robots.index(self.names[robot])
        targets = tuple(
            self.entries[skill, j][position]
            for j, o in enumerate(definition.options)
            if self.enabled[skill, j] and all(s in state.mode.states for s in o.preconditions)
        )
        return self.distances(robot, targets or (self.graphs[robot].start,))[source]

    def candidates(self, state):
        actions = []
        holds = [None] * len(self.names)
        active = set()
        for run in dict.fromkeys(r for r in state.running if r is not None):
            skill = self.skills[run.skill]
            participants = tuple(self.names.index(n) for n in skill.robots)
            if any(state.running[r] != run for r in participants):
                raise ContractError("task.running", "partial bimanual commitment")
            action = TaskAction(
                participants,
                self.exits[run.skill, run.option],
                "skill",
                run.skill,
                run.option,
                run.phase,
            )
            actions.append(action)
            for r in participants:
                active.add(r)
                holds[r] = action
        coupled = []
        for obj in state.mode.states:
            if isinstance(obj, HeldObject) and len(obj.grasps) == 2:
                pair = tuple(sorted(self.names.index(g.robot) for g in obj.grasps))
                if not set(pair) <= active:
                    if set(pair).intersection(active):
                        raise ContractError("task.running", "dual holders must stay coupled")
                    if any(set(pair).intersection(other) and pair != other for other in coupled):
                        raise UnsupportedModelError(
                            "task.mode", "overlapping dual-holder groups need a joint skill"
                        )
                    if pair not in coupled:
                        coupled.append(pair)
        for pair in coupled:
            action = TaskAction(pair, tuple(state.vertices[r] for r in pair), "wait")
            actions.append(action)
            for r in pair:
                holds[r] = action
        grouped = {r for pair in coupled for r in pair}
        for r in range(len(self.names)):
            if r in active or r in grouped:
                continue
            for target in self.adjacency[r][state.vertices[r]]:
                action = TaskAction(
                    (r,), (target,), "wait" if target == state.vertices[r] else "move"
                )
                actions.append(action)
                if target == state.vertices[r]:
                    holds[r] = action
        for i, skill in enumerate(self.skills):
            if not self.ready(state, i):
                continue
            participants = tuple(self.names.index(n) for n in skill.robots)
            if set(participants).intersection(active):
                continue
            # A dual-holder group cannot be split by a single-participant action.
            if any(
                set(pair).intersection(participants) and not set(pair) <= set(participants)
                for pair in coupled
            ):
                continue
            for j, option in enumerate(skill.options):
                if (
                    self.enabled[i, j]
                    and all(s in state.mode.states for s in option.preconditions)
                    and tuple(state.vertices[r] for r in participants) == self.entries[i, j]
                ):
                    actions.append(TaskAction(participants, self.exits[i, j], "skill", i, j, 0))
        actions = tuple(dict.fromkeys(actions))
        options = tuple(
            tuple(i for i, a in enumerate(actions) if r in a.robots) for r in range(len(self.names))
        )
        if any(len(row) > self.config.max_candidates_per_robot for row in options):
            raise SearchLimit("candidate_limit", "candidate_generation")
        return actions, options, tuple(actions.index(a) for a in holds)

    def prepare(self, state):
        self.evaluator._check_open()  # Also detects numerical-mode changes.
        precision = self.evaluator._precision
        key = state, precision
        if key in self._tables:
            self._tables.move_to_end(key)
            self.stats["table_cache_hits"] += 1
            return self._tables[key]
        actions, options, holds = self.candidates(state)
        table = self.builder.build(state, actions, object_owners(self, state))
        result = PreparedStep(table, options, holds)
        cells = len(actions) ** 2
        cfg = self.config
        if cfg.max_cached_tables and cells <= cfg.max_cached_table_pairs:
            while (
                len(self._tables) >= cfg.max_cached_tables
                or self._table_cells + cells > cfg.max_cached_table_pairs
            ):
                _, old = self._tables.popitem(last=False)
                self._table_cells -= len(old.table.actions) ** 2
            self._tables[key] = result
            self._table_cells += cells
            self.stats["peak_cached_table_pairs"] = max(
                self.stats["peak_cached_table_pairs"], self._table_cells
            )
        return result

    def preferred(self, state, prepared, robot):
        def cost(i):
            action = prepared.table.actions[i]
            target = action.targets[action.robots.index(robot)]
            return (action.kind != "skill", self.distance(state, robot, target), i)

        return tuple(sorted(prepared.options[robot], key=cost))

    def selection(self, prepared, choices):
        assigned = {}
        for action_id in choices:
            action = prepared.table.actions[action_id]
            if not prepared.table.valid[action_id]:
                return None
            for robot in action.robots:
                if robot in assigned and assigned[robot] != action_id:
                    return None
            for other in set(assigned.values()):
                if not prepared.table.compatible(action_id, other):
                    return None
            assigned.update((r, action_id) for r in action.robots)
        return assigned

    def advance(self, state, prepared, choices):
        assigned = self.selection(prepared, choices)
        if assigned is None or len(assigned) != len(self.names):
            return None
        unique = tuple(sorted(set(assigned.values())))
        vertices = list(state.vertices)
        running = list(state.running)
        completed = set(state.completed)
        events = []
        for i in unique:
            action = prepared.table.actions[i]
            if action.kind == "skill":
                clock = ActionClock(self, state, action)
                events.extend(clock.events)
                finished = action.phase + 1 >= clock.ticks
                for robot, target in zip(action.robots, action.targets, strict=True):
                    running[robot] = (
                        None
                        if finished
                        else RunningSkill(action.skill, action.option, action.phase + 1)
                    )
                    if finished:
                        vertices[robot] = target
                if finished:
                    completed.add(action.skill)
            else:
                for robot, target in zip(action.robots, action.targets, strict=True):
                    vertices[robot] = target
        modes = {s.object: s for s in state.mode.states}
        for _at, _rank, _index, event in sorted(events, key=lambda e: e[:3]):
            if modes[event.before.object] != event.before:
                return None
            modes[event.after.object] = event.after
        return TaskState(
            tuple(vertices),
            tuple(running),
            frozenset(completed),
            canonical_mode(self.problem, modes),
        )
