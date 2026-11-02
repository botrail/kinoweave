"""Optional cuRobo V2 geometry, with world-frame outputs and named joint ordering."""

from __future__ import annotations

import copy
import math
import tempfile
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from typing import TYPE_CHECKING

from .. import BackendInfo, Box, ContractError, PlanningProblem, Pose, Sphere
from ._description import (
    UnsupportedModelError,
    check_float32,
    description,
    has_self_pairs,
    rotation,
    validate_model,
)

if TYPE_CHECKING:
    import torch

CUROBO_COMMIT = "78fd485fa82d9b9a063fb4985e371814587e666a"
CUROBO_VERSION = "0.8.0.post1.dev43"


class BackendUnavailableError(RuntimeError):
    """The optional CUDA stack is missing or cannot run on the requested device."""


@dataclass(frozen=True)
class FKBatch:
    link_names: tuple[str, ...]
    positions: torch.Tensor  # [batch, horizon, links, 3], world coordinates
    quaternions: torch.Tensor  # [batch, horizon, links, 4], xyzw
    spheres: torch.Tensor  # [batch, horizon, spheres, 4], xyzr in input sphere order
    sphere_links: tuple[str, ...]


@dataclass(frozen=True)
class CollisionBatch:
    self_collision: torch.Tensor  # bool [batch, horizon]
    world_collision: torch.Tensor  # bool [batch, horizon], static obstacles only
    within_limits: torch.Tensor  # bool [batch, horizon]

    @property
    def valid(self) -> torch.Tensor:
        return self.within_limits & ~self.self_collision & ~self.world_collision


@dataclass(frozen=True)
class IKBatch:
    joint_names: tuple[str, ...]
    positions: torch.Tensor  # [batch, dof]; use only rows marked successful
    success: torch.Tensor  # bool [batch]
    position_error: torch.Tensor  # [batch], metres
    orientation_error: torch.Tensor  # [batch], radians


class CuroboBackend:
    """Prepare one robot against static obstacles.

    GpuCollisionEvaluator composes these backends for inter-robot motion checks.

    Inputs accept [dof], [batch,dof], or [batch,horizon,dof] and use the
    model's joint order unless explicit joint names are supplied. Computation
    and returned batches stay on CUDA. The instance owns reusable cuRobo buffers
    and is not thread-safe. Use one instance per concurrent worker.
    """

    def __init__(
        self, problem: PlanningProblem, robot: str, *, use_cuda_graph: bool = True
    ) -> None:
        matches = [instance for instance in problem.robots if instance.name == robot]
        if not matches:
            raise ContractError("robot", f"unknown robot {robot!r}")
        self.instance = matches[0]
        self.model = next(model for model in problem.models if model.name == self.instance.model)
        validate_model(self.model)
        if problem.config.backend != "curobo":
            raise ContractError("config.backend", "explicitly select the curobo backend")
        if not problem.config.device.startswith("cuda"):
            raise BackendUnavailableError("cuRobo requires a CUDA device; CPU fallback is disabled")
        try:
            import curobo
            import torch
            from curobo.collision_checking import RobotCollisionChecker, RobotCollisionCheckerCfg
            from curobo.types import DeviceCfg
        except ImportError as error:
            raise BackendUnavailableError(
                "Install the qualified environments/cuda12 stack to use cuRobo V2"
            ) from error
        self._torch = torch
        self.device = torch.device(problem.config.device)
        if not torch.cuda.is_available():
            raise BackendUnavailableError(
                "CUDA is unavailable: check driver and device access, including sandbox permissions"
            )
        try:
            torch.empty(1, device=self.device)
            torch.cuda.synchronize(self.device)
        except RuntimeError as error:
            raise BackendUnavailableError(f"Cannot run on {self.device}: {error}") from error
        if curobo.__version__ != CUROBO_VERSION:
            raise BackendUnavailableError(f"Unqualified cuRobo API version {curobo.__version__}")
        self.problem = problem
        self.info = BackendInfo(name="curobo", version=curobo.__version__, device=str(self.device))
        self.dependency_versions = {
            name: version(name) for name in ("torch", "warp-lang", "cuda-core", "nvidia-curobo")
        }
        self._device_cfg = DeviceCfg(device=self.device, dtype=torch.float32)
        self._directory = tempfile.TemporaryDirectory(prefix="kinoweave-curobo-")
        self._closed = False
        self._ik_solvers = {}
        self._use_cuda_graph = use_cuda_graph
        try:
            check_float32(self.instance.base_pose.position, "base_pose")
            self._data, self._sphere_indices = description(
                self.model,
                Path(self._directory.name) / "robot.urdf",
                problem.config.collision_margin + problem.config.collision_tolerance,
            )
            self._world_from_base = torch.tensor(
                rotation(self.instance.base_pose), device=self.device, dtype=torch.float32
            )
            self._base_position = torch.tensor(
                self.instance.base_pose.position, device=self.device, dtype=torch.float32
            )
            self._base_quaternion = torch.tensor(
                self.instance.base_pose.quaternion, device=self.device, dtype=torch.float32
            )
            self._base_quaternion /= torch.linalg.vector_norm(self._base_quaternion)
            scene = {"cuboid": {}}
            self._sphere_obstacles = []
            self._boxes = []
            for obstacle in problem.obstacles:
                local = self._local_pose(obstacle.pose)
                check_float32((*obstacle.pose.position, *local.position), obstacle.name)
                if isinstance(obstacle.shape, Box):
                    check_float32(obstacle.shape.size, obstacle.name)
                    scene["cuboid"][obstacle.name] = {
                        "pose": [*local.position, local.quaternion[3], *local.quaternion[:3]],
                        "dims": list(obstacle.shape.size),
                    }
                    self._boxes.append(
                        (
                            torch.tensor(local.position, device=self.device, dtype=torch.float32),
                            torch.tensor(rotation(local), device=self.device, dtype=torch.float32),
                            torch.tensor(
                                obstacle.shape.size, device=self.device, dtype=torch.float32
                            )
                            / 2,
                        )
                    )
                elif isinstance(obstacle.shape, Sphere):
                    check_float32((obstacle.shape.radius,), obstacle.name)
                    self._sphere_obstacles.append((*obstacle.pose.position, obstacle.shape.radius))
                else:
                    raise UnsupportedModelError(obstacle.name, "unsupported obstacle shape")
            self._scene = scene if scene["cuboid"] else None
            checker_cfg = RobotCollisionCheckerCfg.load_from_config(
                robot_config=copy.deepcopy(self._data),
                scene_model=self._scene,
                device_cfg=self._device_cfg,
                collision_activation_distance=(
                    problem.config.collision_margin + problem.config.collision_tolerance
                ),
                n_meshes=0,
                n_cuboids=max(1, len(scene["cuboid"])),
            )
            self._checker = RobotCollisionChecker(checker_cfg)
            self._kinematics = self._checker.kinematics
            self._native_names = tuple(self._kinematics.joint_names)
            self._check_joint_set(self._native_names)
            self._has_self_pairs = has_self_pairs(self.model)
            self._pairs = self._kinematics.get_self_collision_config().collision_pairs.long()
            self.joint_names = self.model.joint_order
            self.sphere_links = tuple(sphere.link for sphere in self.model.collision_spheres)
        except Exception:
            self._directory.cleanup()
            raise

    def _check_open(self) -> None:
        if self._closed:
            raise RuntimeError("backend is closed")

    def _check_joint_set(self, names: tuple[str, ...]) -> None:
        if len(names) != len(set(names)) or set(names) != set(self.model.joint_order):
            raise UnsupportedModelError(self.model.name, f"joint set changed: {names!r}")

    def _positions(self, positions, joint_names: tuple[str, ...] | None = None):
        self._check_open()
        torch = self._torch
        names = self.model.joint_order if joint_names is None else tuple(joint_names)
        self._check_joint_set(names)
        try:
            tensor = torch.as_tensor(positions, device=self.device)
        except (ValueError, TypeError) as error:
            raise ContractError("positions", "expected a rectangular numeric batch") from error
        if tensor.dtype == torch.bool or tensor.is_complex():
            raise ContractError("positions", "boolean and complex joint values are unsupported")
        tensor = tensor.to(dtype=torch.float32)
        if tensor.ndim == 1:
            tensor = tensor[None, None, :]
        elif tensor.ndim == 2:
            tensor = tensor[:, None, :]
        if tensor.ndim != 3 or tensor.shape[-1] != len(names) or min(tensor.shape) == 0:
            raise ContractError("positions", f"expected non-empty [batch,horizon,{len(names)}]")
        if not torch.isfinite(tensor).all().item():
            raise ContractError("positions", "joint values must remain finite in float32")
        return tensor[..., [names.index(name) for name in self._native_names]].contiguous()

    def _local_pose(self, world: Pose) -> Pose:
        # Transform in host precision so scalar pose conversion does not add a GPU sync.
        matrix = rotation(self.instance.base_pose)
        delta = tuple(
            a - b for a, b in zip(world.position, self.instance.base_pose.position, strict=True)
        )
        position = tuple(sum(matrix[j][i] * delta[j] for j in range(3)) for i in range(3))
        x, y, z, w = self.instance.base_pose.quaternion
        a, b, c, d = world.quaternion
        quaternion = (
            w * a - x * d - y * c + z * b,
            w * b + x * c - y * d - z * a,
            w * c - x * b + y * a - z * d,
            w * d + x * a + y * b + z * c,
        )
        norm = math.hypot(*quaternion)
        return Pose(position=position, quaternion=tuple(v / norm for v in quaternion))

    def _state(self, tensor):
        from curobo.types import JointState

        state = self._kinematics.compute_kinematics(
            JointState.from_position(tensor, joint_names=list(self._native_names))
        )
        for values in (state.tool_poses.position, state.tool_poses.quaternion, state.robot_spheres):
            if not self._torch.isfinite(values).all().item():
                raise ContractError("geometry", "non-finite GPU geometry")
        return state

    def _world_batch(self, state) -> FKBatch:
        torch = self._torch
        positions = state.tool_poses.position @ self._world_from_base.T + self._base_position
        local = state.tool_poses.quaternion[..., [1, 2, 3, 0]]
        xyz, w = local[..., :3], local[..., 3:]
        base_xyz, base_w = self._base_quaternion[:3], self._base_quaternion[3]
        base_xyz = base_xyz.expand_as(xyz)
        quaternions = torch.cat(
            (
                base_w * xyz + w * base_xyz + torch.linalg.cross(base_xyz, xyz, dim=-1),
                base_w * w - (base_xyz * xyz).sum(dim=-1, keepdim=True),
            ),
            dim=-1,
        )
        spheres = state.robot_spheres[..., list(self._sphere_indices), :].clone()
        spheres[..., :3] = spheres[..., :3] @ self._world_from_base.T + self._base_position
        if not torch.isfinite(positions).all().item() or not torch.isfinite(spheres).all().item():
            raise ContractError("geometry", "non-finite world geometry")
        return FKBatch(
            tuple(state.tool_frames),
            positions,
            quaternions,
            spheres,
            self.sphere_links,
        )

    def forward(self, positions, *, joint_names: tuple[str, ...] | None = None) -> FKBatch:
        """Compute link poses and exact input sphere geometry in the world frame."""
        return self._world_batch(self._state(self._positions(positions, joint_names)))

    def collision(self, positions, *, joint_names: tuple[str, ...] | None = None) -> CollisionBatch:
        """Check discrete configurations against self and static obstacles.

        cuRobo collision costs are activation costs, not signed clearances.
        Static sphere obstacles use exact sphere arithmetic on CUDA, avoiding
        upstream mesh approximation. No moving-robot interactions are checked.
        """
        tensor = self._positions(positions, joint_names)
        state = self._state(tensor)
        return self._collision_state(tensor, state)

    def evaluate(
        self, positions, *, joint_names: tuple[str, ...] | None = None
    ) -> tuple[FKBatch, CollisionBatch]:
        """Return owned FK geometry and collision masks using one native FK evaluation."""
        tensor = self._positions(positions, joint_names)
        state = self._state(tensor)
        return self._world_batch(state), self._collision_state(tensor, state)

    def _collision_state(self, tensor, state) -> CollisionBatch:
        torch = self._torch
        box_cost = self._checker.get_collision_distance(state)
        if not torch.isfinite(box_cost).all().item():
            raise ContractError("collision", "non-finite native collision cost")
        world_collision = (box_cost > 0).any(dim=-1)
        threshold = self.problem.config.collision_margin + self.problem.config.collision_tolerance
        # Activation costs are zero at their boundary. Analytic CUDA guards
        # include exact contact (also when margin=tolerance=0) in the mask.
        local_spheres = state.robot_spheres
        for center, matrix, half_size in self._boxes:
            excess = torch.abs((local_spheres[..., :3] - center) @ matrix) - half_size
            signed = torch.linalg.vector_norm(excess.clamp_min(0), dim=-1) + excess.amax(
                -1
            ).clamp_max(0)
            world_collision |= (signed <= local_spheres[..., 3] + threshold).any(dim=-1)
        if self._has_self_pairs:
            cost = self._checker.get_self_collision_distance(state.robot_spheres)
            if not torch.isfinite(cost).all().item():
                raise ContractError("collision", "non-finite native self-collision cost")
            self_collision = (cost > 0).any(dim=-1)
            for chunk in self._pairs.split(256):
                first, second = (
                    local_spheres[..., chunk[:, 0], :],
                    local_spheres[..., chunk[:, 1], :],
                )
                distance = torch.linalg.vector_norm(first[..., :3] - second[..., :3], dim=-1)
                self_collision |= (distance <= first[..., 3] + second[..., 3] + threshold).any(-1)
        else:
            self_collision = torch.zeros(tensor.shape[:2], dtype=torch.bool, device=self.device)
        if self._sphere_obstacles:
            spheres = self._world_batch(state).spheres
            for x, y, z, radius in self._sphere_obstacles:
                center = torch.tensor((x, y, z), device=self.device, dtype=torch.float32)
                clearance = (
                    torch.linalg.vector_norm(spheres[..., :3] - center, dim=-1)
                    - spheres[..., 3]
                    - radius
                )
                world_collision = world_collision | (clearance <= threshold).any(dim=-1)
        within = torch.ones(tensor.shape[:2], dtype=torch.bool, device=self.device)
        limits = {
            joint.name: joint.limits for joint in self.model.joints if joint.limits is not None
        }
        for index, name in enumerate(self._native_names):
            limit = limits[name]
            within = (
                within & (tensor[..., index] >= limit.lower) & (tensor[..., index] <= limit.upper)
            )
        return CollisionBatch(self_collision, world_collision, within)

    def inverse(
        self,
        link: str,
        goals: tuple[Pose, ...],
        *,
        position_tolerance: float = 1e-3,
        orientation_tolerance: float = 1e-2,
    ) -> IKBatch:
        """Solve world-frame tool poses, then screen IK candidates for static validity."""
        self._check_open()
        if (
            link not in self.model.links
            or not goals
            or not all(isinstance(goal, Pose) for goal in goals)
        ):
            raise ContractError("goals", "supply a known link and a non-empty tuple of Pose goals")
        if not (
            math.isfinite(position_tolerance)
            and position_tolerance > 0
            and math.isfinite(orientation_tolerance)
            and orientation_tolerance > 0
        ):
            raise ContractError("tolerance", "IK tolerances must be finite and positive")
        from curobo.inverse_kinematics import InverseKinematics, InverseKinematicsCfg
        from curobo.types import GoalToolPose
        from curobo.types import Pose as NativePose

        key = (link, len(goals), position_tolerance, orientation_tolerance)
        if key not in self._ik_solvers:
            data = copy.deepcopy(self._data)
            data["kinematics"]["tool_frames"] = [link]
            cfg = InverseKinematicsCfg.create(
                robot=data,
                device_cfg=self._device_cfg,
                scene_model=self._scene,
                num_seeds=16,
                seed_solver_num_seeds=16,
                max_batch_size=len(goals),
                use_cuda_graph=self._use_cuda_graph,
                self_collision_check=self._has_self_pairs,
                random_seed=self.problem.config.seed,
                position_tolerance=position_tolerance,
                orientation_tolerance=orientation_tolerance,
                optimizer_collision_activation_distance=(
                    self.problem.config.collision_margin + self.problem.config.collision_tolerance
                ),
            )
            if not self._has_self_pairs:
                # The pinned API's flag disables optimizer checks only; its
                # separate metrics rollout must also ignore synthetic pairs.
                for cost_cfg in cfg.metrics_rollout_config.get_cost_manager_configs():
                    cost_cfg.disable_self_collision()
            solver = InverseKinematics(cfg)
            self._check_joint_set(tuple(solver.joint_names))
            self._ik_solvers[key] = solver
        solver = self._ik_solvers[key]
        torch = self._torch
        local = [self._local_pose(goal) for goal in goals]
        for pose in local:
            check_float32(pose.position, "goals")
        target = NativePose(
            position=torch.tensor(
                [pose.position for pose in local], device=self.device, dtype=torch.float32
            ),
            quaternion=torch.tensor(
                [[pose.quaternion[3], *pose.quaternion[:3]] for pose in local],
                device=self.device,
                dtype=torch.float32,
            ),
        )
        solved = solver.solve_pose(GoalToolPose.from_poses({link: target}, num_goalset=1))
        native_positions = solved.js_solution.position.reshape(len(goals), -1).clone()
        names = tuple(solved.js_solution.joint_names)
        self._check_joint_set(names)
        positions = native_positions[:, [names.index(name) for name in self.joint_names]]
        collision = self.collision(positions)
        actual = self.forward(positions)
        index = actual.link_names.index(link)
        target_position = torch.tensor(
            [g.position for g in goals], device=self.device, dtype=torch.float32
        )
        target_quaternion = torch.tensor(
            [g.quaternion for g in goals], device=self.device, dtype=torch.float32
        )
        target_quaternion /= torch.linalg.vector_norm(target_quaternion, dim=-1, keepdim=True)
        observed = actual.quaternions[:, 0, index]
        observed = observed / torch.linalg.vector_norm(observed, dim=-1, keepdim=True)
        observed = torch.where(
            (observed * target_quaternion).sum(-1, keepdim=True) < 0, -observed, observed
        )
        position_error = torch.linalg.vector_norm(
            actual.positions[:, 0, index] - target_position, dim=-1
        )
        orientation_error = 4 * torch.atan2(
            torch.linalg.vector_norm(observed - target_quaternion, dim=-1),
            torch.linalg.vector_norm(observed + target_quaternion, dim=-1),
        )
        success = (
            solved.success.reshape(-1).clone()
            & collision.valid[:, 0]
            & (position_error <= position_tolerance)
            & (orientation_error <= orientation_tolerance)
        )
        return IKBatch(
            self.joint_names,
            positions,
            success,
            position_error,
            orientation_error,
        )

    def close(self) -> None:
        if not self._closed:
            self._directory.cleanup()
            self._closed = True
            self._ik_solvers.clear()

    def __enter__(self) -> CuroboBackend:
        self._check_open()
        return self

    def __exit__(self, *args) -> None:
        self.close()
