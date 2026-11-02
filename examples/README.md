# Examples

Run these examples from a source checkout after installing KinoWeave.
The [installation guide](../docs/installation.md) describes the core and
optional environments.

## CPU planning and file exchange

```bash
python -m pip install .
python examples/coordinated_planning.py --backend reference --output /tmp/retreat-run
kinoweave validate /tmp/retreat-run/problem.json /tmp/retreat-run/result.json
```

The supplied T-shaped graphs require one robot to retreat while the other
passes. They restrict candidate motions; they do not represent physical
corridor walls.

| Example | Purpose |
| --- | --- |
| [contract_roundtrip.py](contract_roundtrip.py) | Read and write native data records. |
| [manipulation_contract.py](manipulation_contract.py) | Create a symbolic object-transfer input and an unstarted result, without planning. |
| [single_arm_pick_place.py](single_arm_pick_place.py) | Generate PRM connectors around configured skills, verify object geometry, and optionally display a Pick & Place. |
| [two_arm_handover.py](two_arm_handover.py) | Coordinate two arms through Pick, an atomic bimanual Handover, and Place using prefetched conflict tables. |
| [task_scheduling.py](task_scheduling.py) | Retry configured assignments and skill options, then improve a serial inspection schedule to parallel work. |
| [many_arm_inspection.py](many_arm_inspection.py) | Explore experimental two/four/eight-tool inspection in shared or paired synthetic workspaces. |
| [reference_coordination.py](reference_coordination.py) | Search a small supplied graph with the CPU reference. |
| [coordinated_planning.py](coordinated_planning.py) | Plan a retreat with the main planner on CPU or CUDA. |
| [playback_plan.py](playback_plan.py) | Record kinematic playback of a saved plan. |

Pass `--refine` to `two_arm_handover.py` to save the first plan, improve it,
and display the best plan with `--view`. See
[objectives and refinement](../docs/refinement.md) for metric definitions,
budgets, progress callbacks, and the cost history.

Run `python examples/task_scheduling.py --output /tmp/task-run` for finite
assignment/parameter/order search. Add `--view` in the visualization environment
for standalone playback, or `--backend curobo` in the CUDA environment.
See [task scheduling](../docs/task-scheduling.md) for catalogs, retry feedback,
shared geometry, budgets and the selected input/result pair.

For a larger proxy example, run
`python examples/many_arm_inspection.py --robots 8 --tasks 40 --topology paired --output /tmp/eight-tools`.
Add `--view` in the visualization environment. See
[scaling and measurement](../docs/performance.md) for the layout and evidence boundaries.

## GPU geometry and roadmaps

Use the locked CUDA environment and an accessible NVIDIA GPU:

```bash
uv sync --project environments/cuda12 --frozen
uv run --project environments/cuda12 --frozen python examples/gpu_geometry.py
uv run --project environments/cuda12 --frozen python examples/gpu_roadmap.py
```

[gpu_geometry.py](gpu_geometry.py) demonstrates FK, IK, and single-robot
collision queries. [gpu_roadmap.py](gpu_roadmap.py) demonstrates PRM construction
and batched inter-robot motion compatibility. See [CUDA](../docs/curobo.md)
and [roadmaps](../docs/roadmaps.md) for the supported model subset.

## Two UR5 models with browser playback

```bash
uv sync --project environments/viz --frozen
uv run --project environments/viz --frozen python examples/fetch_ur5.py
uv run --project environments/viz --frozen python examples/two_ur5.py --output /tmp/two-ur5 --view
```

[two_ur5.py](two_ur5.py) imports the [scene YAML](scenes/two_ur5.yaml), plans
coordinated motions, and optionally displays the saved trajectory. Use
`--independent` for wider base spacing or `--backend curobo` in the CUDA
environment for GPU planning. The fetch is explicit; import and replay do
not download assets.

See [visualization](../docs/visualization.md) for controls and separate
import/plan/view commands, [scene authoring](../docs/scene-authoring.md) for
custom models, and [asset licensing](../docs/third-party-assets.md) for the
UR5 source and notices.
