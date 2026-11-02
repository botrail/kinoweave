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
| [reference_coordination.py](reference_coordination.py) | Search a small supplied graph with the CPU reference. |
| [coordinated_planning.py](coordinated_planning.py) | Plan a retreat with the main planner on CPU or CUDA. |
| [playback_plan.py](playback_plan.py) | Record kinematic playback of a saved plan. |

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
