# KinoWeave

GPU-accelerated coordination planning for multiple robots.

KinoWeave coordinates fixed robot arms in shared workspaces. It combines
reusable individual roadmaps, batched collision checking, and CPU search to
plan movements, waits, and retreats. Results use a shared timeline and undergo
independent CPU validation.

The core has no runtime dependencies. It provides strict JSON exchange, PRM
construction, prioritized and LaCAM-style search, analytic joint-rate checks,
sampled collision validation, and an SDK-free memory player. The optional
cuRobo V2 backend adds CUDA geometry and tiled motion-pair checking.
Optional Viser visualization and URDF/scene-YAML import support standalone
browser demos, including two UR5 arms with real meshes.
The [single-arm manipulation API](docs/manipulation.md) plans Pick & Place
connectors around configured skills, with placed/held object geometry,
mode-specific PRM masks, independent CPU verification, and object playback.
The [multi-arm manipulation planner](docs/multi-arm-manipulation.md) adds
task progress, uninterrupted skills, atomic bimanual choices, and eager
CPU/CUDA conflict tables before lazy coordinated search.
[Anytime refinement](docs/refinement.md) evaluates the declared objective
and adopts independently verified single- or multi-arm shortcuts.
[Task schedule search](docs/task-scheduling.md) selects configured assignments,
skill parameters and resource orders, retries failed skeletons, and retains
the best verified plan across candidates.
An [experimental scaling example](docs/performance.md) exercises two, four
and eight synthetic tools with longer inspection task lists.

This is a development release. Planning uses link-attached collision spheres
and static sphere/box obstacles. Collision validation is sampled, and playback
is kinematic. Simulator SDK adapters and physical control are not included.

## Quick start

Install the source checkout with Python 3.11 or newer, then plan and replay
the included noninterfering two-arm example on the CPU:

```bash
python -m pip install .
kinoweave example --output /tmp/arms.json
kinoweave plan /tmp/arms.json --samples 0 --output /tmp/arms-run
kinoweave validate /tmp/arms-run/problem.json /tmp/arms-run/result.json
kinoweave playback /tmp/arms-run/problem.json /tmp/arms-run/result.json --output /tmp/playback.json
```

For CUDA planning, install the [locked GPU environment](docs/curobo.md)
and explicitly select `--backend curobo`. GPU requests never silently switch
to CPU planning. See the [command-line guide](docs/cli.md) for saved inputs,
failure results, and file exchange.

To display the saved example in a browser:

```bash
python -m pip install '.[viz]'
kinoweave view /tmp/arms-run/problem.json /tmp/arms-run/result.json
```

For a real-model demo, follow the [two-UR5 walkthrough](docs/visualization.md#two-ur5-arms).
Define your own robots with URDF plus explicit planning settings, and place
them in a [scene YAML](docs/scene-authoring.md).

## Documentation

Start with the [documentation index](docs/README.md),
[installation guide](docs/installation.md), and [runnable examples](examples/README.md).
The guides cover the CLI, Python planner, file formats, visualization, scene
authoring, and adapter contracts. English is the default documentation language.

## Method and scope

KinoWeave is an independent implementation inspired by
[cuArms](https://arxiv.org/abs/2610.36250) and
[LaCAM](https://arxiv.org/abs/2211.13432). Its current scope is joint-space
start-to-goal coordination, multi-arm manipulation and finite task-catalog search.
It does not reproduce cuArms' full manipulation
planning system or claim the paper's benchmark performance. See the
[FAQ](docs/faq.md#is-kinoweave-a-reproduction-of-cuarms) for the distinction.

Botrail, MuJoCo, Isaac Sim, and other execution platforms can connect through
separate [adapter contracts](docs/playback.md). The planner and viewer run
independently of these platforms.

## License

KinoWeave is licensed under [Apache-2.0](LICENSE). Third-party robot assets
retain their original licenses and notices; see [asset licensing](docs/third-party-assets.md).
