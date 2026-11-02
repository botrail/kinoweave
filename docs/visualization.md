# Standalone visualization

KinoWeave displays saved plans in a browser using Viser. All robots share
the native memory player's original trajectory clock, including waits and
retreats. No simulator or GPU is needed for viewing.

Install the viewer extra, or use the source checkout's locked Python 3.12
environment:

```bash
python -m pip install '.[viz]'
# Alternatively, from the source checkout:
uv sync --project environments/viz --frozen
```

The locked environment includes both visualization and URDF import. For a
pip installation that needs both, select `'.[viz,urdf]'`.

## Display existing JSON

After producing `problem.json` and `result.json` with the CLI:

```bash
kinoweave view /tmp/arms-run/problem.json /tmp/arms-run/result.json
```

Using the locked environment:

```bash
uv run --project environments/viz --frozen kinoweave view /tmp/arms-run/problem.json /tmp/arms-run/result.json
```

Open the URL printed to stderr, normally `http://127.0.0.1:8080`. Stop the
server with Ctrl+C. `--port`, `--fps`, and `--autoplay` control serving and
initial playback. The viewer independently revalidates the saved plan before
starting; `--max-samples` controls the validation resource limit.

The time slider seeks to arbitrary times. Play, Loop, Reset, and Speed control
the display clock. Speed changes do not modify the plan's timing or limits.
The inspector shows joint positions and velocities. Browser clients connected
to one viewer share the same timeline.

Without visual assets, the scene shows collision spheres, link skeletons,
goal outlines, tool-link trails, and static obstacles. Axes and geometry can
be toggled separately. A sibling `assets.manifest.json` adds hash-checked link
meshes; `--assets PATH` selects another manifest. Meshes are visual geometry;
collision checking uses spheres and obstacles saved in the native problem.

The server binds to loopback by default. `--host` selects another interface
when you intend to expose the viewer. A Python server is required; this
command does not produce an offline HTML player.

## Two UR5 arms

From the source checkout, fetch the pinned descriptions and meshes, then
compile, plan, and view the scene:

```bash
uv run --project environments/viz --frozen python examples/fetch_ur5.py
uv run --project environments/viz --frozen python examples/two_ur5.py --output /tmp/two-ur5 --view
```

Two instances of a UR5 description face one another. Their simultaneous direct
motions interfere. The coordinated plan moves alpha forward, retreats it, and
waits while beta passes, then finishes alpha's motion. The saved trajectory
lasts seven seconds. `--independent` uses wider base spacing for a simpler
simultaneous-motion example.

The description comes from a pinned revision of
[example-robot-data](https://github.com/Gepetto/example-robot-data/tree/6249cab1cdffa4fadb9a53dda964a50d79c5eaaf/robots/ur_description).
The explicit fetch checks upstream Git blob hashes. Import and replay do not
download assets. License and attribution documents travel with the compiled
visual bundle.

For GPU planning, import with the visualization environment, plan with the
[CUDA environment](curobo.md), then view the result:

```bash
uv run --project environments/viz --frozen kinoweave import-scene examples/scenes/two_ur5.yaml --output /tmp/ur5-scene
uv run --project environments/cuda12 --frozen kinoweave plan /tmp/ur5-scene/problem.json --backend curobo --samples 0 --output /tmp/ur5-gpu
uv run --project environments/viz --frozen kinoweave view /tmp/ur5-gpu/problem.json /tmp/ur5-gpu/result.json
```

`plan` copies the visual bundle and binds it to the effective input, including
the explicit backend override. Move a run by copying its whole directory.
Viewing that directory needs only `viz`, without the original URDF, YAML,
Torch, or cuRobo installation.

This is kinematic playback with sampled collision validation. The demo uses
explicit planning limits and proxies enclosing the source collision triangles.
Self-collision exclusions are declared in the robot asset. It does not
establish physical robot accuracy, continuous-time safety, mounting constraints,
contact behavior, or controller tracking. The recessed table is a static
planning obstacle.

## Python API

```python
from kinoweave import PlanningProblem, PlanResult
from kinoweave.io import load
from kinoweave.visualization import ViserViewer

problem = load("run/problem.json", expected=PlanningProblem)
plan = load("run/result.json", expected=PlanResult)
with ViserViewer(problem, plan) as viewer:
    viewer.run()
```

For meshes, call `kinoweave.assets.load_visuals(manifest_path, problem)` and
pass the returned mapping as `visuals=`. Its static GLB bytes use each link's
local frame; the viewer applies native world poses once. `viewer.seek(t)`
evaluates the original time law, and `viewer.frame` exposes the displayed frame.

See [scene authoring](scene-authoring.md) to define your own robots and workspace.
