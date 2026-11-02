# Command-line workflow

Both `kinoweave` and `python -m kinoweave` provide the same file-based interface.
No simulator SDK is required. All successful commands write one JSON document
to stdout; native SDK diagnostics go to stderr.

## Start with an included problem

```bash
kinoweave example --output /tmp/arms.json
kinoweave plan /tmp/arms.json --samples 0 --output /tmp/arms-run
kinoweave validate /tmp/arms-run/problem.json /tmp/arms-run/result.json
kinoweave playback /tmp/arms-run/problem.json /tmp/arms-run/result.json --step 0.25 --output /tmp/playback.json
```

The packaged example has two separated, noninterfering fixed arms with seven
and six independent joints. `--samples 0` checks direct endpoint connections,
including any common-tick subdivisions needed for joint-rate limits. It is a
small installation example, not a congested industrial cell.

For a retreat example on a supplied T-shaped graph, use the source checkout's
`examples/coordinated_planning.py` described in [planning](planning.md).

## Plan a saved problem

```bash
kinoweave plan problem.json --algorithm auto --samples 64 --neighbors 8 --output run
```

The CLI builds individual PRMs and calls `Planner`. The input contains models,
collision geometry, starts/goals, common timing, limits, seed, sampling conditions,
and the time budget. It uses the input backend and device unless an explicit
`--backend reference` or `--backend curobo` is supplied. Those overrides select
`cpu` or `cuda:0`, respectively.

The output directory always receives the effective `problem.json` and
`result.json`, including unsuccessful planning results. Use this saved input
for validation and playback: an override changes its digest. Output files with
the same names are replaced. Invalid input is rejected before creating a run.

| Option | Meaning |
| --- | --- |
| `--algorithm` | `lacam` (default), `prioritized`, `auto`, or `joint_astar`. |
| `--samples` | PRM sample count; default 128. |
| `--neighbors` | PRM neighbor count; default 12. |
| `--backend` | Explicitly override the saved problem's backend/device. |

For detailed search/resource settings, tiling, reusable planners, or supplied
graphs, use the [Python API](planning.md).

## Use CUDA explicitly

From the source checkout after installing the [CUDA environment](curobo.md):

```bash
uv run --project environments/cuda12 --frozen kinoweave plan problem.json --backend curobo --algorithm auto --samples 64 --output /tmp/gpu-run
```

The result records the backend actually used. Missing GPU dependencies or
inaccessible CUDA produce `unsupported`, with no CPU planning fallback.
You can copy the saved pair to a CPU-only environment for validation and
playback, even when its input configuration requests CUDA.

## Import and display

```bash
kinoweave import-scene scene.yaml --output /tmp/scene
kinoweave plan /tmp/scene/problem.json --samples 0 --output /tmp/run
kinoweave view /tmp/run/problem.json /tmp/run/result.json
```

`import-scene` requires the `urdf` extra; `view` requires `viz`. Import checks
the URDF, explicit planning settings, local resources, and scene YAML, then
saves the normalized problem and portable link visuals. Planning copies a
sibling `assets.manifest.json` and its resources to the result directory,
checking hashes and rebinding explicit backend overrides. Native JSON without
assets displays primitive collision geometry.

See [scene authoring](scene-authoring.md) for definitions and supported URDF
features, and [visualization](visualization.md) for browser controls, the
locked environment, and the two-UR5 example. The viewer URL is printed to
stderr; the command runs until Ctrl+C and then emits its closing JSON status.

## Revalidate and replay

`validate` checks input/result identity, declared collision conditions, producer
sampling, analytic joint limits, and CPU sampled collisions. It rejects a
producer's success label if independent revalidation fails. Its stdout is the
new validation report, not a modified plan.

`playback` performs the same acceptance checks, evaluates the original time law,
and emits a kinematic `ExecutionReport`. `--step` is the maximum nominal sample
spacing; recording includes all knots and the final time. Both commands accept
`--max-samples` (default 100,000). See [playback](playback.md) for clock and
observation semantics.

| Exit code | Meaning |
| --- | --- |
| 0 | The requested operation succeeded. |
| 1 | Planning completed with a non-success result; inspect the saved status and diagnostics. |
| 2 | Invalid arguments, input/validation failure, or a file error; inspect stderr. |

Use `kinoweave --help`, subcommand `--help`, and `kinoweave --version` to inspect
the installed interface.
