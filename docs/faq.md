# Frequently asked questions

## Is KinoWeave a reproduction of cuArms?

KinoWeave is an independent implementation inspired by the reusable per-robot
roadmaps, GPU-batched conflict tables, and lazy coordinated search described
in [cuArms](https://arxiv.org/abs/2610.36250). Its lazy search also draws on
[LaCAM](https://arxiv.org/abs/2211.13432).

The current scope is fixed-base, joint-space start-to-goal coordination,
including waits and retreats. It does not implement cuArms' manipulation
modes, skill edges, schedule-skeleton planning, or anytime trajectory
refinement. It makes no claim of reproducing the paper's full algorithm,
benchmark results, or performance.

## Which parts run on the GPU?

With the `curobo` backend, forward kinematics and self/static geometry checks
use cuRobo V2. PyTorch evaluates batches of candidate motions and inter-robot
sphere conflicts on CUDA. Coordinated search runs on the CPU and looks up
reduced compatibility tables. Returned plans undergo independent CPU
validation. See [planning](planning.md) and [GPU interference](roadmaps.md).

## Can I use KinoWeave without a GPU?

Yes. Select `backend="reference", device="cpu"`, or pass `--backend reference`
to the CLI. The core can plan, validate, and record playback without optional
SDKs. GPU requests fail explicitly when CUDA is unavailable; they do not
silently switch to CPU planning.

## Can I view a saved GPU result on a CPU-only machine?

Yes. Copy the run directory and install the `viz` extra. The viewer accepts
the native input and result, independently revalidates them on the CPU, and
loads any accompanying visual assets. Original URDF files and GPU libraries
are not needed for replay. See [visualization](visualization.md).

## Does visualization execute the robots physically?

The viewer evaluates the saved trajectory's kinematics at a shared reference
time. It does not simulate dynamics, contacts, actuators, or controller
tracking. Collision validation uses the supplied sphere/box geometry at
sampled times. A displayed or validated plan does not establish
continuous-time collision safety or suitability for physical execution.

## Can I connect Botrail, MuJoCo, or Isaac Sim?

These are optional integration targets. The package defines separate scene,
playback, and physical-execution contracts; native SDK adapters for these
platforms are not included. The planner and viewer operate independently.
See [adapter contracts](playback.md).

## Why can a problem return no solution even when free space seems available?

Planning uses finite individual roadmaps and a shared tick interval. A route
may require samples or connections absent from those graphs. Position/rate
limits and collision margins can also reject a visually plausible motion.
`no_solution_in_discretization` refers to that finite problem. `timeout`
means a time or search-resource limit was reached. Inspect the saved status
and diagnostics before changing geometry or increasing resources.
