# Changelog

## 0.1.0.dev0

- Experimental two/four/eight-tool inspection cases with shared or paired
  workspaces, and named scheduling phase/first-verified timings.
- Finite task catalogs with grounded robot/parameter alternatives, symbolic
  ordering, external producer feedback, bounded retries, shared PRMs/geometry,
  optional refinement, and verified best-plan histories with standalone playback.
- Explicit makespan, robot-completion, joint, tool and sphere-volume path
  metrics; clock-invariant FK sampling and auditable per-robot cost breakdowns.
- Bounded anytime single-/multi-arm shortcuts, shared-phase compression,
  greedy composition, CPU/CUDA filtering, independently verified incumbents,
  cancellation/budget retention, and recorded cost histories.
- Multi-arm manipulation task-progress search with ongoing skill clocks,
  atomic bimanual choices, eager bounded CPU/CUDA candidate conflict tables,
  PIBT-style inheritance, lazy constraints, and an exhaustive small-graph oracle.
- A verified two-arm Pick/Handover/Place demo with standalone object playback.
- Placed/held object FK, dual-grasp closure, scoped contacts and box-top
  support checks; CPU/CUDA mode masks over reusable static PRMs.
- Fixed-skeleton single-arm Pick & Place construction, independent CPU
  verification, and optional Viser object playback preserving skill clocks.
- Dependency-free manipulation input/result contracts with explicit object
  modes, configured skills, precedence, local traces, scoped contacts, and
  structural binding of unverified candidates. Motion-only JSON is unchanged.
- Native fixed-base robot problems, strict schema v1 JSON, explicit units,
  joint ordering, frames, and input/result identity.
- CPU quintic motion, analytic joint-rate checks, and sampled sphere/box
  collision validation.
- Optional cuRobo V2 geometry, tiled motion-pair evaluation, reusable PRMs,
  prioritized and LaCAM-style coordination, and finite product A*.
- SDK-free scene snapshots and kinematic playback with read-back reports.
- Optional Viser browser playback with shared-time seek, playback controls,
  collision overlays, link axes, and visual meshes.
- Optional URDF 1.0 and scene-YAML import with explicit planning limits,
  locked/mimic joints, declared sphere proxies, and portable visual bundles.
- A pinned two-UR5 demo with real meshes and coordinated retreat/wait motion.
- Installed commands for example creation, import, planning, validation,
  playback, and browser visualization.

This development release targets two fixed arms. Simulator SDK connections,
physical execution, arbitrary mesh collision, continuous collision guarantees,
and larger robot fleets are outside its supported scope.
