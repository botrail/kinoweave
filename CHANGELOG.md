# Changelog

## 0.1.0.dev0

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
