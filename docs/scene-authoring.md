# URDF robots and scene YAML

The optional importer combines an expanded URDF, explicit planning settings,
and scene YAML into a native `PlanningProblem`. The native JSON remains the
planner input. Simulator-specific assets and execution adapters can use the
same native contracts independently.

```bash
python -m pip install '.[urdf]'
kinoweave import-scene path/to/scene.yaml --output /tmp/scene
kinoweave plan /tmp/scene/problem.json --samples 0 --output /tmp/run
```

Use the `viz` extra to [view the result](visualization.md). Import produces
`problem.json`, a visual manifest, link-local GLBs, and supplied notices.
The manifest binds geometry to the problem and model hashes, with source
content hashes. Planning copies and checks the bundle, so saved runs can be
replayed without the authoring files.

## File layout and paths

```text
scene.yaml
robots/slider/
  robot.urdf
  robot.yaml
  collision_spheres.yaml
  meshes/...
```

Scene asset paths are relative to the scene file. URDF, proxy, notice, and
package-root paths in `robot.yaml` are relative to that file. Relative mesh
filenames resolve from the URDF directory. `package://NAME/path` requires an
explicit `packages` mapping and cannot escape that package root. Remote mesh
URIs are rejected. Import and replay do not fetch missing assets.

YAML fields are checked strictly. Duplicate keys, unknown fields, anchors,
aliases, nonfinite values, and incomplete joint states are rejected. Units
are metres, radians, and seconds. Authoring poses use `quaternion_xyzw`;
omitted poses are identity transforms.

## Robot asset

For a URDF with one bounded prismatic joint named `slide`:

```yaml
format: kinoweave.robot
schema_version: 1
name: slider
urdf: robot.urdf
packages:
  slider_description: .
joint_order: [slide]
joint_limits:
  slide: {velocity: 0.4, acceleration: 1.0}
locked_joints: {}
collision_spheres: collision_spheres.yaml
allowed_collisions: []
states:
  home: {slide: 0.0}
  work: {slide: 0.2}
notices: []
```

`joint_order` names every independent planning joint exactly once. Position
bounds and velocity limits come from the URDF; an overlay may tighten them.
Acceleration must be supplied explicitly for every moving joint, including
moving mimic joints. Values are never inferred from a nominal robot type.
Continuous joints remain unbounded, with unwrapped angles.

`locked_joints` fixes named independent joints at explicit source-valid
coordinates. Their motion is baked into fixed transforms, keeping all links,
geometry, and descendant frames. Dependent mimic coordinates propagate; lock
their independent source. At least one independent joint must remain.
Represent static bodies as obstacles.

Collision proxies are link-local spheres:

```yaml
spheres:
  - link: carriage
    center: [0.05, 0.0, 0.0]
    radius: 0.1
```

Every link with a URDF collision element needs an explicit proxy. The importer
checks coverage by link name; it does not certify geometric enclosure of
arbitrary source meshes. Prepare and verify conservative proxies for your
model. The included UR5 uses prepared spheres enclosing its source collision
triangles. Import does not automatically fit spheres.

`allowed_collisions` lists link-name pairs excluded from self-collision
checks. No adjacent pair is implicitly excluded. Review these exceptions for
your robot and proxies. `states` maps names to complete independent-joint
mappings. `notices` lists local license or attribution files to include in
portable bundles.

## Scene

```yaml
format: kinoweave.scene
schema_version: 1
name: two-sliders
robots:
  - name: alpha
    asset: robots/slider/robot.yaml
    base_pose:
      position: [-1.0, 0.0, 0.0]
      quaternion_xyzw: [0.0, 0.0, 0.0, 1.0]
    start: home
    goal: work
  - name: beta
    asset: robots/slider/robot.yaml
    base_pose:
      position: [1.0, 0.0, 0.0]
    start: {slide: 0.0}
    goal: {slide: 0.2}
obstacles:
  - name: table
    shape: {kind: box, size: [3.0, 1.0, 0.1]}
    pose:
      position: [0.0, 0.0, -0.2]
config:
  backend: reference
  device: cpu
  time_step: 2.0
  validation_time_step: 0.05
```

Instances can share one asset and use different base poses and tasks. Start
and goal accept named states or complete joint-name mappings. Obstacles are
`box` with full edge lengths in `size`, or `sphere` with `radius`, both with a
world pose. `config` uses existing [`PlannerConfig`](contracts.md) fields.
Optional `scene_revision` is a string.

## Supported URDF subset

The importer accepts expanded URDF 1.0 trees with fixed, revolute, continuous,
prismatic, and compatible mimic joints. It preserves joint origin/axis,
explicit fixed coordinates, visual origin, positive mesh scale, and internal
mesh transforms. Visual boxes, spheres, cylinders, local triangle meshes,
URDF RGBA materials, and embedded mesh materials are supported. External URDF
texture overrides are rejected; embed textures in mesh assets.

URDF 1.1/1.2, extended limit attributes, capsules, floating/planar joints,
unexpanded xacro, and unsafe XML entities are rejected. Unknown geometry is
not silently omitted. Inertial, dynamics, transmission, Gazebo, and control
declarations are source provenance, not physical execution behavior.

Import capability and planner capability differ. The CPU reference supports
continuous and mimic joints; the CUDA backend requires fixed or
bounded revolute/prismatic joints with signed coordinate axes. Unsupported
CUDA models produce an explicit failure. Visual meshes do not enable mesh
collision planning: the checker still uses declared link spheres and static
sphere/box obstacles.

See [the two-UR5 scene](../examples/scenes/two_ur5.yaml) and
[its robot settings](../examples/assets/ur5/robot.yaml) for a complete real-model
example. Botrail, MuJoCo, and Isaac Sim remain optional execution adapters;
this importer does not implement their native SDKs or model formats.
