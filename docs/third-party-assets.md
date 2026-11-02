# Licenses and third-party assets

KinoWeave source code and documentation are licensed under
[Apache-2.0](../LICENSE). See also [NOTICE](../NOTICE).

## UR5 demonstration

The optional demo fetches robot descriptions and original visual/collision
meshes from
[Gepetto/example-robot-data](https://github.com/Gepetto/example-robot-data/tree/6249cab1cdffa4fadb9a53dda964a50d79c5eaaf/robots/ur_description).
The source is pinned to commit `6249cab1cdffa4fadb9a53dda964a50d79c5eaaf`.
The upstream package identifies ROS-Industrial's
[`universal_robot/ur_description`](https://github.com/ros-industrial/universal_robot/tree/kinetic-devel/ur_description)
as its source and describes its license as BSD. The fetched repository
LICENSE contains the BSD 3-Clause terms, subject to any notices in individual
files.

These assets retain their original licensing. The explicit download includes
the upstream LICENSE and UR description README. Compiled visual bundles
carry these documents together with the demo's
[asset notice](../examples/assets/ur5/NOTICE.md). Keep the complete bundle,
including its notices, when sharing a run.

The distributed Python wheel does not contain the downloaded robot meshes.
The source distribution contains the demo settings, fetch helper and compiled
GLB copies for the standalone [eight-arm showcase](eight-arm-demo.md), with
their original license and attribution documents. See the
[eight-arm asset notice](../examples/assets/eight_ur5/NOTICE.md). It does not
bundle the downloaded source directory. See
[the two-arm walkthrough](visualization.md#two-ur5-arms) for acquisition and use.

## Dependencies and your own assets

Optional libraries such as cuRobo, Viser, yourdfpy, and trimesh retain their
own licenses. KinoWeave's license does not replace the licenses of those
packages or of robot/environment assets you supply.

When creating a robot asset, use its `notices` list to include applicable
license and attribution documents in portable visual bundles. The importer
checks and preserves supplied notice contents; you remain responsible for
choosing appropriate documents for your assets.
