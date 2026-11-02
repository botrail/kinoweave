# UR5 demonstration asset

The robot description and original visual/collision meshes are fetched from
[Gepetto/example-robot-data](https://github.com/Gepetto/example-robot-data) at
commit `6249cab1cdffa4fadb9a53dda964a50d79c5eaaf`. That package identifies
ROS-Industrial's `universal_robot/ur_description` (kinetic-devel) as its upstream
and identifies the description license as BSD. The original repository LICENSE
and package README accompany compiled visual bundles.

The first ten allowed self-collision pairs are transcribed from the pinned
`robots/ur_description/srdf/ur5.srdf`. Three explicit wrist/ee_link pairs are
added for the URDF's fixed end-effector box at the wrist flange. These exclusions
are declared planning inputs, not a proof that excluded physical bodies never
collide. No general adjacent-link exclusion is applied by the importer.

KinoWeave's demo joint-rate limits are author-supplied planning settings, with
velocity caps below the URDF limits. They do not qualify a physical controller.
Collision spheres enclose the source collision triangles with 1 mm padding.
This approximation describes those source surfaces; neither visual-mesh
accuracy nor physical/continuous-time safety is certified.
