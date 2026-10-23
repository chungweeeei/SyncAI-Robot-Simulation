# Findings about the navigation stack, from the simulator

What the sim turned up about the workspace's 3D navigation that is not a
property of the sim. Each entry says what was seen, why it is believed to be
the stack's, and what the sim did about it (if anything). Dates are run dates;
the workspace was `feature/octomap-planner-3d` at `4c95cdd`.

## 1. The planner and TerrainLayer disagree about thin map artefacts (2026-10-23)

**Seen.** On dp, the only route to the stairs (stair link kf 2208 → 2358) runs
along a 1 m corridor. The octomap holds a one-voxel-thick, 0.3–0.4 m tall line
across that corridor at x ≈ −7.5 m, y −15.1…−15.9, and the mapping track goes
straight through it. `Octo3D` plans through it (its footprint rule accepts a
disc that is 80 % supported, and a 0.1 m line is less than 20 % of the disc).
The sim's lidar rendered it, `TerrainLayer` marked it lethal (0.3 m is inside
its 0.10–0.60 m obstacle band), MPPI logged `Optimizer fail to compute path`
for 30 s and FollowPath aborted with `Failed to make progress`. The robot
stood 0.27 m from the line.

**Why the stack's.** Whether the line exists physically is unknown, but the
mapping robot walked through the space it occupies, so it is not solid at
body height. Two components of the stack judge the same voxels differently:
the planner routes through, the costmap forbids. On the robot this shows up
whenever a map keeps a sliver of something dynamic (a person who walked with
the mapper) — the lidar then sees nothing there and the run passes, but a
real thin object of that height would reproduce the deadlock exactly.

**Sim side.** `WorldOptions.track_clear_radius_m` erases what stands inside
the mapping robot's swept body volume along `poses.txt` (and
`clip_spikes_above_neighbours_m` the one-column posts that are its own feet),
so the sim's world agrees with what the mapper physically passed through.
The planner is unchanged; it still plans through the line. A possible
stack-side change: have `is_path_valid` (or the planner's footprint rule)
treat a step cell on the path that `TerrainLayer` would mark as blocking, so
the BT replans instead of MPPI stalling.

## 2. A FollowPath retry after `Failed to make progress` gets an empty plan (2026-10-23)

**Seen.** After the abort above, `move_3d.xml` cleared the local costmap and
retried FollowPath four times; every retry failed within 200 ms with
`Resulting plan has 0 poses in it.` and the navigation ended `FOLLOW_PATH_FAILED`
(100) without a replan.

**Why the stack's.** The retries reuse the path the planner produced at the
start (the tree replans only when `is_path_valid` says blocked, and the
planner's overlay does not mark static map geometry), and the controller's
path handler had already pruned the path up to the robot. The retry budget is
therefore spent in under a second. A replan between retries, or not pruning a
path that is about to be followed again, would make the four retries mean
something.

## 3. The ground floor of dp is a staircase of 0.1 m voxel steps (2026-10-23)

**Seen.** Not a stack defect, but worth knowing when reading sim results: the
mapped floor undulates by one voxel (dp's ground floor is at −0.4 m in one
room and −0.3 / −0.2 m in the corridor), so anything that compares a return
with a cell's surface sees 0.1 m jumps where the real floor is flat. The sim
places returns at the voxel's mid-height for surface voxels and keeps its
range noise at 1 cm (`lidar.py`) so those jumps do not read as dynamic
obstacles; the planner's overlay threshold (0.10 m) is exactly one voxel, and
with 2 cm noise it fired on flat floor in the sim.

**Cause, measured.** The undulation is a rigid tilt of the whole map, not
noise: a plane fit to the ground floor's top voxels (284 k columns) gives
z = −0.31 cm/m · x − 0.78 cm/m · y − 0.39, i.e. ~0.5°, with a residual of
10.8 cm — one voxel. The floor falls smoothly from −0.2 m at the south end to
−0.7 m at the north, in identical numbers in the raw `octomap.bt` and in the
sim's grid, so it is in the map. `syncai_mapping`'s README names the likely
origin: Point-LIO's gravity alignment runs at start and reset, and "doing that
in motion tilts the whole map without any error". The sim keeps the tilt on
purpose — the planner and the localizer read the same tilted map, and a
levelled sim world would disagree with them by up to 0.25 m at the site's
edges; a remap of dp with the robot standing still at start is the fix.

## 4. Stair mode climbs two flights; the turn at the next landing is where it stops (2026-10-23)

**Seen.** With the world cleaned of the mapper's own artefacts (findings 1 and
3, floor filled under the mapping track where the map never saw it, tread
holes bridged by the ground query), `move_3d.xml` planned the 49.5 m route
through the stair link and MPPI's stair mode did what it is written to do:
`Entering stair mode` 1 m before the flight → `set_policy_mode 1 sent` →
`In stair mode: policy settled` → `Stairs: turning in place -82 deg onto the
path` → climbed the first flight to the landing (1.2 m) → took the landing's
two corners (`turning in place -27 deg`, `-68 deg`) → climbed the second flight
to 2.4 m — all with zero collisions. At the next landing, at (-10.44, -12.35,
z 2.4), the body stood against the landing's wall and the controller logged
`Stairs: turning in place 44 deg onto the path` / `46 deg` alternately every
200-500 ms for 30 s without the heading changing, then `Failed to make
progress`; the retries hit finding 2.

**Why the stack's.** The sim allows a blocked body to rotate in place, so the
heading not changing means the commanded turn did not persist: the angle to
the path oscillated by 2-4 degrees between two values, which is the controller
re-deciding the turn direction each cycle, not the robot turning. Whether
that is the carrot point jumping between the landing's two segments, the
corner detection (`StairMode::findCorner()`), or the turn's own hysteresis
is for the controller log plus a `cmd_vel` trace to say; the sim reproduces
it in about 140 s (`nav_test.py --goal -8.94 -16.08 -1.12 --goal-z 5.11`
from the default spawn on dp), and `sim/status` shows the pose, the policy
and the refusal reason at each tick. The earlier run that stopped at the
first landing walked straight to the landing's edge instead; that one was
the sim's (holes in the map's landing, since filled), not the controller's.

**Sim side.** Nothing further pending. The things the sim had to learn on the
way — the one-voxel floor jitter, the mapper's own feet and the sliver of a
companion, the tread holes, the landing's one-voxel nosing, that a blocked
body still turns — are each a knob or rule with its reason next to it
(`WorldOptions`, `KinematicsConfig`).
