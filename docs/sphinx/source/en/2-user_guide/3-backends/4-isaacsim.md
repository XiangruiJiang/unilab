# IsaacSim Backend

UniLab's `isaacsim` backend runs PhysX through IsaacSim 6.0 and IsaacLab 3
(develop `4ecd0b036da1`) in a dedicated Python 3.12 worker process. The host
process keeps the regular `SimBackend` NumPy contract with the MuJoCo state
layout: shared memory carries `qpos`/`qvel`, body frames and contact-sensor data,
and pipe messages carry lifecycle commands.

## Runtime boundary

IsaacSim 6 requires Python 3.12, so it lives outside the UniLab environment. The
setup entry point is `scripts/tools/setup_isaacsim_env.sh`; it installs a venv and
the pinned IsaacLab tree under `$UNISIM_ISAACSIM_HOME` (default
`$HOME/.cache/unisim/isaacsim`) and runs IsaacLab's own installer, so the Isaac
Sim, torch and Newton versions pinned by that IsaacLab commit are used.

- `UNISIM_ISAACSIM_HOME` selects the runtime root.
- `UNISIM_ISAACSIM_PYTHON` overrides the worker interpreter path.
- `OMNI_KIT_ACCEPT_EULA=YES` keeps worker startup non-interactive (the adapter
  sets it for the worker).

The host needs the `isaacsim` extra of `unisim-core`, which adds MuJoCo.

## How a scene reaches PhysX

The MJCF scene stays the single source of truth:

1. The host compiles the scene (and every same-layout fixed variant) with
   MuJoCo and rejects features PhysX cannot express: equality constraints,
   tendons, flexes, mocap bodies, ball joints, joint springs, ellipsoid/height
   field geoms, geom margins, contact dimensions other than 1 or 3, and
   actuators that are not unit-gear position servos.
2. MuJoCo weld groups become articulation links. A body with several joints
   becomes a chain of links with small helper links, and a fixed-base actor gets
   an anchor link fixed to the world. Free single bodies become rigid objects.
3. Mass properties, joint frames and limits, armature, `frictionloss` (PhysX
   joint friction effort), `damping` (viscous joint friction), gravity
   compensation, colliders and collision filtering (parent/child, `<exclude>`,
   `contype`/`conaffinity`, explicit pairs) are derived from the compiled model.
4. MuJoCo resolves friction per geom pair; PhysX per material combine mode. The
   host chooses the combine-mode assignment with the smallest deviation over all
   collidable pairs and logs the pairs it cannot reproduce.

The worker authors one USD layer per variant, references it into every
environment and binds IsaacLab articulations and rigid objects.

## Supported capabilities

- Position-servo control, selected-row resets, fixed model variants
  (same layout).
- Reset randomization of body mass, center of mass, inertia, joint armature, geom
  friction (movable bodies) and kp/kd; interval body forces and torques at body
  centers of mass.
- `<contact>` sensors reported per rigid-body pair. A sensor whose geom
  selection is narrower than the rigid-body report fails closed at construction.
  After a reset the sensor rows read zero until the next physics step.
- Frame, gyro, velocimeter and joint sensors computed from the published state.

## Differences from MuJoCo

These are engine boundaries the adapter cannot remove; they are stated here
rather than hidden behind an approximation:

- **Friction.** MuJoCo resolves friction per geom pair (explicit pair, then
  priority, then maximum); PhysX resolves it per material combine mode. The
  compiler reports the pairs it cannot reproduce at construction.
- **Contact force.** Sensor forces carry the PhysX normal force plus the
  friction force of the same pair. PhysX anchors friction per body pair rather
  than per contact point, so a single-contact reduction (`mindist`, `maxforce`)
  resolves the pair's friction in the contact frame of the contact it selected.
  Contact magnitudes stay below MuJoCo's: from one identical state a fingertip
  pair read 0.44 N against MuJoCo's 0.58 N, because PhysX has no equivalent of
  `solref`/`solimp` contact compliance.
- **Reset.** Randomized body mass, center of mass and inertia land one physics
  step late: the first step after a reset still integrates with the values from
  the previous reset, and every later step uses the new ones. Contact sensors of
  reset rows read zero until the next physics step, and after a body is
  teleported PhysX can take up to two steps to report its contacts.
- **Solver.** PhysX TGS iteration counts come from the MJCF solver iteration
  budget; MuJoCo's `solref`/`solimp` contact compliance has no PhysX equivalent.

## Playback

The worker never renders. `record` and `interactive` playback replay physics
snapshots through the MuJoCo renderer, like `mjwarp`, including debug overlays
such as `ghost_model`:

```bash
uv run train --algo ppo --task g1_walk_flat --sim isaacsim
uv run eval --algo ppo --task g1_walk_flat --sim isaacsim --load-run <run-id> \
  --render-mode record training.play_steps=300
```
