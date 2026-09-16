"""UniLab-facing contract tests for the IsaacSim subprocess adapter.

The adapter compiles the MJCF scene with MuJoCo on the host and only launches
the Python 3.12 IsaacSim/IsaacLab worker at materialization, so routing, runtime
discovery, cold metadata and playback planning are testable without Kit or a
GPU.  Physics coverage lives in UniSim's adapter tests and the runtime probes.
"""

from __future__ import annotations

import textwrap
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from unisim.backend.isaacsim.backend import IsaacSimBackend
from unisim.backend.isaacsim.dependencies import (
    ENV_HOME,
    ENV_PYTHON,
    IsaacSimDependencyError,
    build_worker_env,
    resolve_isaacsim_runtime,
)

from unilab.base.backend_factory import create_backend
from unilab.base.base import EnvCfg
from unilab.base.scene import SceneCfg

SIM_DT = 0.005
NUM_ENVS = 2

_SCENE_XML = """
<mujoco>
  <worldbody>
    <geom name="floor_geom" type="plane" size="0 0 0.05"/>
    <body name="base" pos="0 0 0.8">
      <freejoint/>
      <site name="imu_site"/>
      <geom name="base_geom" size="0.1" mass="1"/>
      <body name="link0">
        <joint name="j0" type="hinge" range="-1.5 1.5"/>
        <geom name="g0" size="0.1" mass="0.5"/>
        <body name="link1">
          <joint name="j1" type="hinge" range="-1.5 1.5"/>
          <geom name="foot_geom" size="0.1" mass="0.5"/>
        </body>
      </body>
    </body>
  </worldbody>
  <actuator>
    <position name="j0" joint="j0" kp="10" kv="0.5"/>
    <position name="j1" joint="j1" kp="20" kv="1.0"/>
  </actuator>
  <sensor>
    <gyro name="base_gyro" site="imu_site"/>
    <contact name="foot_contact" body1="link1" body2="world" data="found"/>
  </sensor>
  <keyframe>
    <key name="home" qpos="0 0 0.8 1 0 0 0 0.1 0.2"/>
  </keyframe>
</mujoco>
"""


@pytest.fixture()
def scene_file(tmp_path: Path) -> str:
    scene = tmp_path / "scene.xml"
    scene.write_text(textwrap.dedent(_SCENE_XML), encoding="utf-8")
    return str(scene)


def _make_backend(scene_file: str, **kwargs: Any) -> IsaacSimBackend:
    backend = create_backend(
        "isaacsim", SceneCfg(model_file=scene_file), NUM_ENVS, SIM_DT, base_name="base", **kwargs
    )
    assert isinstance(backend, IsaacSimBackend)
    return backend


def _make_runtime_tree(home: Path) -> None:
    python = home / "venv" / "bin" / "python"
    python.parent.mkdir(parents=True)
    python.write_text("#!/bin/sh\n", encoding="utf-8")
    (home / "venv" / "lib" / "python3.12" / "site-packages").mkdir(parents=True)
    (home / "IsaacLab" / "source").mkdir(parents=True)


def test_factory_routes_isaacsim_without_launching_the_worker(scene_file: str) -> None:
    backend = _make_backend(scene_file, isaacsim_device_id=0, isaacsim_worker_timeout_s=30.0)
    try:
        assert backend.backend_type == "isaacsim"
        assert backend._proc is None
        assert backend.get_actuator_names() == ("j0", "j1")
        np.testing.assert_allclose(
            backend.get_keyframe_qpos("home"), [0, 0, 0.8, 1, 0, 0, 0, 0.1, 0.2]
        )
        assert backend.get_joint_state_qpos_indices(["j0", "j1"]).tolist() == [7, 8]
        assert backend.get_root_state_layout("base").qvel_indices == tuple(range(6))
        assert backend.get_play_capabilities().supports_physics_state_playback
    finally:
        backend.close()


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"isaacsim_device_id": -1}, "isaacsim_device_id"),
        ({"isaacsim_worker_timeout_s": 0}, "isaacsim_worker_timeout_s"),
    ],
)
def test_env_cfg_rejects_invalid_isaacsim_settings(kwargs: dict[str, Any], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        EnvCfg(**kwargs).validate()


def test_dependencies_resolve_default_layout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(ENV_HOME, str(tmp_path))
    monkeypatch.delenv(ENV_PYTHON, raising=False)
    monkeypatch.delenv("OMNI_KIT_ACCEPT_EULA", raising=False)
    _make_runtime_tree(tmp_path)
    runtime = resolve_isaacsim_runtime()
    assert runtime.python == tmp_path / "venv" / "bin" / "python"
    assert runtime.package_path == tmp_path / "venv" / "lib" / "python3.12" / "site-packages"
    assert runtime.isaaclab_source == tmp_path / "IsaacLab" / "source"
    env = build_worker_env(runtime)
    assert env["PATH"].split(":")[0] == str(tmp_path / "venv" / "bin")
    assert env["OMNI_KIT_ACCEPT_EULA"] == "YES"


def test_dependencies_override_and_missing_layout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(ENV_HOME, str(tmp_path))
    _make_runtime_tree(tmp_path)
    custom = tmp_path / "custom-python"
    custom.write_text("#!/bin/sh\n", encoding="utf-8")
    monkeypatch.setenv(ENV_PYTHON, str(custom))
    assert resolve_isaacsim_runtime().python == custom

    monkeypatch.setenv(ENV_HOME, str(tmp_path / "empty"))
    monkeypatch.delenv(ENV_PYTHON, raising=False)
    with pytest.raises(IsaacSimDependencyError, match="Python 3.12 worker interpreter"):
        resolve_isaacsim_runtime()


def test_playback_plans_use_offline_snapshots(scene_file: str, tmp_path: Path) -> None:
    backend = _make_backend(scene_file)
    try:
        record = backend.resolve_play_render_plan(
            play_render_mode="record", play_steps=7, output_video=tmp_path / "play.mp4"
        )
        assert record.record_video and record.num_steps == 7
        assert (
            backend.resolve_play_render_plan(
                play_render_mode="none", play_steps=None, output_video=None
            ).mode
            == "none"
        )
        with pytest.raises(ValueError, match="play_steps"):
            backend.resolve_play_render_plan(
                play_render_mode="record", play_steps=None, output_video=tmp_path / "play.mp4"
            )
        with pytest.raises(NotImplementedError, match="offline MuJoCo"):
            backend.init_renderer()
    finally:
        backend.close()
