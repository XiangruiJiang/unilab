"""Persistent term snapshots stay separate from episode/reset buffers."""

from __future__ import annotations

import numpy as np
import pytest

from unilab.managers import ObservationGroupCfg, ObservationManager, ObservationTermCfg
from unilab.managers.manager_base import ManagerTermBase

from .conftest import FakeEnv


class _PersistentObservation(ManagerTermBase):
    def __init__(self, cfg, env):
        super().__init__(env)
        self.offset = cfg.params["offset"]

    def __call__(self, env, offset):
        return env.obs + self.offset

    def state_dict(self):
        return {"offset": self.offset}

    def load_state_dict(self, state):
        self.offset = state["offset"]


def test_observation_checkpoint_names_include_group(fake_env: FakeEnv):
    manager = ObservationManager(
        {
            group: ObservationGroupCfg(
                terms={
                    "same_name": ObservationTermCfg(
                        func=_PersistentObservation, params={"offset": offset}
                    )
                }
            )
            for group, offset in (("policy", 1), ("critic", 2))
        },
        fake_env,
    )
    assert manager.state_dict() == {
        "policy/same_name": {"offset": 1},
        "critic/same_name": {"offset": 2},
    }
    manager.load_state_dict({"policy/same_name": {"offset": 3}, "critic/same_name": {"offset": 4}})
    observations = manager.compute()
    np.testing.assert_array_equal(observations["policy"], fake_env.obs + 3)
    np.testing.assert_array_equal(observations["critic"], fake_env.obs + 4)
    with pytest.raises(ValueError, match="cannot restore checkpoint term"):
        manager.load_state_dict({"missing/name": {"offset": 0}})


def test_stateless_term_refuses_nonempty_state(fake_env: FakeEnv):
    term = ManagerTermBase(fake_env)
    assert term.state_dict() == {}
    term.load_state_dict({})
    with pytest.raises(ValueError, match="does not support checkpoint state"):
        term.load_state_dict({"unexpected": 1})
