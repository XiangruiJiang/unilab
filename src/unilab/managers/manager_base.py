# Derived from mujocolab/mjlab v1.6.0 (0fb8a681), src/mjlab/managers/manager_base.py.
# Copyright 2025, The mjlab Developers.
# Modified by UniLab for NumPy and UniLab contracts; licensed under Apache-2.0.
from __future__ import annotations

import abc
import inspect
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from unilab.base.config_overrides import (
    CONFIG_MAPPING_POLICY_KEY,
    MANAGER_PARAMS_MAPPING_POLICY,
)
from unilab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
    from unilab.managers._types import ManagerBasedRlEnv


@dataclass
class ManagerTermBaseCfg:
    """Base configuration for manager terms.

    This is the base config for terms in observation, reward, termination, curriculum,
    and event managers. It provides a common interface for specifying a callable
    and its parameters.

    The ``func`` field accepts either a function or a class:

    **Function-based terms** are simpler and suitable for stateless computations:

    .. code-block:: python

        RewardTermCfg(func=mdp.joint_torques_l2, weight=-0.01)

    **Class-based terms** are instantiated with ``(cfg, env)`` and useful when you need
    to:

    - Cache computed values at initialization (e.g., resolve regex patterns to indices)
    - Maintain state across calls
    - Perform expensive setup once rather than every call

    .. code-block:: python

        class posture:
          def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
            # Resolve std dict to tensor once at init
            self.std = resolve_std_to_tensor(cfg.params["std"], env)

          def __call__(self, env, **kwargs) -> np.ndarray:
            # Use cached self.std
            return compute_posture_reward(env, self.std)

        RewardTermCfg(func=posture, params={"std": {".*knee.*": 0.3}}, weight=1.0)

    Class-based terms can optionally implement ``reset(env_ids)`` for per-episode state.
    """

    func: Any
    """The callable that computes this term's value. Can be a function or a class.
  Classes are auto-instantiated with ``(cfg=term_cfg, env=env)``."""

    params: dict[str, Any] = field(
        default_factory=dict,
        metadata={CONFIG_MAPPING_POLICY_KEY: MANAGER_PARAMS_MAPPING_POLICY},
    )
    """Additional keyword arguments passed to func when called."""


class ManagerTermBase:
    def __init__(self, env: ManagerBasedRlEnv):
        self._env = env

    # Properties.

    @property
    def num_envs(self) -> int:
        return self._env.num_envs

    @property
    def name(self) -> str:
        return self.__class__.__name__

    # Methods.

    def reset(self, env_ids: np.ndarray | slice | None) -> Any:
        """Resets the manager term."""
        del env_ids  # Unused.
        pass

    def __call__(self, *args, **kwargs) -> Any:
        """Returns the value of the term required by the manager."""
        raise NotImplementedError

    def state_dict(self) -> dict[str, Any]:
        """Export persistent training state, such as curriculum or sampling history.

        Terms opt in by overriding this method and ``load_state_dict``. Do not
        include simulator handles or episode buffers: resume starts fresh episodes.
        Payloads must contain checkpoint-serializable CPU data. Validate task/data
        compatibility in ``load_state_dict`` before changing the term's state.
        """
        return {}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """Restore an opted-in term's persistent state. Stateless terms accept {}."""
        if state:
            raise ValueError(f"{type(self).__name__} does not support checkpoint state")


class ManagerBase(abc.ABC):
    """Base class for all managers."""

    def __init__(self, env: ManagerBasedRlEnv):
        self._env = env
        self._checkpoint_terms: dict[str, Any] = {}

        self._prepare_terms()

    # Properties.

    @property
    def num_envs(self) -> int:
        return self._env.num_envs

    @property
    @abc.abstractmethod
    def active_terms(self) -> list[str] | dict[Any, list[str]]:
        raise NotImplementedError

    # Methods.

    def reset(self, env_ids: np.ndarray) -> dict[str, Any]:
        """Resets the manager and returns logging info for the current step."""
        del env_ids  # Unused.
        return {}

    def state_dict(self) -> dict[str, Any]:
        """Snapshot persistent state from explicitly registered, stateful terms."""
        state = {}
        for name, term in self._checkpoint_terms.items():
            export = getattr(term, "state_dict", None)
            if export is None:
                continue
            payload = export()
            if not isinstance(payload, Mapping):
                raise TypeError(f"{type(self).__name__} term '{name}' state must be a mapping")
            if payload:
                if not callable(getattr(term, "load_state_dict", None)):
                    raise TypeError(f"{type(self).__name__} term '{name}' has no load_state_dict")
                state[name] = deepcopy(dict(payload))
        return state

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """Restore named terms; reject removed or incompatible checkpoint terms."""
        if not isinstance(state, Mapping):
            raise TypeError(f"{type(self).__name__} checkpoint state must be a mapping")
        for name, payload in state.items():
            term = self._checkpoint_terms.get(name)
            if term is None or not callable(getattr(term, "load_state_dict", None)):
                raise ValueError(f"{type(self).__name__} cannot restore checkpoint term '{name}'")
            if not isinstance(payload, Mapping):
                raise TypeError(f"{type(self).__name__} term '{name}' state must be a mapping")
        for name, payload in state.items():
            self._checkpoint_terms[name].load_state_dict(deepcopy(dict(payload)))

    def _check_term_shape(self, term_name: str, value: np.ndarray) -> None:
        if not isinstance(value, np.ndarray):
            manager_name = type(self).__name__
            raise TypeError(
                f"{manager_name} term '{term_name}' returned {type(value).__name__}, "
                "expected np.ndarray."
            )
        if value.shape != (self.num_envs,):
            manager_name = type(self).__name__
            raise ValueError(
                f"{manager_name} term '{term_name}' returned shape {tuple(value.shape)}, "
                f"expected ({self.num_envs},)."
            )

    def _check_term_finite(self, term_name: str, value: np.ndarray) -> None:
        if np.isfinite(value).all():
            return
        manager_name = type(self).__name__
        has_nan = np.isnan(value).any()
        has_inf = np.isinf(value).any()
        invalid_kind = "NaN/Inf" if has_nan and has_inf else "NaN" if has_nan else "Inf"
        env_ids = np.flatnonzero(~np.isfinite(value)).tolist()
        raise ValueError(
            f"{manager_name} term '{term_name}' returned {invalid_kind} for "
            f"environments {env_ids[:10]}."
        )

    def get_active_iterable_terms(self, env_idx: int) -> Sequence[tuple[str, Sequence[float]]]:
        raise NotImplementedError

    @abc.abstractmethod
    def _prepare_terms(self) -> None:
        raise NotImplementedError

    def _resolve_common_term_cfg(
        self,
        term_name: str,
        term_cfg: ManagerTermBaseCfg,
        *,
        checkpoint_name: str | None = None,
    ) -> None:
        for param_name, value in term_cfg.params.items():
            if isinstance(value, SceneEntityCfg):
                try:
                    value.resolve(self._env.scene)
                except (KeyError, TypeError, ValueError, NotImplementedError) as exc:
                    message = (
                        f"{type(self).__name__} term '{term_name}' parameter '{param_name}': {exc}"
                    )
                    raise type(exc)(message) from exc
        if inspect.isclass(term_cfg.func):
            term_cfg.func = cast(Any, term_cfg.func)(cfg=term_cfg, env=self._env)
        self._checkpoint_terms[checkpoint_name or term_name] = term_cfg.func
