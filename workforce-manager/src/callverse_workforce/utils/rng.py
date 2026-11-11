"""Deterministic, independent random-number streams.

Seed strategy
-------------
Every stream is a ``numpy.random.Generator`` built from
``SeedSequence(entropy=seed, spawn_key=(crc32(stream_name), [contact_index]))``.

* Global streams (``arrivals``, ``request_types``) are consumed only while
  generating the day's arrivals, before any routing happens.
* Per-contact streams (``patience``, ``service_times``, ``ai_success``, ...)
  are keyed by contact index, so a contact's draws do not depend on how many
  draws other contacts made, nor on the order of events.

Consequence: changing one routing action cannot change the arrival sequence or
any other contact's intrinsic randomness (common random numbers across policies).
"""

from __future__ import annotations

import zlib

import numpy as np


class Stream:
    """Canonical stream names."""

    ARRIVALS = "arrivals"
    REQUEST_TYPES = "request_types"
    CONTACT_ATTRIBUTES = "contact_attributes"
    PATIENCE = "patience"
    SERVICE_TIMES = "service_times"
    AI_SUCCESS = "ai_success"
    FALSE_AI_RESOLUTION = "false_ai_resolution"  # visible failure vs false resolution split
    FALSE_AI_DISCOVERY = "false_ai_discovery"  # delay until a false resolution surfaces
    CALLBACK = "callback"  # per-dial answer draws
    CALLBACK_DROPOUT = "callback_dropout"  # pending-customer dropout clock (B1.5)
    INTAKE_NOISE = "intake_noise"
    AI_FAILURE_HANGUP = "ai_failure_hangup"  # V1.1 immediate hangup after a visible AI failure
    AI_DEGRADATION_TIMING = "scenario_ai_degradation_timing"  # hidden day-level scenario timing
    DEMAND_SHOCK_TIMING = "scenario_demand_shock_timing"
    HUMAN_SHORTAGE_TIMING = "scenario_human_shortage_timing"
    RECONTACT = "recontact"  # keyed by the PARENT's rng_key: occurrence, delay and latent attributes


FUTURE_TAG = 0xF0F0  # marks counterfactual future-variant keys


def _name_key(name: str) -> int:
    return zlib.crc32(name.encode("utf-8"))


class RngManager:
    """Factory for named, reproducible RNG streams derived from one root seed."""

    def __init__(self, seed: int) -> None:
        if seed < 0:
            raise ValueError("seed must be non-negative")
        self.seed = int(seed)

    def stream(self, name: str) -> np.random.Generator:
        """A fresh generator for a global stream (same name -> same sequence)."""
        return np.random.Generator(np.random.PCG64(np.random.SeedSequence(self.seed, spawn_key=(_name_key(name),))))

    def contact_stream(
        self, name: str, contact_index: int, recontact_depth: int = 0, future_variant: int | None = None
    ) -> np.random.Generator:
        """A fresh generator dedicated to one contact instance and one purpose.

        Depth-0 (exogenous) keys are ``(name, index)``; a recontact of root ``index``
        uses ``(name, index, depth)``, which never collides with any other contact.
        Headroom counterfactual futures append ``(FUTURE_TAG, variant)``: draws made
        after a branch point use a variant-specific but action-independent stream.
        """
        key = (_name_key(name), int(contact_index))
        if recontact_depth or future_variant is not None:
            key += (int(recontact_depth),)
        if future_variant is not None:
            key += (FUTURE_TAG, int(future_variant))
        seq = np.random.SeedSequence(self.seed, spawn_key=key)
        return np.random.Generator(np.random.PCG64(seq))
