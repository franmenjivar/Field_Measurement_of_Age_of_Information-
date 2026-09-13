"""Deterministic random-number plumbing.

Every stochastic component (network jitter, packet loss, Monte-Carlo geometry)
draws from a generator derived from a single run seed, so any run is exactly
reproducible from its config (research plan, section 21).
"""

from __future__ import annotations

import hashlib

import numpy as np


def _stream_id(name: str) -> int:
    """Stable 63-bit integer for a named substream (hash randomisation-proof)."""
    return int.from_bytes(hashlib.blake2b(name.encode(), digest_size=8).digest(), "big") >> 1


def make_rng(seed: int, stream: str = "") -> np.random.Generator:
    """Return an independent generator for ``stream`` under ``seed``.

    Different named streams under the same seed are statistically independent,
    so adding a new source of randomness never perturbs existing ones.
    """
    entropy = [int(seed)] + ([_stream_id(stream)] if stream else [])
    return np.random.default_rng(np.random.SeedSequence(entropy))


def spawn(rng: np.random.Generator, n: int) -> list[np.random.Generator]:
    """Split ``rng`` into ``n`` independent child generators (for joblib workers)."""
    return [np.random.default_rng(s) for s in rng.bit_generator.seed_seq.spawn(n)]  # type: ignore[attr-defined]
