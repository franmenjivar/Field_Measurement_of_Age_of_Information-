"""Transport model between the aircraft and the UTM surveillance store.

Configurable update rate, latency, jitter and loss (research plan, section 12).
Everything stochastic draws from an injected generator so a run is reproducible
from its seed alone.

Two latency families are provided.  ``fixed``/``lognormal`` are the controlled
knobs used for the sweeps; ``empirical`` resamples a measured latency
distribution -- which for this project means the one recovered from BUBBLES,
letting an experiment be driven by a real link rather than an assumed one.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class TelemetryConfig:
    """Link and sampling configuration for one aircraft's telemetry stream."""

    update_rate_hz: float = 1.0
    latency_ms: float = 0.0
    jitter_ms: float = 0.0
    packet_loss_probability: float = 0.0
    processing_delay_ms: float = 0.0
    #: "fixed" | "lognormal" | "empirical"
    latency_model: str = "fixed"
    #: sample pool for latency_model="empirical", in seconds
    latency_samples_s: tuple[float, ...] = field(default=())
    #: stagger each aircraft's sampling phase instead of firing them in lockstep
    randomise_phase: bool = True
    #: allow a delayed message to be overtaken by a newer one
    allow_reordering: bool = True

    def __post_init__(self) -> None:
        if self.update_rate_hz <= 0:
            raise ValueError("update_rate_hz must be > 0")
        if not 0.0 <= self.packet_loss_probability < 1.0:
            raise ValueError("packet_loss_probability must be in [0, 1)")
        if self.latency_model == "empirical" and not self.latency_samples_s:
            raise ValueError("latency_model='empirical' needs latency_samples_s")
        if self.latency_model not in {"fixed", "lognormal", "empirical"}:
            raise ValueError(f"unknown latency_model {self.latency_model!r}")

    @property
    def interval_s(self) -> float:
        return 1.0 / self.update_rate_hz

    def label(self) -> str:
        return f"{self.update_rate_hz:g}Hz+{self.latency_ms:g}ms"


class NetworkModel:
    """Draws per-message end-to-end delay and loss decisions."""

    def __init__(self, config: TelemetryConfig, rng: np.random.Generator) -> None:
        self.config = config
        self.rng = rng
        if config.latency_model == "lognormal" and config.latency_ms > 0:
            # Parameterise so the *median* equals latency_ms and the spread is
            # set by jitter_ms; a heavy right tail is the realistic shape and it
            # is what makes P95/P99 AoI diverge from the mean.
            self._sigma = float(
                np.sqrt(np.log1p((config.jitter_ms / max(config.latency_ms, 1e-9)) ** 2))
            )
            self._mu = float(np.log(config.latency_ms / 1000.0))
        else:
            self._sigma = 0.0
            self._mu = 0.0

    def delay_s(self, n: int = 1) -> np.ndarray:
        """Draw ``n`` end-to-end delays in seconds (transport + processing)."""
        cfg = self.config
        if cfg.latency_model == "fixed":
            base = np.full(n, cfg.latency_ms / 1000.0)
            if cfg.jitter_ms > 0:
                base = base + self.rng.normal(0.0, cfg.jitter_ms / 1000.0, n)
        elif cfg.latency_model == "lognormal":
            base = (
                self.rng.lognormal(self._mu, self._sigma, n)
                if cfg.latency_ms > 0
                else np.zeros(n)
            )
        else:  # empirical
            pool = np.asarray(cfg.latency_samples_s, dtype=float)
            base = self.rng.choice(pool, size=n, replace=True)

        base = base + cfg.processing_delay_ms / 1000.0
        # A negative end-to-end delay is not physical; clip rather than reject so
        # the message count stays deterministic for a given seed.
        return np.maximum(base, 0.0)

    def is_lost(self, n: int = 1) -> np.ndarray:
        p = self.config.packet_loss_probability
        if p <= 0:
            return np.zeros(n, dtype=bool)
        return self.rng.random(n) < p
