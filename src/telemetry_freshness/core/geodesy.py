"""Geodetic <-> local ENU conversion.

Conflict geometry (CPA, TTC, separation) must be computed in metres, never in
degrees (research plan, section 34).  A single :class:`GeodeticFrame` per
scenario fixes the tangent-plane origin so that every aircraft in a scenario
shares one Cartesian frame.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from pyproj import Transformer

_WGS84 = "EPSG:4979"  # 3-D geographic, WGS-84


@dataclass(frozen=True)
class GeodeticFrame:
    """Local East-North-Up tangent frame anchored at a reference point.

    Uses a proper ellipsoidal transformation (pyproj topocentric operation),
    not a flat-earth approximation, so results stay valid over the multi-km
    extents of the BUBBLES and U-AGREE scenarios.
    """

    lat0_deg: float
    lon0_deg: float
    alt0_m: float = 0.0

    def __post_init__(self) -> None:
        # PROJ's topocentric operation consumes geocentric XYZ, so the pipeline
        # first converts geographic -> geocentric (+proj=cart), then to ENU.
        pipeline = (
            "+proj=pipeline "
            "+step +proj=unitconvert +xy_in=deg +xy_out=rad "
            "+step +proj=cart +ellps=WGS84 "
            f"+step +proj=topocentric +ellps=WGS84 "
            f"+lat_0={self.lat0_deg} +lon_0={self.lon0_deg} +h_0={self.alt0_m}"
        )
        object.__setattr__(self, "_fwd", Transformer.from_pipeline(pipeline))
        object.__setattr__(
            self, "_inv", Transformer.from_pipeline("+inv " + pipeline)
        )

    @classmethod
    def from_points(
        cls,
        lat_deg: np.ndarray,
        lon_deg: np.ndarray,
        alt_m: np.ndarray | float = 0.0,
    ) -> GeodeticFrame:
        """Anchor the frame at the median of a set of points.

        Median rather than mean so a few bad GPS fixes cannot drag the origin.
        """
        lat = np.asarray(lat_deg, dtype=float)
        lon = np.asarray(lon_deg, dtype=float)
        alt = np.asarray(alt_m, dtype=float)
        ok = np.isfinite(lat) & np.isfinite(lon)
        if not ok.any():
            raise ValueError("no finite lat/lon to anchor a frame on")
        a0 = float(np.nanmedian(alt[ok])) if alt.ndim else float(alt)
        return cls(float(np.median(lat[ok])), float(np.median(lon[ok])), a0)

    def to_enu(
        self,
        lat_deg: np.ndarray,
        lon_deg: np.ndarray,
        alt_m: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Geodetic -> local ENU metres."""
        e, n, u = self._fwd.transform(  # type: ignore[attr-defined]
            np.asarray(lon_deg, dtype=float),
            np.asarray(lat_deg, dtype=float),
            np.asarray(alt_m, dtype=float),
        )
        return np.asarray(e), np.asarray(n), np.asarray(u)

    def to_geodetic(
        self, x_m: np.ndarray, y_m: np.ndarray, z_m: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Local ENU metres -> geodetic (lat, lon, alt)."""
        lon, lat, alt = self._inv.transform(  # type: ignore[attr-defined]
            np.asarray(x_m, dtype=float),
            np.asarray(y_m, dtype=float),
            np.asarray(z_m, dtype=float),
        )
        return np.asarray(lat), np.asarray(lon), np.asarray(alt)

    def as_dict(self) -> dict[str, float]:
        return {
            "origin_lat_deg": self.lat0_deg,
            "origin_lon_deg": self.lon0_deg,
            "origin_alt_m": self.alt0_m,
        }


def track_angle_to_enu_velocity(
    speed_mps: np.ndarray, track_angle_deg: np.ndarray, vert_speed_mps: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Ground speed + track angle (deg CW from North) -> ENU velocity."""
    th = np.radians(np.asarray(track_angle_deg, dtype=float))
    s = np.asarray(speed_mps, dtype=float)
    return s * np.sin(th), s * np.cos(th), np.asarray(vert_speed_mps, dtype=float)


def ned_to_enu(
    vn: np.ndarray, ve: np.ndarray, vd: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """[North, East, Down] -> [East, North, Up]."""
    return np.asarray(ve), np.asarray(vn), -np.asarray(vd)
