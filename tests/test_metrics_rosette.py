"""The rosette metric must find an unpainted pocket only where one exists.

The artifact: at a k-fold rotation centre, k tile copies pinwheel around one
vertex, so a flat unpainted patch at that UV corner is replicated k times and
the render's per-tile colourisation turns it into a hard-edged multicoloured
polygon. The metric measures how far that patch extends.
"""

from __future__ import annotations

import numpy as np
import pytest

from escher.geometry.spherical_kite_mesh import get_kite_mesh
from escher.metrics_rosette import WHITE_LEVEL, rosette_metrics


@pytest.fixture(scope="module")
def kite():
    return get_kite_mesh((2, 3, 4), n=12)


def corners_of(mesh):
    return {n: getattr(mesh, n) for n in ("cone1", "cone2a", "cone3", "cone2b")}


def painted_atlas(mesh, res=128, pocket_corner=None, pocket_deg=0.0):
    """A dark 'painted' atlas, optionally with a white pocket at one corner.

    Built in TEXTURE space by marking the texels of faces within ``pocket_deg``
    of the chosen corner, so the fixture and the metric agree by construction on
    where the pocket is without sharing any code path.
    """
    lum = np.full((res, res), 0.2, dtype=np.float64)
    if pocket_corner is None:
        return lum
    pts = np.asarray(mesh.points, dtype=np.float64)
    faces = np.asarray(mesh.faces)
    p = pts[getattr(mesh, pocket_corner)]
    p = p / np.linalg.norm(p)
    cent = pts[faces].mean(axis=1)
    cent /= np.linalg.norm(cent, axis=1, keepdims=True)
    ang = np.degrees(np.arccos(np.clip(cent @ p, -1.0, 1.0)))
    uvc = np.asarray(mesh.uv)[faces].mean(axis=1)
    for f in np.nonzero(ang <= pocket_deg)[0]:
        c = int(np.clip(uvc[f, 0] * res, 0, res - 1))
        r = int(np.clip(uvc[f, 1] * res, 0, res - 1))
        lum[max(r - 2, 0) : r + 3, max(c - 2, 0) : c + 3] = 1.0
    return lum


def test_fully_painted_tile_has_no_rosette(kite):
    m = rosette_metrics(
        kite.points, kite.faces, kite.uv, painted_atlas(kite), corners_of(kite)
    )
    assert m["rosette_radius_deg_max"] == 0.0
    assert m["white_fraction"] == 0.0


def test_pocket_is_found_at_the_right_corner(kite):
    m = rosette_metrics(
        kite.points,
        kite.faces,
        kite.uv,
        painted_atlas(kite, pocket_corner="cone3", pocket_deg=12.0),
        corners_of(kite),
    )
    assert m["rosette_radius_deg_cone3"] > 4.0, "the pocket must be detected"
    # cone2a/cone2b/cone1 are far from cone3 on this kite -- they must stay clean
    for other in ("cone2a", "cone2b"):
        assert m[f"rosette_radius_deg_{other}"] < m["rosette_radius_deg_cone3"]
    assert m["pocket_solid_angle_cone3"] > 0.0


def test_radius_grows_with_the_pocket(kite):
    small = rosette_metrics(
        kite.points, kite.faces, kite.uv,
        painted_atlas(kite, pocket_corner="cone3", pocket_deg=6.0), corners_of(kite),
    )
    big = rosette_metrics(
        kite.points, kite.faces, kite.uv,
        painted_atlas(kite, pocket_corner="cone3", pocket_deg=18.0), corners_of(kite),
    )
    assert big["rosette_radius_deg_cone3"] > small["rosette_radius_deg_cone3"]
    assert big["white_fraction"] > small["white_fraction"]


def test_an_all_white_tile_is_all_pocket(kite):
    lum = np.full((64, 64), 1.0)
    m = rosette_metrics(kite.points, kite.faces, kite.uv, lum, corners_of(kite))
    assert m["white_fraction"] == pytest.approx(1.0)
    # every corner's disc stays above the level however far it grows
    assert m["rosette_radius_deg_max"] > 20.0


def test_threshold_separates_painted_from_unpainted(kite):
    """A uniformly mid-grey tile is 'painted' -- the metric keys on WHITE_LEVEL,
    not on darkness, so a grey figure must not read as a pocket."""
    grey = np.full((64, 64), WHITE_LEVEL - 0.05)
    m = rosette_metrics(kite.points, kite.faces, kite.uv, grey, corners_of(kite))
    assert m["rosette_radius_deg_max"] == 0.0
    assert m["white_fraction"] == 0.0
