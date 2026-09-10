"""Elevation for a region, draped on the map.

Built against USGS 3DEP rather than the rs-embed service on purpose: that service is the only
thing holding the deployment's Earth Engine credential — a personal Google account that has
expired twice — and elevation does not need it. 3DEP takes no key and no quota.

The failure this file exists for: 3DEP answers a request OUTSIDE its coverage with HTTP 200 and
a full frame of NoData. Drape that and the map gains an invisible layer while the run reports
success, which is worse than an error.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from agent_runtime import terrain_tools


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_FILE_STORAGE_ROOT", str(tmp_path))
    monkeypatch.setenv("AGENT_PUBLIC_BASE_URL", "")
    return tmp_path


def _tool():
    return {t.name: t for t in terrain_tools.make_terrain_tools()}["dem_for_region"]


def _geotiff(values: np.ndarray, bbox=(-88.28, 40.06, -88.16, 40.16)) -> bytes:
    """A real GeoTIFF, so the tool's rasterio path is exercised rather than mocked away.

    The CRS comes from terrain_tools._wgs84() rather than the string "EPSG:4326" because this
    suite poisons PROJ: importing qgis (test_qgis_headless_tools) repoints the PROJ already
    loaded in this process at QGIS's own proj.db, which is an older DATABASE.LAYOUT than
    rasterio's PROJ accepts. Written as a literal, these eleven tests passed alone and failed
    after that import — the same order-dependence, on a laptop, that the deployed container has
    permanently.
    """
    from rasterio.io import MemoryFile
    from rasterio.transform import from_bounds

    from agent_runtime.terrain_tools import _wgs84

    h, w = values.shape
    with MemoryFile() as mem:
        with mem.open(driver="GTiff", height=h, width=w, count=1, dtype="float32",
                      crs=_wgs84(), nodata=-999999.0,
                      transform=from_bounds(*bbox, w, h)) as dst:
            dst.write(values.astype("float32"), 1)
        return mem.read()


def _run(monkeypatch, values, **kw):
    monkeypatch.setattr(terrain_tools, "_fetch_dem", lambda bbox, size: _geotiff(values))
    return json.loads(_tool().func(bbox=[-88.28, 40.06, -88.16, 40.16], **kw))


def test_it_returns_metres_not_only_a_picture(store, monkeypatch):
    """A question about height is answered with numbers. A draped image alone cannot say how
    high anything is."""
    values = np.linspace(200.0, 260.0, 64 * 64).reshape(64, 64)
    out = _run(monkeypatch, values)

    assert out["ok"] is True
    assert out["min_m"] == 200.0 and out["max_m"] == 260.0
    assert out["relief_m"] == 60.0
    assert out["units"] == "metres above sea level"


def test_it_lands_on_the_map_as_a_georeferenced_raster(store, monkeypatch):
    out = _run(monkeypatch, np.full((32, 32), 210.0))

    layer = out["map_layer"]
    assert layer["render"] == "raster"
    assert layer["bounds"] == out["region_bbox"], "wrong bounds silently misregister the layer"
    assert layer["url"] and layer["id"]
    assert out["on_map"] is True


def test_it_leaves_the_real_metres_behind_as_a_geotiff(store, monkeypatch):
    """The PNG is a stretch between this region's own min and max, so it cannot be computed
    with. The GeoTIFF is what a later step reads."""
    out = _run(monkeypatch, np.full((16, 16), 187.5))

    from agent_runtime.file_store import resolve_file_id
    import rasterio

    with rasterio.open(resolve_file_id(out["geotiff"]["file_id"])) as src:
        assert src.crs is not None
        assert pytest.approx(float(src.read(1).max()), abs=0.01) == 187.5


def test_an_all_nodata_frame_is_refused_rather_than_draped(store, monkeypatch):
    """THE failure this tool has to get right. 3DEP answers outside its coverage with 200 and
    NoData everywhere; draping it reports success over an invisible layer."""
    out = _run(monkeypatch, np.full((32, 32), -999999.0))

    assert out["ok"] is False
    assert "NoData" in out["error"]
    assert "United States" in out["hint"], "the reader has to be told WHY it is empty"
    assert "map_layer" not in out


def test_nodata_is_transparent_not_the_bottom_of_the_ramp(store, monkeypatch):
    """The sea-level end of `terrain` is deep blue, so colouring holes with it draws water that
    is not there."""
    values = np.full((8, 8), 300.0)
    values[0, 0] = -999999.0
    out = _run(monkeypatch, values)

    from PIL import Image

    from agent_runtime.file_store import resolve_file_id
    img = np.asarray(Image.open(resolve_file_id(out["image"]["file_id"])).convert("RGBA"))
    assert img[0, 0, 3] == 0, "the nodata pixel must be transparent"
    assert img[4, 4, 3] == 255


def test_one_image_pixel_per_dem_cell(store, monkeypatch):
    """A draped layer is positioned by its bounds alone, so any margin, axis or colorbar shifts
    every pixel off the ground it describes."""
    out = _run(monkeypatch, np.full((24, 24), 200.0))

    from PIL import Image

    from agent_runtime.file_store import resolve_file_id
    assert Image.open(resolve_file_id(out["image"]["file_id"])).size == (24, 24)


def test_it_states_the_resolution_it_actually_got(store, monkeypatch):
    """3DEP's nominal 10 m only holds when the box is small enough; a 512-pixel frame over a
    state is a 2 km pixel, and an answer must not call that 10 m."""
    out = _run(monkeypatch, np.full((32, 32), 200.0), size=64)
    assert out["ground_resolution_m"] > 0
    # Illinois in one 64-pixel frame against a 10 km box in the same frame: the same nominal
    # source, two ground resolutions three orders of magnitude apart.
    state = terrain_tools._ground_resolution_m([-91.5, 37.0, -87.5, 42.5], 64)
    town = terrain_tools._ground_resolution_m([-88.28, 40.06, -88.16, 40.16], 64)
    assert state > 1000 > town


def test_an_arcgis_error_wearing_http_200_is_caught(monkeypatch):
    """ArcGIS returns its errors as JSON with a 200. Handing that to the TIFF reader raises
    something unrelated to the real problem.

    Patched through monkeypatch, not by assignment: a bare `urllib.request.urlopen = ...` is
    never put back, and this file did exactly that — every later test in the whole suite that
    opened a URL got this fake response. Eleven tests passed alone and seven failed together.
    """
    import urllib.request

    class _Resp:
        def read(self):
            return b'{"error":{"code":400,"message":"Unable to complete operation."}}'

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **k: _Resp())
    got = terrain_tools._fetch_dem([-88.3, 40.0, -88.2, 40.1], 64)
    assert isinstance(got, dict) and "3DEP returned an error" in got["error"]


def test_the_bbox_is_validated_before_any_request(store):
    out = json.loads(_tool().func(bbox=[-88.1, 40.2, -88.3, 40.1]))
    assert out["ok"] is False
    assert "inverted" in out["error"]


def test_the_tool_reaches_both_peers_without_an_upload(monkeypatch):
    """The region can be a bbox from the map or a boundary fetched the same turn, so gating it
    on an attached file would hide it from every request that names a place."""
    import agent_runtime.executor_factory as ef
    from agent_runtime.supervisor import graph as g

    for factory in (g.default_analyze_fn, g.default_code_fn):
        captured = {}

        def _fake(**kw):
            captured["names"] = [str(getattr(t, "name", ""))
                                 for t in (kw.get("preloaded_tools") or [])]
            raise RuntimeError("far enough")

        monkeypatch.setattr(ef, "build_agent_executor", _fake)
        try:
            factory(llm=object(), input_file_ids=None)("how high is it?", [],
                                                       {"query": "q", "thread_id": "t1"})
        except Exception:  # noqa: BLE001
            pass
        assert "dem_for_region" in captured.get("names", []), factory.__name__


def test_it_is_allowlisted_for_the_analysis_intent():
    from agent_runtime.graph_state import ANALYSIS_TOOL_NAMES
    from agent_runtime.tool_policy import select_allowed_tools

    assert "dem_for_region" in ANALYSIS_TOOL_NAMES
    assert "dem_for_region" in select_allowed_tools("analysis_task",
                                                    ["dem_for_region", "keyword_search"])
