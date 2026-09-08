"""Zonal embedding aggregation: the parts that must be right without touching Earth Engine."""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from agent_runtime.rs_embed_zonal_worker import _to_lonlat, _to_merc  # noqa: E402


def test_mercator_roundtrip_is_exact_enough_to_place_a_pixel():
    for lon, lat in [(-87.65, 41.93), (121.5, 31.2), (-0.1, 51.5), (0.0, 0.0)]:
        x, y = _to_merc(lon, lat)
        lon2, lat2 = _to_lonlat(x, y)
        # 1e-9 degrees is ~0.1 mm; a 10 m pixel needs nothing like this precision.
        assert abs(lon - lon2) < 1e-9 and abs(lat - lat2) < 1e-9


def test_mercator_metres_are_not_ground_metres():
    """The whole reason the affine is derived in 3857: a 'metre' here is 1/cos(lat) long.
    Reading scale_m as ground metres predicted a 170x111 grid where 224x146 came back."""
    lat = 40.07
    x0, y0 = _to_merc(-88.00, lat)
    x1, _ = _to_merc(-87.98, lat)
    span_3857 = x1 - x0
    span_ground = 0.02 * 111320 * math.cos(math.radians(lat))
    assert span_3857 / span_ground == pytest.approx(1 / math.cos(math.radians(lat)), rel=0.01)


def test_unknown_zone_id_field_lists_the_real_columns(tmp_path):
    """A silent fallback to row numbers made the vectors key on 0..n while the label join
    expected geoids, and the two never met. The caller must be told."""
    import geopandas as gpd
    from shapely.geometry import box

    from agent_runtime.rs_embed_zonal_worker import run

    gdf = gpd.GeoDataFrame({"geoid": ["a", "b"], "geometry": [box(0, 0, 1, 1), box(1, 0, 2, 1)]},
                           crs="EPSG:4326")
    p = tmp_path / "z.geojson"
    gdf.to_file(p, driver="GeoJSON")

    out = run({"polygons_path": str(p), "zone_id_field": "GEOID", "model": "gse"})
    assert out["ok"] is False
    assert "GEOID" in out["error"]
    assert "geoid" in out["available_columns"], "must name the column that DOES exist"


def test_sums_and_counts_roll_up_exactly_where_means_would_not():
    """Why the worker stores sum+pixels rather than only a mean: a mean of means is wrong
    across unequal zones, which is the whole reason cross-scale aggregation is possible."""
    zones = [{"sum": [10.0, 20.0], "pixels": 100}, {"sum": [30.0, 30.0], "pixels": 400}]
    total_sum = [sum(z["sum"][i] for z in zones) for i in range(2)]
    total_px = sum(z["pixels"] for z in zones)
    rolled = [v / total_px for v in total_sum]

    mean_of_means = [sum(z["sum"][i] / z["pixels"] for z in zones) / 2 for i in range(2)]
    assert rolled == pytest.approx([0.08, 0.1])
    assert mean_of_means != pytest.approx(rolled), "the naive average is a different number"


def test_zonal_tools_are_registered_with_their_map_layer_contract():
    from agent_runtime.rs_embed_tools import make_rs_embed_zonal_tools
    from agent_runtime.supervisor.graph import _MAP_LAYER_TOOLS

    names = {t.name for t in make_rs_embed_zonal_tools()}
    assert names == {"embed_zones", "fit_zone_model"}
    # Both deliver layers, so a turn that ran them has genuinely put something on the map.
    assert names <= set(_MAP_LAYER_TOOLS)


def _two_squares(tmp_path):
    """A tiny polygon layer on disk — run_zonal_worker reads before it calls the service."""
    import geopandas as gpd
    from shapely.geometry import box

    gdf = gpd.GeoDataFrame(
        {"geoid": ["a", "b", "c"], "spare": [1, 2, 3],
         "geometry": [box(0, 0, 1, 1), box(1, 0, 2, 1), box(2, 0, 3, 1)]},
        crs="EPSG:4326")
    p = tmp_path / "zones.geojson"
    gdf.to_file(p, driver="GeoJSON")
    return p


def _service_reply(covered=("a", "b"), dims=4):
    """What /api/zones sends back: every zone, plus the provenance in meta."""
    rows = []
    for i, zid in enumerate(["a", "b", "c"]):
        row = {"zone_id": zid, "pixels": 100 + i if zid in covered else 0,
               "area_km2": 1.5 + i}
        if zid in covered:
            row.update({f"e{d:03d}": float(i + d) for d in range(dims)})
        rows.append(row)
    return {"ok": True, "model": "gse", "year": 2022, "zones": 3, "dim": dims,
            "rows": rows,
            "meta": {"model": "gse", "dims": dims, "bands": ["A0"], "scale_m": 10.0,
                     "pixel_ground_m": 7.44, "tile_px": 256, "tiles_planned": 4,
                     "tiles_fetched": 2, "tiles_capped": True, "zone_id_field": "geoid",
                     "zones_total": 3, "zones_with_pixels": len(covered),
                     "tile_errors": [], "pixel_size_warnings": []}}


def test_unreachable_zones_service_is_reported_with_the_fix_not_swallowed(tmp_path, monkeypatch):
    """The old failure said "the rs_embed runtime is unavailable" and stopped there, because
    the interpreter was looked up by a path that existed on one laptop. Whatever goes wrong
    now, the caller must be told which service and what to do about it."""
    import agent_runtime.rs_embed_tools as T

    monkeypatch.setattr(T, "_svc", lambda *a, **k: {
        "error": "the rs-embed service is not reachable at http://localhost:8077",
        "hint": "Start it, or set RS_EMBED_URL to where it runs."})
    out = T.run_zonal_worker({"polygons_path": str(_two_squares(tmp_path)),
                              "zone_id_field": "geoid"})
    assert out["ok"] is False
    assert "http://localhost:8077" in out["error"]
    assert "RS_EMBED_URL" in out["hint"], "the next action must survive the hand-off"


def test_service_rows_and_meta_become_the_result_the_tools_read(tmp_path, monkeypatch):
    """embed_zones indexes res["scale_m"], res["tiles_fetched"] and friends directly, so a
    missing meta key is a KeyError in the middle of a delivered turn, not a soft failure."""
    import agent_runtime.rs_embed_tools as T

    monkeypatch.setattr(T, "_svc", lambda *a, **k: _service_reply())
    out = T.run_zonal_worker({"polygons_path": str(_two_squares(tmp_path)),
                              "zone_id_field": "geoid", "clusters": 2})

    assert out["ok"] is True
    for key in ("scale_m", "pixel_ground_m", "tiles_planned", "tiles_fetched", "dims",
                "zones_total", "zones_with_pixels", "bands", "tile_errors",
                "pixel_size_warnings"):
        assert key in out, f"embed_zones reads {key!r} off this result"
    assert out["dims"] == 4 and out["scale_m"] == 10.0

    # An uncovered zone is REPORTED, not dropped: "40 zones, 31 embedded" and "31 zones" are
    # different answers, and only the first says the sweep was short.
    assert out["zones_total"] == 3 and out["zones_with_pixels"] == 2
    assert [z["zone_id"] for z in out["zones"]] == ["a", "b", "c"]
    uncovered = out["zones"][2]
    assert uncovered["pixels"] == 0 and uncovered["mean"] is None and "group" not in uncovered

    covered = [z for z in out["zones"] if z["pixels"]]
    assert all(len(z["mean"]) == 4 for z in covered)
    assert covered[0]["mean"] == [0.0, 1.0, 2.0, 3.0], "values must keep dimension order"
    assert sorted(z["group"] for z in covered) == [1, 2], "groups are 1-based for the map"
    assert out["image"]["error"], "the pixel raster is not produced on this route — say so"


def test_only_the_identifier_travels_with_the_geometry(tmp_path, monkeypatch):
    """A tract layer carries forty attribute columns. Re-encoding them into the request body
    costs bandwidth and ships attributes the service has no use for."""
    import agent_runtime.rs_embed_tools as T

    sent = {}

    def _capture(path, body=None, **kwargs):
        sent["path"], sent["body"] = path, body
        return _service_reply()

    monkeypatch.setattr(T, "_svc", _capture)
    T.run_zonal_worker({"polygons_path": str(_two_squares(tmp_path)),
                        "zone_id_field": "geoid", "clusters": 2})
    assert sent["path"] == "/api/zones"
    props = sent["body"]["zones_geojson"]["features"][0]["properties"]
    assert set(props) == {"geoid"}, f"only the id should travel, got {sorted(props)}"


def test_unknown_zone_field_is_refused_before_the_service_is_called(tmp_path, monkeypatch):
    """Naming a column that does not exist used to reach the service and come back as a
    stack trace. The layer is in hand here, so the real columns can be listed."""
    import agent_runtime.rs_embed_tools as T

    called = []
    monkeypatch.setattr(T, "_svc", lambda *a, **k: called.append(1) or _service_reply())
    out = T.run_zonal_worker({"polygons_path": str(_two_squares(tmp_path)),
                              "zone_id_field": "GEOID"})
    assert out["ok"] is False and "GEOID" in out["error"]
    assert "geoid" in out["available_columns"]
    assert not called, "no point paying for a request that cannot succeed"


def test_dimension_columns_are_ordered_by_number_not_by_name():
    """e000..e063 sorts the same either way, which is exactly why this is worth pinning: a
    rename to unpadded names would transpose dimensions silently, and nothing downstream
    could tell. The first row may also be an uncovered zone, which carries no e-columns."""
    from agent_runtime.rs_embed_tools import _dimension_keys

    rows = [{"zone_id": "a", "pixels": 0},
            {"zone_id": "b", "pixels": 5, "e0": 1, "e9": 2, "e10": 3, "e2": 4}]
    assert _dimension_keys(rows) == ["e0", "e2", "e9", "e10"]
    assert _dimension_keys([{"zone_id": "a", "pixels": 0}]) == []


def test_look_alike_groups_are_shared_with_the_standalone_worker():
    """One clustering implementation, not two: the service path and the local sweep must put
    the same zones in the same group, or a rerun through the other route relabels the map."""
    from agent_runtime.rs_embed_zonal_worker import assign_look_alike_groups

    zones = [{"pixels": 10, "mean": [1.0, 0.0]}, {"pixels": 10, "mean": [0.98, 0.02]},
             {"pixels": 10, "mean": [0.0, 1.0]}, {"pixels": 0, "mean": None}]
    n = assign_look_alike_groups(zones, 2)
    assert n == 2
    assert zones[0]["group"] == zones[1]["group"], "near-identical vectors group together"
    assert zones[2]["group"] != zones[0]["group"]
    assert "group" not in zones[3], "a zone with no pixels has no vector to group by"
    assert min(z["group"] for z in zones[:3]) == 1, "1-based: the map reads 0 as ungrouped"


def test_numpy_stand_ins_reproduce_scikit_learn(tmp_path):
    """scikit-learn cannot be imported where this code runs — its k-means segfaults a process
    that has torch loaded, taking the worker down rather than failing a call — so ridge, the
    two fold splitters and k-means were rewritten in numpy. That is only defensible if the
    numbers are the same, so scikit-learn runs HERE in a fresh subprocess, where torch is
    absent and it is safe, and the two are compared.
    """
    import subprocess

    import numpy as np

    from agent_runtime.rs_embed_zonal_worker import (group_kfold_indices, kfold_indices,
                                                     kmeans_labels, ridge_loo_predict)

    rng = np.random.default_rng(7)
    n, p_dim = 60, 8
    X = rng.normal(size=(n, p_dim))
    y = X @ rng.normal(size=p_dim) + rng.normal(scale=0.3, size=n)
    groups = np.repeat(np.arange(5), 12)
    blobs = np.repeat([[0.0, 0.0], [30.0, 0.0], [0.0, 30.0]], 12, axis=0) \
        + rng.normal(scale=0.4, size=(36, 2))
    alphas = (0.01, 0.1, 1.0, 10.0, 100.0, 1000.0)

    script = """
import json, sys
import numpy as np
from sklearn.cluster import KMeans
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GroupKFold, KFold
d = json.load(open(sys.argv[1]))
X, y = np.array(d["X"]), np.array(d["y"])
groups, blobs = np.array(d["groups"]), np.array(d["blobs"])
alphas = tuple(d["alphas"])
n = len(X)
pred = np.full(n, np.nan)
for tr, te in GroupKFold(n_splits=5).split(X, y, groups):
    pred[te] = RidgeCV(alphas=alphas).fit(X[tr], y[tr]).predict(X[te])
lab = KMeans(n_clusters=3, n_init=10, random_state=0).fit_predict(blobs)
json.dump({
    "ridge_oof": pred.tolist(),
    "kfold": [sorted(int(i) for i in te) for _, te in KFold(5, shuffle=True, random_state=0).split(X)],
    "groupkfold": [sorted(int(i) for i in te) for _, te in GroupKFold(n_splits=5).split(X, y, groups)],
    "kmeans": sorted(sorted(int(i) for i in np.flatnonzero(lab == c)) for c in set(lab)),
}, open(d["out"], "w"))
"""
    payload = tmp_path / "in.json"
    result = tmp_path / "out.json"
    payload.write_text(json.dumps({"X": X.tolist(), "y": y.tolist(),
                                   "groups": groups.tolist(), "blobs": blobs.tolist(),
                                   "alphas": list(alphas), "out": str(result)}))
    proc = subprocess.run([sys.executable, "-c", script, str(payload)],
                          capture_output=True, text=True, timeout=300)
    if proc.returncode != 0:
        pytest.skip(f"scikit-learn unavailable for the comparison: {proc.stderr[-200:]}")
    ref = json.loads(result.read_text())

    ours = np.full(n, np.nan)
    for tr, te in group_kfold_indices(groups, 5):
        ours[te] = ridge_loo_predict(X[tr], y[tr], X[te], alphas)[0]
    assert np.allclose(ours, np.array(ref["ridge_oof"]), atol=1e-8), \
        "out-of-fold ridge predictions must match RidgeCV's, or the reported r2 moves"

    assert [sorted(int(i) for i in te) for _, te in kfold_indices(n, 5, seed=0)] == ref["kfold"]
    assert [sorted(int(i) for i in te)
            for _, te in group_kfold_indices(groups, 5)] == ref["groupkfold"]

    lab = kmeans_labels(blobs, 3)
    ours_partition = sorted(sorted(int(i) for i in np.flatnonzero(lab == c)) for c in set(lab))
    assert ours_partition == ref["kmeans"], "the same zones must land in the same group"


def test_one_covered_zone_still_lands_on_the_map():
    """"Embed this one polygon" is the commonest form of the request, and the group layer is
    the only thing embed_zones puts on the map. Asking for five groups across one zone used to
    tag nothing, so the turn delivered a CSV, an empty map, and no explanation."""
    from agent_runtime.rs_embed_zonal_worker import assign_look_alike_groups

    zones = [{"pixels": 8431, "mean": [0.1, 0.2]}, {"pixels": 0, "mean": None}]
    assert assign_look_alike_groups(zones, 5) == 1
    assert zones[0]["group"] == 1
    assert "group" not in zones[1]


def test_clusters_is_a_ceiling_not_a_requirement():
    """Five groups across three zones is three groups, not zero."""
    from agent_runtime.rs_embed_zonal_worker import assign_look_alike_groups

    zones = [{"pixels": 10, "mean": [1.0, 0.0]}, {"pixels": 10, "mean": [0.0, 1.0]},
             {"pixels": 10, "mean": [-1.0, 0.0]}]
    assert assign_look_alike_groups(zones, 5) == 3
    assert sorted(z["group"] for z in zones) == [1, 2, 3]
    assert assign_look_alike_groups([{"pixels": 10, "mean": [1.0]}], 0) == 0, \
        "clusters < 2 still means 'do not group'"


def test_a_zone_id_that_cannot_key_uniquely_is_refused_with_the_alternative(tmp_path, monkeypatch):
    """The id is the only thing joining a vector back to a polygon — on the map, in the CSV,
    and in fit_zone_model's merge. A duplicated value does not fail: it paints one zone's
    vector onto every polygon sharing it, and puts one feature row on both sides of a CV
    split. The column is in hand here, so this is catchable before anything is paid for."""
    import geopandas as gpd
    from shapely.geometry import box

    import agent_runtime.rs_embed_tools as T

    called = []
    monkeypatch.setattr(T, "_svc", lambda *a, **k: called.append(1) or _service_reply())

    def _write(values):
        gdf = gpd.GeoDataFrame({"name": values,
                                "geometry": [box(i, 0, i + 1, 1) for i in range(len(values))]},
                               crs="EPSG:4326")
        p = tmp_path / f"z{len(called)}_{abs(hash(tuple(map(str, values))))}.geojson"
        gdf.to_file(p, driver="GeoJSON")
        return str(p)

    out = T.run_zonal_worker({"polygons_path": _write(["a", "a", "b"]), "zone_id_field": "name"})
    assert out["ok"] is False and "uniquely" in out["error"]
    assert "row number" in out["hint"], "the failure must name what to do instead"

    out = T.run_zonal_worker({"polygons_path": _write(["a", None, "b"]), "zone_id_field": "name"})
    assert out["ok"] is False and "no value" in out["error"]
    assert "row number" in out["hint"]
    assert not called, "neither layer can produce a usable answer, so neither is sent"


def test_numeric_zone_ids_travel_as_text(tmp_path, monkeypatch):
    """A GEOID is a number to pandas. Sent as one it comes back as 17031836500.0, which
    str()-matches no polygon on the way home, and the zone silently vanishes from the map."""
    import geopandas as gpd
    from shapely.geometry import box

    import agent_runtime.rs_embed_tools as T

    gdf = gpd.GeoDataFrame({"geoid": [17031836500, 17031836600],
                            "geometry": [box(0, 0, 1, 1), box(1, 0, 2, 1)]}, crs="EPSG:4326")
    path = tmp_path / "numeric.geojson"
    gdf.to_file(path, driver="GeoJSON")

    sent = {}
    monkeypatch.setattr(T, "_svc", lambda path_, body=None, **k: (sent.update(body=body)
                                                                  or _service_reply()))
    T.run_zonal_worker({"polygons_path": str(path), "zone_id_field": "geoid", "clusters": 2})
    values = [f["properties"]["geoid"] for f in sent["body"]["zones_geojson"]["features"]]
    assert values == ["17031836500", "17031836600"], f"ids must be sent as text, got {values}"


def test_both_cross_validation_scores_use_the_same_fold_count(tmp_path, monkeypatch):
    """fit reports the blocked r2 and the r2 a naive random split WOULD have claimed, and the
    difference decides whether the answer says the skill was really adjacency. Scoring them
    over different fold counts moves that difference on its own."""
    import geopandas as gpd
    import numpy as np
    from shapely.geometry import box

    import agent_runtime.rs_embed_zonal_worker as W

    n, dims = 24, 6
    rng = np.random.default_rng(3)
    feats = rng.normal(size=(n, dims))
    labels = feats[:, 0] * 2 + rng.normal(scale=0.2, size=n)
    csv = tmp_path / "vectors.csv"
    header = ["zone_id", "pixels", "area_km2"] + [f"e{i:03d}" for i in range(dims)]
    rows = [",".join(header)]
    for i in range(n):
        rows.append(",".join([str(i), "100", "1.0"] + [f"{v:.6f}" for v in feats[i]]))
    csv.write_text("\n".join(rows) + "\n")

    gdf = gpd.GeoDataFrame({"zid": [str(i) for i in range(n)], "truth": labels,
                            "geometry": [box(i % 6, i // 6, i % 6 + 1, i // 6 + 1)
                                         for i in range(n)]}, crs="EPSG:4326")
    poly = tmp_path / "poly.geojson"
    gdf.to_file(poly, driver="GeoJSON")

    seen = []
    real_kfold = W.kfold_indices
    monkeypatch.setattr(W, "kfold_indices",
                        lambda n_, splits, seed=0: seen.append(splits) or real_kfold(n_, splits, seed))
    # Three spatial blocks come back where five were asked for — what happens whenever the
    # zones sit at fewer distinct locations than the requested block count.
    real_group = W.group_kfold_indices
    monkeypatch.setattr(W, "group_kfold_indices",
                        lambda groups, splits: real_group(groups, splits)[:3])

    out = W.fit({"vectors_csv": str(csv), "polygons_path": str(poly), "label_column": "truth",
                 "zone_id_field": "zid", "blocks": 5, "out_geojson": str(tmp_path / "out.geojson")})
    assert out["ok"] is True
    assert out["spatial_block_cv"]["blocks"] == 3, "report the folds that actually ran"
    assert seen == [3], f"the naive split must use the same fold count, got {seen}"


def test_embed_zones_puts_a_single_polygon_on_the_map(tmp_path, monkeypatch):
    """The whole tool, end to end over a stubbed service, for the request it gets most: one
    polygon, by geoid. It must come back with a vectors CSV AND a map layer — nothing below
    the tool is allowed to decide that one zone is not worth mapping."""
    import geopandas as gpd
    from shapely.geometry import box

    import agent_runtime.file_store as FS
    import agent_runtime.langchain_geo_tools as G
    import agent_runtime.rs_embed_tools as T

    gdf = gpd.GeoDataFrame({"geoid": ["17031836500"], "geometry": [box(-87.61, 41.82, -87.60, 41.83)]},
                           crs="EPSG:4326")
    src = tmp_path / "one.geojson"
    gdf.to_file(src, driver="GeoJSON")

    monkeypatch.setattr(G, "_stage_vector_source", lambda *a, **k: (str(src), None))
    monkeypatch.setattr(G, "_index_attached", lambda *a, **k: {})
    monkeypatch.setattr(FS, "create_output_file_from_path",
                        lambda p, filename=None: {"file_id": f"id_{filename}", "filename": filename,
                                                  "download_url": f"/files/{filename}",
                                                  "size_bytes": 10})
    monkeypatch.setattr(T, "_svc", lambda *a, **k: {
        "ok": True, "model": "gse", "year": 2022, "zones": 1, "dim": 3,
        "rows": [{"zone_id": "17031836500", "pixels": 8431, "area_km2": 0.468075,
                  "e000": 0.1, "e001": -0.2, "e002": 0.3}],
        "meta": {"model": "gse", "dims": 3, "bands": [], "scale_m": 10.0,
                 "pixel_ground_m": 7.44, "tiles_planned": 1, "tiles_fetched": 1,
                 "tiles_capped": False, "zone_id_field": "geoid", "zones_total": 1,
                 "zones_with_pixels": 1, "tile_errors": [], "pixel_size_warnings": []}})

    tool = {t.name: t for t in T.make_rs_embed_zonal_tools()}["embed_zones"]
    out = json.loads(tool.func(file_id="file_x", zone_id_field="geoid", model="gse"))

    assert out["ok"] is True and out["zones_with_pixels"] == 1
    assert out["vectors_csv"]["file_id"], "the CSV of per-zone vectors is the ML artifact"
    assert out.get("on_map") is True, "one polygon is still a map"
    layer = out["map_layer"]
    assert layer["count"] == 1 and layer["url"]
    assert "cluster_note" not in out, "nothing went wrong, so nothing to apologise for"


def _stage_layer(tmp_path, geoms, ids, name="layer.geojson"):
    import geopandas as gpd

    from agent_runtime.file_store import create_output_file_from_path

    gdf = gpd.GeoDataFrame({"geoid": ids}, geometry=geoms, crs="EPSG:4326")
    p = tmp_path / name
    gdf.to_file(p, driver="GeoJSON")
    return create_output_file_from_path(p, filename=name)["file_id"]


def test_embed_region_declines_a_polygon_layer_and_names_embed_zones(tmp_path):
    """Observed: "get the embeddings for the area with geoid 17031330100" called BOTH
    embed_zones and embed_region, and reported embed_region — a bbox tool that accepts a
    file_id looks like the direct answer. It embedded the tract's bounding box; the tract
    fills 69% of it, so 46% of what was embedded lay outside, mostly Lake Michigan."""
    import json

    from shapely.geometry import Polygon

    from agent_runtime.rs_embed_tools import make_rs_embed_tools

    # An L-shape fills well under its bounding box, which is the point.
    poly = Polygon([(0, 0), (2, 0), (2, 1), (1, 1), (1, 2), (0, 2)])
    fid = _stage_layer(tmp_path, [poly], ["z0"])
    tools = {t.name: t for t in make_rs_embed_tools()}
    out = json.loads(tools["embed_region"].func(file_id=fid, models=["gse"]))

    assert out["ok"] is False
    assert out["use_instead"] == "embed_zones"
    assert "RECTANGLE" in out["error"] and "%" in out["error"]
    assert out["bounding_box"], "the box is still reported for a deliberate rectangle"


def test_the_redirect_still_leaves_a_way_to_get_a_pixel_image(tmp_path):
    """embed_zones returns vectors, not a picture, so refusing the polygon must not close the
    only image route — the hint has to say how to still get one, and that it is the rectangle."""
    import json

    from shapely.geometry import Polygon

    from agent_runtime.rs_embed_tools import make_rs_embed_tools

    fid = _stage_layer(tmp_path, [Polygon([(0, 0), (2, 0), (2, 1), (1, 1), (1, 2), (0, 2)])],
                       ["z0"])
    tools = {t.name: t for t in make_rs_embed_tools()}
    hint = json.loads(tools["embed_region"].func(file_id=fid, models=["gse"]))["hint"]

    assert "bbox=" in hint, "must name the parameter that still yields an image"
    assert "rectangle" in hint.lower(), "and say the image is wider than the shape"


def test_an_explicit_bbox_is_still_honoured(monkeypatch):
    """The model-name check must be reached, i.e. the region check did not block first.

    The service is stubbed. embed_region calls /api/models BEFORE validating the name, so
    without a stub this asserted on a connection error instead of the check under test — and
    passed only on a machine that happened to be running rs-embed on localhost:8077. Sixteen
    of this file's tests already stub `_svc`; this one was reaching the network.
    """
    import json

    import agent_runtime.rs_embed_tools as T
    from agent_runtime.rs_embed_tools import make_rs_embed_tools

    monkeypatch.setattr(T, "_svc", lambda *a, **k: {
        "models": [{"id": "gse", "type": "precomputed"}]})

    tools = {t.name: t for t in make_rs_embed_tools()}
    out = json.loads(tools["embed_region"].func(bbox=[-87.63, 41.84, -87.60, 41.87],
                                                models=["definitely-not-a-model"]))
    assert "unknown model" in out["error"], "the region check must be past, not blocking"


def test_a_point_layer_may_still_use_its_extent(tmp_path):
    """A point cloud has no shape to respect, so its extent is a fair reading of the area."""
    from shapely.geometry import Point

    from agent_runtime.rs_embed_tools import _resolve_bbox

    fid = _stage_layer(tmp_path, [Point(0, 0), Point(1, 1)], ["a", "b"], name="pts.geojson")
    got = _resolve_bbox(None, None, None, fid, 2048.0, polygon_extent_ok=False)
    assert isinstance(got, list) and len(got) == 4


def _tiny_png_data_uri(w=4, h=3):
    import base64
    import io

    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGBA", (w, h), (10, 120, 90, 255)).save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def test_the_pixel_image_asked_for_and_carried_back(tmp_path, monkeypatch):
    """A 64-number average does not answer "what does this area look like to the model" — the
    picture does, and it was the half that went missing when the sweep moved behind the
    service. The request must ask for it and the result must carry it."""
    import agent_runtime.rs_embed_tools as T

    sent = {}
    reply = _service_reply()
    reply["image"] = {"png": _tiny_png_data_uri(), "bounds": [-87.61, 41.82, -87.60, 41.83],
                      "size_px": [3, 4], "pixels_shown": 7, "colour": "PCA to RGB"}
    monkeypatch.setattr(T, "_svc", lambda p_, body=None, **k: sent.update(body=body) or reply)

    out = T.run_zonal_worker({"polygons_path": str(_two_squares(tmp_path)),
                              "zone_id_field": "geoid", "clusters": 2})
    assert sent["body"]["image"] is True, "the picture has to be requested to arrive"
    assert out["image"]["png"].startswith("data:image/png;base64,")
    assert out["image"]["bounds"] == [-87.61, 41.82, -87.60, 41.83]


def test_embed_zones_delivers_the_picture_and_the_groups(tmp_path, monkeypatch):
    """Both views, as before the service move: the pixels the vectors came from, and the zones
    grouped by those vectors. Delivering only the group colour hid the actual data."""
    import geopandas as gpd
    from shapely.geometry import box

    import agent_runtime.file_store as FS
    import agent_runtime.langchain_geo_tools as G
    import agent_runtime.rs_embed_tools as T

    gdf = gpd.GeoDataFrame({"geoid": ["a", "b"],
                            "geometry": [box(-87.61, 41.82, -87.605, 41.83),
                                         box(-87.605, 41.82, -87.60, 41.83)]}, crs="EPSG:4326")
    src = tmp_path / "two.geojson"
    gdf.to_file(src, driver="GeoJSON")

    monkeypatch.setattr(G, "_stage_vector_source", lambda *a, **k: (str(src), None))
    monkeypatch.setattr(G, "_index_attached", lambda *a, **k: {})
    monkeypatch.setattr(FS, "create_output_file_from_path",
                        lambda p, filename=None: {"file_id": f"id_{filename}", "filename": filename,
                                                  "download_url": f"/files/{filename}",
                                                  "size_bytes": 10})
    reply = {"ok": True, "model": "gse", "year": 2022, "zones": 2, "dim": 3,
             "rows": [{"zone_id": "a", "pixels": 100, "area_km2": 1.0,
                       "e000": 1.0, "e001": 0.0, "e002": 0.0},
                      {"zone_id": "b", "pixels": 120, "area_km2": 1.1,
                       "e000": 0.0, "e001": 1.0, "e002": 0.0}],
             "image": {"png": _tiny_png_data_uri(), "bounds": [-87.61, 41.82, -87.60, 41.83],
                       "size_px": [3, 4], "pixels_shown": 220, "colour": "PCA to RGB"},
             "meta": {"model": "gse", "dims": 3, "bands": [], "scale_m": 10.0,
                      "pixel_ground_m": 7.44, "tiles_planned": 1, "tiles_fetched": 1,
                      "tiles_capped": False, "zone_id_field": "geoid", "zones_total": 2,
                      "zones_with_pixels": 2, "tile_errors": [], "pixel_size_warnings": []}}
    monkeypatch.setattr(T, "_svc", lambda *a, **k: reply)

    tool = {t.name: t for t in T.make_rs_embed_zonal_tools()}["embed_zones"]
    out = json.loads(tool.func(file_id="file_x", zone_id_field="geoid", model="gse"))

    assert out["ok"] is True and out["on_map"] is True
    kinds = [layer["render"] for layer in out["map_layers"]]
    assert kinds == ["raster", "categories"], f"both views must be delivered, got {kinds}"
    assert out["map_layers"][0]["bounds"] == [-87.61, 41.82, -87.60, 41.83]
    assert out["pixel_image"]["pixels_shown"] == 220
    assert "image_note" not in out


def test_a_declined_picture_says_why_and_what_would_get_one(tmp_path, monkeypatch):
    """Too large to render is a legitimate answer — but the vectors still stand, and an
    unexplained missing picture reads as a failed analysis."""
    import agent_runtime.rs_embed_tools as T

    reply = _service_reply()
    reply["image"] = {"error": "the zones span about 25,800,000 pixels at 10 m, past the "
                               "4,000,000 this route will request in one call",
                      "hint": "The per-zone vectors are unaffected. Embed fewer zones."}
    monkeypatch.setattr(T, "_svc", lambda *a, **k: reply)
    out = T.run_zonal_worker({"polygons_path": str(_two_squares(tmp_path)),
                              "zone_id_field": "geoid", "clusters": 2})
    assert out["ok"] is True, "the vectors are the answer; the picture is extra"
    assert "25,800,000" in out["image"]["error"] and out["image"]["hint"]


def test_zone_ids_selects_without_carving_up_the_file_first(tmp_path, monkeypatch):
    """Asked for one tract out of 801, the model extracted it into a new file with four
    execute_code steps and then embedded that. Naming the ids is the same answer in one step,
    and only the named zones may reach the service — the sweep is bounded by their extent."""
    import agent_runtime.rs_embed_tools as T

    sent = {}
    monkeypatch.setattr(T, "_svc", lambda p_, body=None, **k: sent.update(body=body)
                        or {"ok": True, "rows": [{"zone_id": "b", "pixels": 9, "area_km2": 1.0,
                                                  "e000": 0.5}],
                            "meta": {"dims": 1, "zones_total": 1, "zones_with_pixels": 1}})
    out = T.run_zonal_worker({"polygons_path": str(_two_squares(tmp_path)),
                              "zone_id_field": "geoid", "zone_ids": ["b"], "clusters": 2})
    props = [f["properties"]["geoid"] for f in sent["body"]["zones_geojson"]["features"]]
    assert props == ["b"], f"only the requested zone should be sent, got {props}"
    assert out["ok"] is True


def test_zone_ids_that_match_nothing_say_so_rather_than_embedding_everything(tmp_path, monkeypatch):
    """Silently falling back to the whole layer would bill an 801-tract sweep for a typo."""
    import agent_runtime.rs_embed_tools as T

    called = []
    monkeypatch.setattr(T, "_svc", lambda *a, **k: called.append(1) or _service_reply())

    out = T.run_zonal_worker({"polygons_path": str(_two_squares(tmp_path)),
                              "zone_id_field": "geoid", "zone_ids": ["nope"]})
    assert out["ok"] is False and "zone_ids" in out["error"]
    assert "drop zone_ids" in out["hint"]

    out = T.run_zonal_worker({"polygons_path": str(_two_squares(tmp_path)),
                              "zone_ids": ["a"]})
    assert out["ok"] is False and "zone_id_field" in out["error"]
    assert not called, "neither request could succeed, so neither was sent"


def test_zone_ids_partly_missing_are_named_not_quietly_dropped(tmp_path, monkeypatch):
    """"I asked for three and got two" is exactly what goes unnoticed."""
    import agent_runtime.rs_embed_tools as T

    monkeypatch.setattr(T, "_svc", lambda *a, **k: {
        "ok": True, "rows": [{"zone_id": "a", "pixels": 9, "area_km2": 1.0, "e000": 0.5}],
        "meta": {"dims": 1, "zones_total": 1, "zones_with_pixels": 1}})
    out = T.run_zonal_worker({"polygons_path": str(_two_squares(tmp_path)),
                              "zone_id_field": "geoid", "zone_ids": ["a", "ghost"]})
    assert out["ok"] is True
    assert out["zone_ids_not_found"] == ["ghost"]


def test_a_single_zone_is_labelled_for_what_it_is_not_as_a_cluster_of_one(tmp_path, monkeypatch):
    """"gse zone groups (k=1)" is a cluster analysis of one thing, with a one-entry legend
    explaining nothing — and it did not read as the area that was asked about, so the model
    added a second layer of the same polygon beside it."""
    import geopandas as gpd
    from shapely.geometry import box

    import agent_runtime.file_store as FS
    import agent_runtime.langchain_geo_tools as G
    import agent_runtime.rs_embed_tools as T

    gdf = gpd.GeoDataFrame({"geoid": ["17031330100"],
                            "geometry": [box(-87.62, 41.85, -87.61, 41.86)]}, crs="EPSG:4326")
    src = tmp_path / "one.geojson"
    gdf.to_file(src, driver="GeoJSON")
    monkeypatch.setattr(G, "_stage_vector_source", lambda *a, **k: (str(src), None))
    monkeypatch.setattr(G, "_index_attached", lambda *a, **k: {})
    monkeypatch.setattr(FS, "create_output_file_from_path",
                        lambda p, filename=None: {"file_id": f"id_{filename}", "filename": filename,
                                                  "download_url": f"/files/{filename}",
                                                  "size_bytes": 10})
    monkeypatch.setattr(T, "_svc", lambda *a, **k: {
        "ok": True, "rows": [{"zone_id": "17031330100", "pixels": 50451, "area_km2": 2.793586,
                              "e000": 0.1, "e001": 0.2}],
        "meta": {"dims": 2, "zones_total": 1, "zones_with_pixels": 1, "scale_m": 10.0,
                 "pixel_ground_m": 7.45, "tiles_planned": 1, "tiles_fetched": 1}})

    tool = {t.name: t for t in T.make_rs_embed_zonal_tools()}["embed_zones"]
    out = json.loads(tool.func(file_id="f", zone_id_field="geoid", model="gse"))

    layer = out["map_layer"]
    assert layer["render"] == "shapes", "one zone is not a categorical map"
    assert "17031330100" in layer["label"] and "k=" not in layer["label"]
    assert "legend" not in layer, "a legend of one class explains nothing"
    # It is drawn over the pixel image of the same polygon. Filled, it covers the picture it
    # is framing — observed as a solid violet slab where the embedding had been.
    assert layer["outline"] is True


# --- provenance ------------------------------------------------------------
# The service sends its embedder's meta verbatim, with a comment saying it "carries the
# provenance a caller must not invent". These pin that it survives the last hop too.

_TERRAMIND_META = {
    "model": "terramind", "type": "on_the_fly", "backend": "gee",
    "source": "COPERNICUS/S2_SR_HARMONIZED",
    "sensor": {"collection": "COPERNICUS/S2_SR_HARMONIZED",
               "bands": ["B1", "B2", "B3"],
               "bands_terramind": ["COASTAL_AEROSOL", "BLUE", "GREEN"],
               "scale_m": 10, "cloudy_pct": 30, "composite": "median", "fill_value": 0.0},
    "temporal": {"mode": "range", "start": "2019-06-01", "end": "2019-08-31"},
    "image_size": 224, "model_key": "terramind_v1_small", "modality": "S2L2A",
    "normalization": "zscore", "device": "cuda", "pretrained": True, "batch_infer": True,
    "input_override": True, "param_mean": -0.000222, "param_std": 0.0522,
    "param_absmax": 0.2624, "tokens_shape": [196, 384],
    "batch_tokens_shape": [64, 196, 384], "layer_index": -1, "tokens_include_cls": False,
    "grid_type": "vit_patch_tokens", "grid_hw": [26, 26], "grid_shape": [26, 26],
    "cls_removed": False, "grid_orientation_policy": "north_up",
    "grid_native_y_axis_direction": "unknown",
    "grid_native_orientation_reason": "no orientation metadata",
    "grid_orientation_applied": False, "y_axis_direction": "unknown",
}


def test_provenance_keeps_what_changes_a_numbers_meaning():
    from agent_runtime.rs_embed_tools import _provenance

    prov = _provenance(_TERRAMIND_META)
    # Which imagery, at what resolution, composited how, over which dates — the questions a
    # user actually asks after seeing an embedding.
    assert prov["collection"] == "COPERNICUS/S2_SR_HARMONIZED"
    assert prov["scale_m"] == 10 and prov["composite"] == "median" and prov["cloudy_pct"] == 30
    assert prov["date_range"] == "2019-06-01..2019-08-31" and prov["temporal_mode"] == "range"
    assert prov["bands"] == ["B1", "B2", "B3"]
    # Which model, exactly — "terramind" alone does not identify the run.
    assert prov["model_key"] == "terramind_v1_small" and prov["modality"] == "S2L2A"
    assert prov["normalization"] == "zscore" and prov["layer_index"] == -1
    assert prov["image_size"] == 224 and prov["grid_hw"] == [26, 26]


def test_provenance_drops_the_embedders_own_diagnostics():
    """These answer nothing a user asks and would crowd the context; the band aliases are the
    same list twice under model-side names."""
    from agent_runtime.rs_embed_tools import _provenance

    prov = _provenance(_TERRAMIND_META)
    for noise in ("param_mean", "param_std", "param_absmax", "device", "batch_infer",
                  "batch_tokens_shape", "tokens_shape", "input_override", "cls_removed",
                  "bands_terramind", "grid_shape", "grid_native_y_axis_direction"):
        assert noise not in prov, f"{noise} is debugging, not provenance"


def test_provenance_raises_the_orientation_caveat_the_numbers_cannot_show():
    from agent_runtime.rs_embed_tools import _provenance

    assert "north-up is assumed, not verified" in _provenance(_TERRAMIND_META)["orientation_caveat"]
    # …and stays quiet when the grid WAS oriented, so the note means something when it appears.
    oriented = dict(_TERRAMIND_META, grid_orientation_applied=True, y_axis_direction="north_up")
    assert "orientation_caveat" not in _provenance(oriented)


def test_no_meta_yields_no_provenance_rather_than_a_shell_of_nulls():
    """'The run had no provenance' and 'this deployment does not send it' are different
    facts, and a dict of nulls would make the second look like the first."""
    from agent_runtime.rs_embed_tools import _provenance

    assert _provenance(None) == {} and _provenance({}) == {} and _provenance("nope") == {}
    assert _provenance({"model": "gse", "device": "cuda"}) == {"model": "gse"}


def test_embed_zones_passes_provenance_through(tmp_path, monkeypatch):
    """It already had the full meta in hand and was whitelisting ~13 keys out of it."""
    import agent_runtime.rs_embed_tools as T

    reply = _service_reply()
    reply["meta"] = dict(reply["meta"], **_TERRAMIND_META)
    monkeypatch.setattr(T, "_svc", lambda *a, **k: reply)
    out = T.run_zonal_worker({"polygons_path": str(_two_squares(tmp_path)),
                              "zone_id_field": "geoid"})
    prov = out["provenance"]
    assert prov["model_key"] == "terramind_v1_small"
    assert prov["collection"] == "COPERNICUS/S2_SR_HARMONIZED"
    assert prov["date_range"] == "2019-06-01..2019-08-31"
    # The long-standing top-level fields stay put — existing callers read them.
    assert out["scale_m"] == 10 and out["dims"] == 4


def _embed_region(monkeypatch, embed_reply):
    import agent_runtime.rs_embed_tools as T

    def _svc(path, payload=None, **kw):
        if path == "/api/models":
            return {"models": [{"id": "terramind", "type": "onthefly"}]}
        return embed_reply

    monkeypatch.setattr(T, "_svc", _svc)
    monkeypatch.setattr(T, "_save_png", lambda *a, **k: None)
    tool = {t.name: t for t in T.make_rs_embed_tools()}["embed_region"]
    return json.loads(tool.func(bbox=[-88.3, 40.0, -88.2, 40.1], models=["terramind"]))


def test_embed_region_reports_provenance_when_the_service_sends_it(monkeypatch):
    out = _embed_region(monkeypatch, {"results": [
        {"model": "terramind", "ok": True, "type": "onthefly", "dim": 384,
         "grid_hw": [26, 26], "norm": 1.0, "meta": _TERRAMIND_META}]})
    prov = out["models"][0]["provenance"]
    assert prov["scale_m"] == 10 and prov["model_key"] == "terramind_v1_small"


def test_embed_region_stays_silent_on_a_service_that_sends_no_meta(monkeypatch):
    """/api/embed does not attach meta yet; the key must be absent, not empty."""
    out = _embed_region(monkeypatch, {"results": [
        {"model": "terramind", "ok": True, "type": "onthefly", "dim": 384,
         "grid_hw": [26, 26], "norm": 1.0}]})
    assert "provenance" not in out["models"][0]


def test_bands_are_read_from_wherever_the_model_puts_them():
    """The on-the-fly path nests them under `sensor`; the precomputed path puts them at the
    top level. Reading only one place dropped them for real gse runs."""
    from agent_runtime.rs_embed_tools import _provenance

    assert _provenance(_TERRAMIND_META)["bands"] == ["B1", "B2", "B3"]      # from sensor
    assert _provenance({"model": "x", "bands": ["B4", "B3"]})["bands"] == ["B4", "B3"]  # top level


def test_a_products_dimension_list_is_counted_not_listed():
    """gse names its 64 embedding DIMENSIONS in `bands` (A00…A63). Listing those is noise;
    the count still answers "how wide is this vector"."""
    from agent_runtime.rs_embed_tools import _provenance

    prov = _provenance({"model": "gse", "bands": [f"A{i:02d}" for i in range(64)]})
    assert prov["bands_count"] == 64 and "bands" not in prov


def test_nodata_fraction_survives():
    """A mostly-empty footprint otherwise reads exactly like a full one."""
    from agent_runtime.rs_embed_tools import _provenance

    assert _provenance({"model": "gse", "nodata_fraction": 0.0164})["nodata_fraction"] == 0.0164


def test_a_date_range_reaches_the_zonal_service():
    """Before this, the polygon path took only `year`, so "the clay embedding of Urbana for
    2025-03-01..2025-05-01" forced a choice: the city boundary (whole-year composite) or the
    dates (a rectangle via embed_region). The agent picked the dates and dropped the boundary
    without saying it had traded one for the other."""
    from agent_runtime.rs_embed_tools import _zonal_service_body

    body = _zonal_service_body({"start": "2025-03-01", "end": "2025-05-01"})
    assert body["start"] == "2025-03-01" and body["end"] == "2025-05-01"
    # Half a range would silently become a whole-year composite, so it takes both or neither.
    assert "start" not in _zonal_service_body({"start": "2025-03-01"})
    assert "end" not in _zonal_service_body({"end": "2025-05-01"})
    assert "start" not in _zonal_service_body({})
