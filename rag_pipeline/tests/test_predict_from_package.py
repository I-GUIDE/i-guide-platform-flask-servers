"""Scoring an embedding the user already has, without re-embedding the region.

``predict_from_package`` hands a saved ``.npz`` to the service's ``/api/predict_package``, which
runs the pretrained heads on the pooled vectors and needs no Earth Engine. That makes it cheap,
and cheap is exactly what makes the guards matter: the heads are three RandomForests fitted on
corn in Illinois in 2022, they record none of that, and a vector from another year, another place
or another model of the same width is scored in silence and comes back looking in-domain.

These tests pin the guards rather than the plumbing. Each one stands for a measured way to get a
confident wrong number: a stack of zone vectors flattened into one row, a right-width
wrong-model vector, an all-nodata region turned into zeros, and a date check that reads the one
manifest field that is actually always present.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from agent_runtime import rs_embed_head_domain as domain  # noqa: E402

# A +/-2048 m square near Champaign: the footprint embed_region uses by default, and small
# enough that the footprint guard stays quiet on the ordinary case.
ILLINOIS_BOX = {"type": "bbox", "minlon": -88.265, "minlat": 40.100,
                "maxlon": -88.217, "maxlat": 40.137}
BAVARIA_BOX = {"type": "bbox", "minlon": 11.4, "minlat": 48.0,
               "maxlon": 11.6, "maxlat": 48.2}


def _manifest(geometry=None, start="2022-06", end="2022-09", models=None, **extra):
    man = {"geometry": geometry if geometry is not None else dict(ILLINOIS_BOX),
           "start": start, "end": end, "buffer_m": 2048, "compute": "get_embedding",
           "models": models if models is not None else [{"model": "gse", "dim": 64, "meta": {}}]}
    man.update(extra)
    return man


def _package(tmp_path, name="pkg.npz", *, pooled=None, grid=True, manifest=...):
    """A package shaped exactly like the service's export: pooled__/grid__/meta."""
    payload = {}
    for model, vec in (pooled or {"gse": np.arange(64, dtype=np.float32)}).items():
        payload[f"pooled__{model}"] = np.asarray(vec, dtype=np.float32)
        if grid:
            # Incompressible on purpose: a grid of zeros would make the size assertion below
            # pass whether or not the repack actually dropped anything.
            rng = np.random.default_rng(len(model))
            payload[f"grid__{model}"] = rng.normal(
                size=(np.size(vec), 32, 32)).astype(np.float32)
    if manifest is not ...:
        if manifest is not None:
            payload["meta"] = np.asarray(json.dumps(manifest))
    else:
        payload["meta"] = np.asarray(json.dumps(_manifest()))
    path = tmp_path / name
    np.savez_compressed(path, **payload)
    return path


# --- reading the file ----------------------------------------------------------
def test_pooled_and_manifest_read_back(tmp_path):
    path = _package(tmp_path)
    vectors, problems = domain.pooled_vectors(path)
    assert problems == []
    assert set(vectors) == {"gse"}
    assert vectors["gse"].shape == (64,)
    man = domain.read_manifest(path)
    assert man["start"] == "2022-06"
    assert domain.package_region(man) == {
        "region_bbox": [-88.265, 40.1, -88.217, 40.137], "months": "2022-06..2022-09"}


def test_a_pickled_package_is_not_deserialised(tmp_path):
    """allow_pickle=False, because this path np.loads whatever file it is handed.

    A package can come from an upload as easily as from the service. The object array here is
    inert, but reaching it at all would mean a hostile .npz could run code; the guard must
    degrade to "no manifest" rather than unpickle.
    """
    path = tmp_path / "pickled.npz"
    np.savez(path, meta=np.array([{"start": "2022-06"}], dtype=object),
             pooled__gse=np.arange(64, dtype=np.float32))
    assert domain.read_manifest(path) == {}
    vectors, _problems = domain.pooled_vectors(path)
    # The float array beside it still reads: refusing pickles must not cost the vectors.
    assert set(vectors) == {"gse"}


def test_repack_drops_the_grid_and_keeps_the_numbers(tmp_path):
    path = _package(tmp_path)
    out = domain.repack_pooled(path, tmp_path / "pooled.npz")
    assert out["models"] == ["gse"]
    assert out["size_bytes"] < path.stat().st_size / 4
    with np.load(out["path"], allow_pickle=False) as z:
        assert sorted(z.files) == ["meta", "pooled__gse"]
        original, _ = domain.pooled_vectors(path)
        assert np.array_equal(z["pooled__gse"], original["gse"])
    assert domain.read_manifest(out["path"])["start"] == "2022-06"


# --- the vector guards ---------------------------------------------------------
def test_a_good_vector_is_accepted():
    assert domain.vector_refusal("gse", np.arange(64, dtype=np.float32), 64) is None


def test_a_stack_of_zone_vectors_is_refused():
    """16 zones x 64 dims flattens to (1, 1024) and is scored as though it were one region."""
    refusal = domain.vector_refusal("satmae", np.zeros((16, 64), dtype=np.float32), 1024)
    assert refusal and "not a single pooled vector" in refusal
    assert "16" in refusal


def test_a_right_width_wrong_model_vector_is_refused_by_width():
    refusal = domain.vector_refusal("gse", np.zeros(768, dtype=np.float32), 64)
    assert refusal and "768" in refusal and "64" in refusal


def test_an_all_nodata_vector_is_refused():
    """/api/predict raises on an all-nodata ROI; the package path would score it as ~0.45."""
    refusal = domain.vector_refusal("gse", np.full(64, np.nan, dtype=np.float32), 64)
    assert refusal and "no finite value" in refusal


def test_a_partly_nodata_vector_is_still_scored():
    vec = np.arange(64, dtype=np.float32)
    vec[:5] = np.nan
    assert domain.vector_refusal("gse", vec, 64) is None


# --- the domain guards ---------------------------------------------------------
def test_an_in_domain_package_warns_about_nothing():
    assert domain.domain_warnings(_manifest()) == []
    assert domain.unverifiable_domain(_manifest()) == []


def test_the_year_check_reads_the_field_that_is_always_present():
    """The top-level start/end, NOT the per-model meta.temporal.

    Across the 100 manifests in the store, per-model ``meta`` is empty for 36 of 105 entries —
    including every satmae and dofa one, two of the three models with a head — while top-level
    start/end is present on all 100. A guard hung on the precise field stays silent on exactly
    the models where it is most needed, so this package (2025 imagery, no per-model meta at all)
    must still be caught.
    """
    man = _manifest(start="2025-03", end="2025-05",
                    models=[{"model": "satmae", "dim": 1024, "meta": {}}])
    warnings = domain.domain_warnings(man)
    assert len(warnings) == 1
    assert "2025" in warnings[0] and "2022" in warnings[0]
    assert domain.unverifiable_domain(man) == []


def test_a_multi_year_window_that_covers_2022_does_not_warn():
    assert domain.domain_warnings(_manifest(start="2022-01", end="2022-12")) == []


def test_a_region_outside_illinois_warns():
    warnings = domain.domain_warnings(_manifest(geometry=dict(BAVARIA_BOX)))
    assert any("outside Illinois" in w for w in warnings)


def test_a_region_straddling_the_border_warns_partially():
    straddle = {"type": "bbox", "minlon": -92.4, "minlat": 40.0,
                "maxlon": -91.0, "maxlat": 40.5}
    warnings = domain.domain_warnings(_manifest(geometry=straddle))
    assert any("only partly overlaps" in w for w in warnings)


def test_a_region_far_larger_than_the_training_footprint_warns():
    big = {"type": "bbox", "minlon": -89.0, "minlat": 40.0,
           "maxlon": -88.0, "maxlat": 40.9}
    warnings = domain.domain_warnings(_manifest(geometry=big))
    assert any("spans about" in w for w in warnings)


def test_the_agents_own_default_footprint_does_not_warn():
    """buffer_m 2048 is 1.6x the training footprint. A guard that fires on every call is noise."""
    assert domain.domain_warnings(_manifest()) == []


def test_a_package_with_no_manifest_says_so_rather_than_passing():
    assert domain.domain_warnings({}) == []
    unverifiable = domain.unverifiable_domain({})
    assert len(unverifiable) == 1 and "no manifest" in unverifiable[0]


def test_a_manifest_missing_geometry_reports_the_gap():
    man = _manifest()
    man.pop("geometry")
    gaps = domain.unverifiable_domain(man)
    assert any("no geometry" in g for g in gaps)


def test_width_note_names_the_models_that_collide():
    note = domain.width_note(768)
    assert note and all(m in note for m in ("dofa", "olmoearth", "prithvi", "terrafm", "thor"))
    assert domain.width_note(64) is None


# --- the tool ------------------------------------------------------------------
HEADS = {"task": "corn_presence", "kind": "classification", "label": "corn presence",
         "units": "P(corn)", "region": "Illinois (CDL 2022)",
         "classes": ["not corn", "corn"],
         "models": [{"model": "gse", "dim": 64, "score": 0.444, "score_name": "accuracy", "n": 30},
                    {"model": "satmae", "dim": 1024, "score": 0.667,
                     "score_name": "accuracy", "n": 30}]}


@pytest.fixture()
def tool(monkeypatch):
    """predict_from_package with the service and the file store stubbed out."""
    from agent_runtime import rs_embed_tools

    calls = {"uploads": []}

    def fake_svc(path, payload=None, *, method="POST", timeout=None):
        assert path == "/api/heads"
        return dict(HEADS)

    def fake_upload(path, file_path, *, field="file", timeout=None):
        calls["uploads"].append(str(file_path))
        with np.load(file_path, allow_pickle=False) as z:
            scored = sorted(k[len("pooled__"):] for k in z.files if k.startswith("pooled__"))
            calls["uploaded_keys"] = sorted(z.files)
        return {"task": "corn_presence", "kind": "classification", "label": "corn presence",
                "units": "P(corn)",
                "results": [{"model": m, "ok": True, "kind": "classification",
                             "prediction": 0.5625, "label_pred": "corn",
                             "score": 0.667, "score_name": "accuracy"} for m in scored]}

    monkeypatch.setattr(rs_embed_tools, "_svc", fake_svc)
    monkeypatch.setattr(rs_embed_tools, "_svc_upload", fake_upload)
    fn = {t.name: t for t in rs_embed_tools.make_rs_embed_tools()}["predict_from_package"].func
    return fn, calls


def _point_at(monkeypatch, path):
    from agent_runtime import file_store

    monkeypatch.setattr(file_store, "resolve_file_id", lambda fid: Path(path))


def test_the_tool_scores_a_saved_package(tool, tmp_path, monkeypatch):
    fn, calls = tool
    _point_at(monkeypatch, _package(tmp_path))
    out = json.loads(fn("file_abc"))
    assert out["ok"] is True
    assert out["scored_models"] == ["gse"]
    assert out["region_bbox"] == [-88.265, 40.1, -88.217, 40.137]
    assert out["months"] == "2022-06..2022-09"
    assert out["region"] == "Illinois (CDL 2022)"
    assert out["validation"]["gse"] == {"score": 0.444, "score_name": "accuracy", "n": 30}
    assert "outside_training_domain" not in out
    assert out["results"][0]["prediction"] == 0.5625


def test_only_the_pooled_keys_are_uploaded(tool, tmp_path, monkeypatch):
    fn, calls = tool
    _point_at(monkeypatch, _package(tmp_path))
    json.loads(fn("file_abc"))
    assert calls["uploaded_keys"] == ["meta", "pooled__gse"]


def test_a_model_with_no_head_is_reported_and_not_uploaded(tool, tmp_path, monkeypatch):
    fn, calls = tool
    path = _package(tmp_path, pooled={"gse": np.arange(64, dtype=np.float32),
                                      "clay": np.arange(1024, dtype=np.float32)})
    _point_at(monkeypatch, path)
    out = json.loads(fn("file_abc"))
    assert out["ok"] is True
    assert out["scored_models"] == ["gse"]
    assert [e["model"] for e in out["not_scored"]] == ["clay"]
    assert calls["uploaded_keys"] == ["meta", "pooled__gse"]


def test_nothing_is_uploaded_when_no_model_has_a_head(tool, tmp_path, monkeypatch):
    fn, calls = tool
    _point_at(monkeypatch, _package(tmp_path, pooled={"thor": np.arange(768, dtype=np.float32)}))
    out = json.loads(fn("file_abc"))
    assert out["ok"] is False
    assert calls["uploads"] == []
    assert out["heads_available"] == ["gse", "satmae"]


def test_an_out_of_domain_package_is_scored_but_flagged(tool, tmp_path, monkeypatch):
    fn, _calls = tool
    path = _package(tmp_path, manifest=_manifest(geometry=dict(BAVARIA_BOX),
                                                 start="2025-03", end="2025-05"))
    _point_at(monkeypatch, path)
    out = json.loads(fn("file_abc"))
    assert out["ok"] is True
    flags = out["outside_training_domain"]
    assert any("2025" in f for f in flags)
    assert any("outside Illinois" in f for f in flags)


def test_a_manifestless_package_is_scored_but_called_unknown(tool, tmp_path, monkeypatch):
    fn, _calls = tool
    _point_at(monkeypatch, _package(tmp_path, manifest=None))
    out = json.loads(fn("file_abc"))
    assert out["ok"] is True
    assert "region_bbox" not in out
    assert any("no manifest" in g for g in out["domain_unverifiable"])


def test_a_zones_csv_is_refused_with_the_right_alternative(tool, tmp_path, monkeypatch):
    """embed_zones writes a CSV, not a package — and its rows should not go to these heads."""
    fn, calls = tool
    csv = tmp_path / "zone_vectors.csv"
    csv.write_text("zone_id,pixels,e000,e001\n17019,12,0.1,0.2\n")
    _point_at(monkeypatch, csv)
    out = json.loads(fn("file_abc"))
    assert out["ok"] is False
    assert "fit_zone_model" in out["hint"]
    assert calls["uploads"] == []


def test_an_unresolvable_file_id_says_what_to_pass(tool, tmp_path, monkeypatch):
    from agent_runtime import file_store

    fn, calls = tool

    def boom(fid):
        raise ValueError(f"file for file_id does not exist: {fid}")

    monkeypatch.setattr(file_store, "resolve_file_id", boom)
    out = json.loads(fn("file_missing"))
    assert out["ok"] is False
    assert "embedding_package.file_id" in out["hint"]
    assert calls["uploads"] == []
