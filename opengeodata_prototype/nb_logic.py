from dataclasses import dataclass, asdict
from typing import List, Optional, Dict, Any, Tuple
from pathlib import Path
import json, re, time, os, requests
from requests.adapters import HTTPAdapter, Retry

try:
    from dotenv import load_dotenv  # type: ignore
except ImportError:
    load_dotenv = None  # type: ignore

if load_dotenv:
    env_candidates = [
        Path(__file__).resolve().with_name(".env"),
        Path(__file__).resolve().parents[1] / ".env",
    ]
    loaded = False
    for env_path in env_candidates:
        if env_path.exists():
            load_dotenv(env_path)
            loaded = True
            break
    if not loaded:
        load_dotenv()

_API_BASE_ENV_VARS = ("OPENAI_API_BASE", "ANVILGPT_URL", "API_BASE")
_API_KEY_ENV_VARS = ("OPENAI_API_KEY", "OPENAI_KEY", "ANVILGPT_KEY", "API_KEY")


def _get_env_value(names: Tuple[str, ...]) -> Optional[str]:
    for name in names:
        val = os.getenv(name)
        if val:
            return val
    return None

@dataclass
class GeoAsset:
    id: str
    title: str
    abstract: Optional[str]
    keywords: List[str]
    bbox: Optional[Tuple[float, float, float, float]]
    datetime: Optional[Tuple[Optional[str], Optional[str]]]
    license: Optional[str]
    links: Dict[str, str]
    source: str
    provider: Optional[str]

class OpenGeoDataError(Exception): pass
class NLQueryError(Exception): pass

def _session(timeout=12) -> requests.Session:
    s = requests.Session()
    retries = Retry(total=3, backoff_factor=0.4,
                    status_forcelist=[429,500,502,503,504],
                    allowed_methods=["HEAD","GET","OPTIONS","POST"])
    s.mount("https://", HTTPAdapter(max_retries=retries))
    s.mount("http://", HTTPAdapter(max_retries=retries))
    orig = s.request
    def _req(method, url, **kw):
        kw.setdefault("timeout", timeout)
        return orig(method, url, **kw)
    s.request = _req  # type: ignore
    return s

def _norm_bbox(bbox) -> Optional[Tuple[float,float,float,float]]:
    if not bbox: return None
    if isinstance(bbox, dict) and "bbox" in bbox: bbox = bbox["bbox"]
    if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
        return (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))
    if isinstance(bbox, dict) and bbox.get("type") == "Polygon":
        coords = bbox["coordinates"][0]
        xs = [c[0] for c in coords]; ys = [c[1] for c in coords]
        return (min(xs), min(ys), max(xs), max(ys))
    return None

def _dt_range_from_props(props: Dict[str, Any]) -> Tuple[Optional[str], Optional[str]]:
    start = props.get("start_datetime") or props.get("datetime")
    end   = props.get("end_datetime") or start
    return (start, end)

def _parse_bbox_from_ckan_spatial(spatial: Optional[str]) -> Optional[Tuple[float,float,float,float]]:
    if not spatial: return None
    try:
        obj = json.loads(spatial)
        if isinstance(obj, dict):
            if "bbox" in obj: return _norm_bbox(obj["bbox"])
            if obj.get("type") in ("Polygon","MultiPolygon"): return _norm_bbox(obj)
    except Exception:
        pass
    m = re.search(r"POLYGON\s*\(\((.*?)\)\)", spatial, re.I)
    if m:
        xs, ys = [], []
        for p in [p.strip() for p in m.group(1).split(",")]:
            parts = p.split()
            if len(parts) >= 2:
                xs.append(float(parts[0])); ys.append(float(parts[1]))
        if xs and ys: return (min(xs), min(ys), max(xs), max(ys))
    return None

def search_stac(endpoint: str,
                q: Optional[str] = None,
                bbox: Optional[Tuple[float,float,float,float]] = None,
                time_range: Optional[Tuple[Optional[str],Optional[str]]] = None,
                collections: Optional[List[str]] = None,
                limit: int = 10) -> List[GeoAsset]:
    from pystac_client import Client
    client = Client.open(endpoint)
    kw: Dict[str, Any] = {"max_items": limit}
    if bbox: kw["bbox"] = list(bbox)
    if time_range:
        s, e = time_range; kw["datetime"] = f"{s or '..'}/{e or '..'}"
    if collections: kw["collections"] = collections
    supports_filter = False
    if hasattr(client, "conforms_to"):
        supports_filter = (
            client.conforms_to("https://api.stacspec.org/v1.0.0/item-search#filter")
            or client.conforms_to("http://www.opengis.net/spec/cql2/1.0/conf/cql2-text")
            or client.conforms_to("http://www.opengis.net/spec/cql2/1.0/conf/cql2-json")
        )
    if q and supports_filter:
        kw["filter_lang"] = "cql2-text"
        kw["filter"] = f"(title ILIKE '%{q}%') OR (description ILIKE '%{q}%')"
    try:
        search = client.search(**kw)
    except Exception:
        kw.pop("filter", None); kw.pop("filter_lang", None)
        search = client.search(**kw)
    out: List[GeoAsset] = []
    for it in search.items():
        landing = getattr(it, "get_self_href", lambda: None)() or ""
        props = it.properties or {}
        kwds = set(props.get("keywords") or [])
        try: coll = it.get_collection()
        except Exception: coll = None
        if coll:
            try:
                for k in (coll.keywords or []): kwds.add(k)
            except Exception:
                pass
        license_ = props.get("license")
        if not license_ and coll:
            try: license_ = coll.license
            except Exception: pass
        out.append(GeoAsset(
            id=it.id,
            title=props.get("title") or it.id,
            abstract=props.get("description"),
            keywords=list(kwds),
            bbox=_norm_bbox(getattr(it, "bbox", None)),
            datetime=_dt_range_from_props(props),
            license=license_,
            links={"landing": landing, "api": endpoint},
            source="stac",
            provider=(getattr(coll.providers[0], "name", None) if coll and getattr(coll, "providers", None) else None)
        ))
    return out

def search_ogc_records(base: str,
                       q: Optional[str] = None,
                       bbox: Optional[Tuple[float,float,float,float]] = None,
                       time_range: Optional[Tuple[Optional[str],Optional[str]]] = None,
                       limit: int = 10) -> List[GeoAsset]:
    s = _session()
    out: List[GeoAsset] = []
    body: Dict[str, Any] = {"limit": limit}
    if q: body["q"] = q
    if bbox: body["bbox"] = list(bbox)
    if time_range:
        start, end = time_range
        body["datetime"] = f"{start or '..'}/{end or '..'}"
    r = s.post(f"{base.rstrip('/')}/search", json=body, headers={"accept":"application/geo+json"})
    features: List[Dict[str, Any]] = []
    if r.ok:
        features = r.json().get("features", [])
    else:
        rc = s.get(f"{base.rstrip('/')}/collections", headers={"accept":"application/json"})
        colls = (rc.json().get("collections", []) if rc.ok else [])[:2]
        for c in colls:
            p = s.get(f"{base.rstrip('/')}/collections/{c['id']}/items",
                      params={"limit": limit}, headers={"accept":"application/geo+json"})
            if p.ok: features += p.json().get("features", [])
    for f in features[:limit]:
        props = f.get("properties", {}) or {}
        exbbox = f.get("bbox") or props.get("bbox")
        start, end = props.get("datetime"), props.get("end_datetime")
        links = props.get("links") or f.get("links") or []
        landing = ""
        if isinstance(links, list) and links:
            landing = links[0].get("href", "") or ""
        out.append(GeoAsset(
            id=str(f.get("id") or props.get("identifier") or props.get("id")),
            title=props.get("title") or props.get("name") or str(f.get("id")),
            abstract=props.get("description"),
            keywords=props.get("keywords") or [],
            bbox=_norm_bbox(exbbox),
            datetime=(start, end),
            license=props.get("license"),
            links={"landing": landing, "api": base},
            source="ogc-records",
            provider=props.get("publisher") or props.get("provider")
        ))
    return out

def search_ckan(base: str,
                api_key: Optional[str] = None,
                q: str = "",
                limit: int = 10) -> List[GeoAsset]:
    s = _session()
    params = {"q": q, "rows": limit}
    headers = {"X-Api-Key": api_key} if api_key else {}
    r = s.get(f"{base.rstrip('/')}/package_search", params=params, headers=headers)
    r.raise_for_status()
    res = r.json().get("result", {}).get("results", [])
    out: List[GeoAsset] = []
    for pkg in res:
        spatial = pkg.get("spatial")
        bbox = _parse_bbox_from_ckan_spatial(spatial) if spatial else None
        landing = pkg.get("url") or (pkg.get("resources",[{}])[0].get("url","") if pkg.get("resources") else "")
        tags = pkg.get("tags", [])
        if tags and isinstance(tags[0], dict): tags = [t.get("name","") for t in tags]
        out.append(GeoAsset(
            id=pkg["id"],
            title=pkg.get("title") or pkg.get("name"),
            abstract=pkg.get("notes"),
            keywords=[t for t in tags if t],
            bbox=bbox,
            datetime=(pkg.get("temporal_start"), pkg.get("temporal_end")),
            license=pkg.get("license_title") or pkg.get("license_id"),
            links={"landing": landing, "api": f"{base.rstrip('/')}/package_show?id={pkg['id']}"},
            source="ckan",
            provider=(pkg.get("organization", {}) or {}).get("title")
        ))
    return out

def search_cmr_collections(q: Optional[str] = None,
                           bbox: Optional[Tuple[float,float,float,float]] = None,
                           time_range: Optional[Tuple[Optional[str],Optional[str]]] = None,
                           limit: int = 10) -> List[GeoAsset]:
    s = _session()
    params: Dict[str, Any] = {"page_size": limit, "include_has_granules": "true"}
    if q: params["keyword"] = q
    if bbox: params["bounding_box"] = ",".join(map(str, bbox))
    if time_range:
        start, end = time_range
        if start or end: params["temporal"] = f"{start or ''},{end or ''}"
    r = s.get("https://cmr.earthdata.nasa.gov/search/collections.json", params=params)
    r.raise_for_status()
    cols = r.json().get("feed", {}).get("entry", []) or []
    out: List[GeoAsset] = []
    for c in cols:
        box = None
        if c.get("boxes"):
            try:
                minlat, minlon, maxlat, maxlon = map(float, c["boxes"][0].split())
                box = (minlon, minlat, maxlon, maxlat)
            except Exception:
                pass
        landing = ""
        for lk in c.get("links", []):
            if lk.get("rel","").endswith("/data#") or lk.get("rel","").endswith("/documentation#") or lk.get("href"):
                landing = lk.get("href",""); break
        out.append(GeoAsset(
            id=c["id"],
            title=c.get("dataset_id") or c.get("short_name") or c["id"],
            abstract=c.get("summary"),
            keywords=[", ".join(k.values()) if isinstance(k, dict) else str(k) for k in (c.get("science_keywords") or [])],
            bbox=box,
            datetime=(c.get("time_start"), c.get("time_end")),
            license=None,
            links={"landing": landing, "api": "https://cmr.earthdata.nasa.gov/search/"},
            source="cmr",
            provider=(c.get("archive_center") or c.get("data_center"))
        ))
    return out

def discover(query: str = "",
             bbox: Optional[Tuple[float,float,float,float]] = None,
             time_range: Optional[Tuple[Optional[str],Optional[str]]] = None,
             limit: int = 6,
             providers: Dict[str, Any] = None) -> List[GeoAsset]:
    providers = providers or dict(
        stac=["https://planetarycomputer.microsoft.com/api/stac/v1"],
        records=[],
        ckan=[("https://api.gsa.gov/technology/datagov/v3/action", None)],
        cmr=True
    )
    results: List[GeoAsset] = []
    for ep in providers.get("stac", []):
        try: results += search_stac(ep, q=query, bbox=bbox, time_range=time_range, limit=limit)
        except Exception: pass
    for ep in providers.get("records", []):
        try: results += search_ogc_records(ep, q=query, bbox=bbox, time_range=time_range, limit=limit)
        except Exception: pass
    for base, key in providers.get("ckan", []):
        try: results += search_ckan(base, api_key=key, q=query, limit=limit)
        except Exception: pass
    if providers.get("cmr"):
        try: results += search_cmr_collections(query, bbox=bbox, time_range=time_range, limit=limit)
        except Exception: pass
    max_res = limit * (bool(providers.get("stac")) + len(providers.get("records", [])) +
                       len(providers.get("ckan", [])) + int(bool(providers.get("cmr"))))
    return results[:max(1, max_res)]

def score(asset: GeoAsset,
          query_terms: List[str],
          bbox: Optional[Tuple[float,float,float,float]] = None,
          time_range: Optional[Tuple[Optional[str],Optional[str]]] = None) -> float:
    text = (" ".join([asset.title or "", asset.abstract or "", " ".join(asset.keywords or [])])).lower()
    text_score = sum(1.0 for t in query_terms if t and t.lower() in text)
    st_score = 0.0
    if bbox and asset.bbox:
        ax1, ay1, ax2, ay2 = asset.bbox; bx1, by1, bx2, by2 = bbox
        inter_w = max(0.0, min(ax2, bx2) - max(ax1, bx1))
        inter_h = max(0.0, min(ay2, by2) - max(ay1, by1))
        inter = inter_w * inter_h
        if inter > 0:
            area = (ax2-ax1)*(ay2-ay1) + (bx2-bx1)*(by2-by1) - inter
            if area > 0: st_score += inter / area
    if time_range and asset.datetime and (asset.datetime[0] or asset.datetime[1]):
        st_score += 0.2
    license_bonus = 0.2 if (asset.license and "by" in asset.license.lower()) else 0.0
    return text_score + st_score + license_bonus

def _iso_date(s: Optional[str]) -> Optional[str]:
    if not s: return None
    m = re.match(r"^\s*(\d{4})-(\d{2})-(\d{2})", s.strip())
    return f"{m.group(1)}-{m.group(2)}-{m.group(3)}" if m else None

def _valid_bbox(b: Optional[List[float]]) -> Optional[Tuple[float,float,float,float]]:
    if not b or len(b) < 4: return None
    x1,y1,x2,y2 = map(float, b[:4])
    if not (-180.0 <= x1 <= 180.0 and -180.0 <= x2 <= 180.0 and -90.0 <= y1 <= 90.0 and -90.0 <= y2 <= 90.0):
        return None
    if x2 <= x1 or y2 <= y1: return None
    return (x1,y1,x2,y2)

def get_q_bbox_timer_openai(user_query: str, *,
    current_date: str, api_base: Optional[str] = None, api_key: Optional[str] = None, model: str,
    timeout: int = 20,
    default_bbox: Optional[Tuple[float,float,float,float]] = None,
    default_timer: Optional[Tuple[Optional[str],Optional[str]]] = None,
    max_retries: int = 2
) -> Tuple[str, Optional[Tuple[float,float,float,float]], Optional[Tuple[Optional[str],Optional[str]]]]:
    api_key = api_key or _get_env_value(_API_KEY_ENV_VARS)
    if not api_key: raise NLQueryError("Missing API key.")
    api_base = api_base or _get_env_value(_API_BASE_ENV_VARS)
    if not api_base: raise NLQueryError("Missing API base.")
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    system_prompt = f"""
You are a geospatial query normalizer.
Output JSON exactly: {{"q":"...","bbox":[minlon,minlat,maxlon,maxlat]|null,"timer":[start_or_null,end_or_null]}}
Use today={current_date}. Remove locations/dates from q and put them into bbox/timer.
"""
    json_schema = {
        "name": "GeoQuery",
        "schema": {"type":"object","properties":{
            "q":{"type":"string"},
            "bbox":{"type":["array","null"],"items":{"type":"number"},"minItems":4,"maxItems":4},
            "timer":{"type":["array","null"],"items":{"type":["string","null"]},"minItems":2,"maxItems":2}
        },"required":["q","bbox","timer"],"additionalProperties":False}
    }
    messages = [{"role":"system","content":system_prompt},{"role":"user","content":user_query}]
    body_strict = {"model":model,"messages":messages,"temperature":0,
                   "response_format":{"type":"json_schema","json_schema":json_schema}}
    body_fallback = {"model":model,"messages":messages+[{"role":"system","content":"Respond ONLY with the JSON schema above."}],
                     "temperature":0}
    last_err = None
    for attempt in range(max_retries+1):
        try:
            use_body = body_strict if attempt == 0 else body_fallback
            resp = requests.post(f"{api_base.rstrip('/')}/chat/completions", headers=headers, json=use_body, timeout=timeout)
            resp.raise_for_status()
            data = json.loads(resp.json()["choices"][0]["message"]["content"])
            q = str(data.get("q","")).strip()
            if not q: raise NLQueryError("Empty q.")
            bbox = _valid_bbox(data.get("bbox"))
            timer_raw = data.get("timer")
            timer: Optional[Tuple[Optional[str],Optional[str]]] = None
            if isinstance(timer_raw, list) and len(timer_raw) >= 2:
                s = _iso_date(timer_raw[0]); e = _iso_date(timer_raw[1]); timer = (s, e)
            if bbox is None: bbox = default_bbox
            if (timer is None or (timer[0] is None and timer[1] is None)) and default_timer:
                timer = default_timer
            return q, bbox, timer
        except Exception as e:
            last_err = e; time.sleep(0.4)
    raise NLQueryError(f"NL parse failed: {last_err}")

def _asset_to_dict(a: GeoAsset) -> Dict[str, Any]:
    d = asdict(a)
    if d.get("bbox") is not None: d["bbox"] = list(d["bbox"])
    return d

def run_opengeodata(*,
    query: Optional[str] = None,
    bbox: Optional[List[float]] = None,
    timer: Optional[List[Optional[str]]] = None,
    limit: int = 10,
    providers: Optional[Dict[str, Any]] = None,
    nl: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    try:
        if nl:
            q, bb, tt = get_q_bbox_timer_openai(
                nl["user_query"],
                current_date=nl["current_date"],
                api_base=nl.get("api_base"),
                api_key=nl.get("api_key"),
                model=nl["model"],
                default_bbox=tuple(nl.get("default_bbox")) if nl.get("default_bbox") else None,
                default_timer=tuple(nl.get("default_timer")) if nl.get("default_timer") else None
            )
        else:
            q = query or ""
            bb = _valid_bbox(bbox) if bbox else None
            tt = None
            if timer and len(timer) >= 2:
                tt = (_iso_date(timer[0]), _iso_date(timer[1]))
        assets = discover(q, bb, tt, limit=limit, providers=providers)
        assets = sorted(assets, key=lambda a: -score(a, q.lower().split(), bb, tt))[:limit]
        return {
            "query": q,
            "bbox": list(bb) if bb else None,
            "timer": [tt[0], tt[1]] if tt else [None, None],
            "count": len(assets),
            "assets": [_asset_to_dict(a) for a in assets]
        }
    except NLQueryError as e:
        raise OpenGeoDataError(str(e))
    except Exception as e:
        raise OpenGeoDataError(str(e))
