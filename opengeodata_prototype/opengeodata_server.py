from flask import Flask, request, jsonify
from werkzeug.exceptions import BadRequest
from nb_logic import run_opengeodata, OpenGeoDataError

app = Flask(__name__)

@app.get("/health")
def health():
    return jsonify({"status": "ok"}), 200

@app.post("/opengeodata/search")
def opengeodata_search():
    try:
        payload = request.get_json(force=True) or {}
    except BadRequest:
        return jsonify({"ok": False, "error": "Invalid JSON"}), 400

    query   = payload.get("query")
    bbox    = payload.get("bbox")            # [minlon,minlat,maxlon,maxlat]
    timer   = payload.get("timer")           # [start_iso,end_iso]
    limit   = int(payload.get("limit", 10))
    providers = payload.get("providers")     # provider overrides
    nl      = payload.get("nl")              # NL block (see below)

    try:
        data = run_opengeodata(query=query, bbox=bbox, timer=timer,
                               limit=limit, providers=providers, nl=nl)
        return jsonify({"ok": True, "data": data}), 200
    except OpenGeoDataError as e:
        return jsonify({"ok": False, "error": str(e)}), 500
    except Exception as e:
        return jsonify({"ok": False, "error": f"Unhandled error: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5010, debug=True)