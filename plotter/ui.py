"""
Local preview UI: `python -m plotter ui` then open http://127.0.0.1:8765

Stdlib only (http.server). The page uploads a photo, sends style parameters,
and gets back the ordered polylines, which it draws on a canvas at true pen
width (with zoom, travel moves and a plot-order animation). G-code / SVG are
written to the work dir and offered as downloads.
"""

import base64
import io
import json
import re
import threading
import time
import traceback
import uuid
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from . import pipeline
from . import stringart as sa
from .gcode import PlotterProfile

HERE = Path(__file__).parent
MAX_UPLOAD = 40 * 1024 * 1024
MAX_SIDE = 2000
ID_RE = re.compile(r"^[0-9a-f]{32}$")
FILE_RE = re.compile(r"^plot\.(gcode|svg|json)$")


def _png_data_url(darkness):
    if darkness is None:
        return None
    buf = io.BytesIO()
    im = Image.fromarray((255 * (1 - darkness.clip(0, 1))).astype("uint8"))
    im.save(buf, "PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


class App:
    def __init__(self, workdir):
        self.work = Path(workdir)
        (self.work / "uploads").mkdir(parents=True, exist_ok=True)
        (self.work / "jobs").mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()   # one render at a time; they are CPU bound anyway

    # -- uploads -------------------------------------------------------------
    def upload(self, data):
        img = ImageOps.exif_transpose(Image.open(io.BytesIO(data)))
        img = img.convert("RGB")
        if max(img.size) > MAX_SIDE:
            img.thumbnail((MAX_SIDE, MAX_SIDE), Image.Resampling.LANCZOS)
        uid = uuid.uuid4().hex
        path = self.work / "uploads" / f"{uid}.png"
        img.save(path)
        gray = img.convert("L")
        box = sa._auto_crop_box(gray)
        a = np.asarray(gray)
        ys, xs = np.where(a < 245)
        content = ([int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]
                   if len(xs) else [0, 0, img.width, img.height])
        return {"id": uid, "width": img.width, "height": img.height, "content": content,
                "crop": [int(v) for v in box], "url": f"/api/image/{uid}"}

    def image_path(self, uid):
        if not ID_RE.match(uid):
            raise ValueError("bad id")
        p = self.work / "uploads" / f"{uid}.png"
        if not p.exists():
            raise FileNotFoundError(uid)
        return p

    # -- render --------------------------------------------------------------
    def render(self, req):
        style = req.get("style")
        if style not in pipeline.STYLES:
            raise ValueError(f"unknown style {style!r}")
        cls, build = pipeline.STYLES[style]
        params = dict(req.get("params") or {})
        if req.get("crop"):
            params["crop"] = req["crop"]
        if style == "hatch":
            n = int(params.pop("layers", 4))
            params["layer_angles"] = [params.pop(f"a{i}", d) for i, d in
                                      zip(range(1, 5), (0, 90, 45, -45))][:n]
            params["layer_thresholds"] = [params.pop(f"t{i}", d) for i, d in
                                          zip(range(1, 5), (0.12, 0.40, 0.62, 0.80))][:n]
        if style == "scribble":
            ha = params.pop("hatch_angles", None)
            if isinstance(ha, str) and ha.strip():
                params["hatch_angles"] = [[float(v) for v in a.split(":")] for a in ha.split(",") if ":" in a]
            if params.get("max_side") in (0, "0"):
                params["max_side"] = None
        cfg = pipeline.config_from_dict(cls, params)
        profile = pipeline.config_from_dict(PlotterProfile, req.get("profile") or {})

        img = self.image_path(req["image"])
        t0 = time.time()
        with self.lock:
            res = build(str(img), cfg)
        elapsed = time.time() - t0

        job = uuid.uuid4().hex
        jdir = self.work / "jobs" / job
        jdir.mkdir(parents=True)
        stats = pipeline.export(res, str(jdir / "plot"), profile,
                                comment=f"{style} via preview UI")
        info = {k: v for k, v in res.info.items() if k != "nail_strokes"}
        meta = {"style": style, "params": params, "profile": req.get("profile"),
                "stats": stats, "info": info}
        if "nail_strokes" in res.info:
            meta["nail_strokes"] = res.info["nail_strokes"]
        (jdir / "plot.json").write_text(json.dumps(meta, indent=1, default=str))

        return {
            "job": job,
            "paper": {"w": res.paper_w, "h": res.paper_h},
            "pen_mm": res.pen_mm,
            "polylines": [[round(c, 2) for pt in pl for c in pt] for pl in res.polylines],
            "stats": {**stats, **info, "seconds": round(elapsed, 1)},
            "images": {"target": _png_data_url(res.target),
                       "simulation": _png_data_url(res.simulation)},
            "files": {ext: f"/api/file/{job}/plot.{ext}" for ext in ("gcode", "svg", "json")},
        }

    def job_file(self, job, name):
        if not ID_RE.match(job) or not FILE_RE.match(name):
            raise ValueError("bad path")
        p = self.work / "jobs" / job / name
        if not p.exists():
            raise FileNotFoundError(name)
        return p


def make_handler(app):
    class Handler(BaseHTTPRequestHandler):
        server_version = "PlotterUI/1"

        def log_message(self, fmt, *args):
            if not self.path.startswith("/api/image"):
                super().log_message(fmt, *args)

        def _send(self, code, body, ctype="application/json", extra=None):
            if isinstance(body, (dict, list)):
                body = json.dumps(body).encode()
            elif isinstance(body, str):
                body = body.encode()
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            for k, v in (extra or {}).items():
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(body)

        def _error(self, e):
            code = (HTTPStatus.NOT_FOUND if isinstance(e, FileNotFoundError)
                    else HTTPStatus.BAD_REQUEST if isinstance(e, (ValueError, KeyError, TypeError))
                    else HTTPStatus.INTERNAL_SERVER_ERROR)
            if code == HTTPStatus.INTERNAL_SERVER_ERROR:
                traceback.print_exc()
            self._send(code, {"error": f"{type(e).__name__}: {e}"})

        def do_GET(self):
            try:
                parts = self.path.split("?")[0].strip("/").split("/")
                if self.path in ("/", "/index.html"):
                    self._send(200, (HERE / "ui_index.html").read_bytes(), "text/html; charset=utf-8")
                elif parts[:2] == ["api", "image"] and len(parts) == 3:
                    self._send(200, app.image_path(parts[2]).read_bytes(), "image/png")
                elif parts[:2] == ["api", "file"] and len(parts) == 4:
                    p = app.job_file(parts[2], parts[3])
                    ctype = {"gcode": "text/plain", "svg": "image/svg+xml",
                             "json": "application/json"}[p.suffix[1:]]
                    self._send(200, p.read_bytes(), ctype,
                               {"Content-Disposition": f'attachment; filename="{p.name}"'})
                else:
                    self._send(404, {"error": "not found"})
            except Exception as e:  # noqa: BLE001
                self._error(e)

        def do_POST(self):
            try:
                n = int(self.headers.get("Content-Length", 0))
                if n <= 0 or n > MAX_UPLOAD:
                    raise ValueError("empty or too large request")
                body = self.rfile.read(n)
                if self.path == "/api/upload":
                    self._send(200, app.upload(body))
                elif self.path == "/api/render":
                    self._send(200, app.render(json.loads(body)))
                else:
                    self._send(404, {"error": "not found"})
            except Exception as e:  # noqa: BLE001
                self._error(e)

    return Handler


def serve(host="127.0.0.1", port=8765, workdir="out/ui", open_browser=True):
    app = App(workdir)
    httpd = ThreadingHTTPServer((host, port), make_handler(app))
    url = f"http://{host}:{port}/"
    print(f"Plotter preview UI on {url}  (work dir: {app.work.resolve()})  Ctrl+C to stop")
    if open_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
