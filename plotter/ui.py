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
import math
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
        self.sources = {}              # layout: id -> (local polylines, w, h)
        self.composed = {}             # compose: key -> placed paper polylines

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

    # -- importance (string art) ---------------------------------------------
    def mask_path(self, uid):
        return self.image_path(uid).with_name(f"{uid}_mask.png")

    def save_mask(self, uid, data):
        """Painted importance mask (PNG, white = important). An all-black mask deletes it."""
        p = self.mask_path(uid)
        im = Image.open(io.BytesIO(data)).convert("L")
        if not np.asarray(im).any():
            p.unlink(missing_ok=True)
            return {"mask": False}
        im.save(p)
        return {"mask": True}

    def face_map(self, uid):
        """Detected faces as a PNG importance map (grayscale, max 512 px) for the overlay."""
        from .importance import source_map
        m, info = source_map(self.image_path(uid), face=True, mask_path=None)
        im = Image.fromarray((m * 255).astype("uint8"))
        im.thumbnail((512, 512), Image.Resampling.BILINEAR)
        buf = io.BytesIO()
        im.save(buf, "PNG")
        return buf.getvalue(), info

    def _with_mask(self, params, uid):
        p = self.mask_path(uid)
        if p.exists():
            params["importance_mask"] = str(p)
        return params

    # -- render --------------------------------------------------------------
    def _config(self, style, params, crop, image):
        """UI params -> (style config, build function)."""
        if style not in pipeline.STYLES:
            raise ValueError(f"unknown style {style!r}")
        cls, build = pipeline.STYLES[style]
        params = dict(params or {})
        if crop:
            params["crop"] = crop
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
            params.setdefault("size_mm", params.get("diameter_mm") or 0)   # the common "Size on paper"
        if style == "stringart":
            self._with_mask(params, image)
        return pipeline.config_from_dict(cls, params), build, params

    def render(self, req):
        style = req.get("style")
        cfg, build, params = self._config(style, req.get("params"), req.get("crop"), req["image"])
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
                "stats": stats, "info": info, "image": req["image"]}
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

    # -- compose: several photos, each solved at the size of its frame on the sheet ---------
    def compose_item(self, req):
        """Render one photo for a frame {x, y, w, h} (mm on the sheet). The drawing is solved at
        exactly the frame's size with the real pen, then moved into place. Returns the placed
        polylines; they're kept server-side under `key` for compose_export."""
        style = req.get("style")
        f = req["frame"]
        x, y, w, h = (float(f[k]) for k in ("x", "y", "w", "h"))
        shape = req.get("shape", "circle")
        params = dict(req.get("params") or {})
        pen = float(req.get("pen_mm") or params.get("pen_mm") or 0.3)
        # a big page so nothing is clipped; the drawing is centred on it and moved afterwards
        params.update(pen_mm=pen, paper="A3", portrait=True, padding_mm=0, tone=float(req.get("tone", 1.0)))
        if style == "scribble":
            params.update(shape=shape, size_mm=max(w, h), diameter_mm=max(w, h), rotate="none")
            if not params.get("dpi"):
                params["dpi"] = 100
        elif style == "hatch":
            params.update(shape="rect" if shape == "rect" else "circle", diameter_mm=min(w, h))
        else:
            params.update(diameter_mm=min(w, h))
        cfg, build, _ = self._config(style, params, req.get("crop"), req["image"])
        img = self.image_path(req["image"])
        t0 = time.time()
        with self.lock:
            res = build(str(img), cfg)
        dx, dy = x + w / 2 - res.paper_w / 2, y + h / 2 - res.paper_h / 2
        placed = [[(px + dx, py + dy) for px, py in pl] for pl in res.polylines]
        key = uuid.uuid4().hex
        self.composed[key] = placed
        length = sum(math.dist(a, b) for pl in placed for a, b in zip(pl, pl[1:]))
        return {"key": key, "polylines": [[round(c, 2) for pt in pl for c in pt] for pl in placed],
                "stats": {"draw_m": round(length / 1000, 1), "strokes": len(placed),
                          "seconds": round(time.time() - t0, 1)}}

    def compose_export(self, req):
        from .gcode import write_gcode, write_svg, limit_ink
        paper = req["paper"]
        pw, ph = float(paper["w"]), float(paper["h"])
        # parts: [{key, dx, dy}] in plot order; dx/dy = how far the frame was moved after rendering
        parts = req.get("parts") or [{"key": k} for k in req.get("keys") or []]
        missing = [p["key"] for p in parts if p.get("key") not in self.composed]
        if not parts or missing:
            raise ValueError("render every frame first" if not parts else
                             "some frames are no longer rendered on the server (restart?); render them again")
        pls = []
        for p in parts:
            dx, dy = float(p.get("dx", 0)), float(p.get("dy", 0))
            pls += [[(x + dx, y + dy) for x, y in pl] for pl in self.composed[p["key"]]]
        keys = [p["key"] for p in parts]
        profile = pipeline.config_from_dict(PlotterProfile, req.get("profile") or {})
        pen = float(req.get("pen_mm") or 0.3)
        pls, ink = limit_ink(pls, pen, profile.ink_cap, profile.ink_cell_mm)
        job = uuid.uuid4().hex
        jdir = self.work / "jobs" / job
        jdir.mkdir(parents=True)
        stats = write_gcode(pls, str(jdir / "plot.gcode"), profile, paper_height=ph, paper_width=pw,
                            comment=f"composed sheet: {len(keys)} photos on {pw:g} x {ph:g} mm")
        stats["ink_removed_pct"] = round(ink["ink_removed_pct"], 1)
        write_svg(pls, str(jdir / "plot.svg"), pw, ph, pen)
        meta = {"style": "compose", "params": {"paper": paper, "items": req.get("items")},
                "profile": req.get("profile"), "stats": stats}
        (jdir / "plot.json").write_text(json.dumps(meta, indent=1, default=str))
        return {"job": job, "stats": stats,
                "files": {ext: f"/api/file/{job}/plot.{ext}" for ext in ("gcode", "svg", "json")}}

    def suggest(self, req):
        """Optimise string-art tone settings for the uploaded photo."""
        from .tune import suggest_stringart
        params = dict(req.get("params") or {})
        if req.get("crop"):
            params["crop"] = req["crop"]
        self._with_mask(params, req["image"])
        cfg = pipeline.config_from_dict(sa.StringArtConfig, params)
        img = self.image_path(req["image"])
        with self.lock:
            best, score, history = suggest_stringart(str(img), cfg, time_limit=float(req.get("seconds", 150)))
        return {"params": best, "score": round(score, 4), "history": history}

    # -- sheet layout -------------------------------------------------------
    def jobs(self, limit=60):
        """Recent renders, newest first, for the layout page."""
        out = []
        dirs = sorted((d for d in (self.work / "jobs").iterdir() if (d / "plot.gcode").exists()),
                      key=lambda d: d.stat().st_mtime, reverse=True)[:limit]
        for d in dirs:
            try:
                meta = json.loads((d / "plot.json").read_text())
            except (OSError, ValueError):
                meta = {}
            st = meta.get("stats", {})
            p = meta.get("params", {})
            label = meta.get("style", "?")
            if p.get("chords"):
                label += f" · {p['chords']}"
            if p.get("hatch_mode"):
                label += f" · {p['hatch_mode']}"
            out.append({"job": d.name, "label": label, "time": d.stat().st_mtime,
                        "minutes": st.get("est_minutes"), "pen_mm": (p.get("pen_mm") or None),
                        "svg": f"/api/file/{d.name}/plot.svg"})
        return out

    def layout_load(self, req):
        """Parse a G-code (uploaded text or a previous render) into a placeable drawing."""
        from .layout import parse_gcode, normalise
        if req.get("job"):
            text = self.job_file(req["job"], "plot.gcode").read_text()
            name = req.get("name") or req["job"][:8]
        else:
            text, name = req["gcode"], req.get("name") or "drawing"
        pls, info = parse_gcode(text)
        local, w, h = normalise(pls)
        sid = uuid.uuid4().hex
        self.sources[sid] = (local, w, h)
        # preview geometry: drop points closer than 0.1 mm to the previous kept one
        prev = []
        for pl in local:
            keep = [pl[0]]
            for q in pl[1:]:
                if abs(q[0] - keep[-1][0]) + abs(q[1] - keep[-1][1]) > 0.1:
                    keep.append(q)
            if keep[-1] != pl[-1]:
                keep.append(pl[-1])
            prev.append([round(c, 2) for pt in keep for c in pt])
        draw_mm = sum(math.dist(a, b) for pl in local for a, b in zip(pl, pl[1:]))
        return {"id": sid, "name": name, "w": w, "h": h, "polylines": prev,
                "strokes": len(local), "draw_mm": draw_mm, "pen": info["pen"]}

    def layout_export(self, req):
        from .gcode import write_gcode, write_svg, limit_ink
        from .layout import compose
        paper = req["paper"]
        pw, ph = float(paper["w"]), float(paper["h"])
        items = req.get("items") or []
        if not items:
            raise ValueError("nothing on the sheet")
        pls = compose(items, self.sources)
        profile = pipeline.config_from_dict(PlotterProfile, req.get("profile") or {})
        profile.corner_tol_mm = 0.0            # the source files were already processed
        job = uuid.uuid4().hex
        jdir = self.work / "jobs" / job
        jdir.mkdir(parents=True)
        pen = float(req.get("pen_mm") or 0.3)
        # scaled-down drawings pack their lines closer: the cap trims the overdraw at the final size
        pls, ink = limit_ink(pls, pen, profile.ink_cap, profile.ink_cell_mm)
        stats = write_gcode(pls, str(jdir / "plot.gcode"), profile, paper_height=ph, paper_width=pw,
                            comment=f"sheet layout: {len(items)} drawings on {pw:g} x {ph:g} mm")
        stats["ink_removed_pct"] = round(ink["ink_removed_pct"], 1)
        write_svg(pls, str(jdir / "plot.svg"), pw, ph, pen)
        meta = {"style": "layout", "params": {"paper": paper, "items": items}, "profile": req.get("profile"),
                "stats": stats}
        (jdir / "plot.json").write_text(json.dumps(meta, indent=1, default=str))
        return {"job": job, "stats": stats,
                "files": {ext: f"/api/file/{job}/plot.{ext}" for ext in ("gcode", "svg", "json")}}

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
                elif self.path.split("?")[0] in ("/layout", "/layout.html"):
                    self._send(200, (HERE / "ui_layout.html").read_bytes(), "text/html; charset=utf-8")
                elif self.path.split("?")[0] in ("/compose", "/compose.html"):
                    self._send(200, (HERE / "ui_compose.html").read_bytes(), "text/html; charset=utf-8")
                elif parts == ["api", "jobs"]:
                    self._send(200, app.jobs())
                elif parts == ["api", "presets", "scribble"]:
                    from dataclasses import asdict
                    from .scribble import ScribbleConfig, PRESETS
                    self._send(200, {"defaults": asdict(ScribbleConfig()), "presets": PRESETS})
                elif parts[:2] == ["api", "facemap"] and len(parts) == 3:
                    png, info = app.face_map(parts[2])
                    self._send(200, png, "image/png", {"X-Faces": str(info["faces"])})
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
                elif self.path == "/api/layout/load":
                    self._send(200, app.layout_load(json.loads(body)))
                elif self.path == "/api/layout/export":
                    self._send(200, app.layout_export(json.loads(body)))
                elif self.path == "/api/suggest":
                    self._send(200, app.suggest(json.loads(body)))
                elif self.path == "/api/compose/item":
                    self._send(200, app.compose_item(json.loads(body)))
                elif self.path == "/api/compose/export":
                    self._send(200, app.compose_export(json.loads(body)))
                elif self.path.startswith("/api/mask/"):
                    self._send(200, app.save_mask(self.path.rsplit("/", 1)[1], body))
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
