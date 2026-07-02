"""Interactive manual eddy review (Keep / Reject / Skip) for one SWOT pass.

Shows one candidate eddy at a time as a SLA + ADT zoom (true 2-km pixel aspect, same
conventions as the notebook panels) with Keep/Reject/Skip/Back buttons, and writes every
decision to a resumable CSV keyed by (cycle, pass, eddy). Re-launching the reviewer resumes
at the first un-reviewed eddy and pre-loads prior verdicts.

Usage (in the notebook, after detect):
    from swot_eddy.review import EddyReviewer
    EddyReviewer(dets, full, pad_left, cycle="cycle_004", pass_name=path.stem,
                 labels_path=Path(cfg["out_root"]) / "labels" / "decisions.csv").show()
"""
from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np

try:                                               # cartopy -> coastline + lat/lon gridlines on (e)/(f)
    import cartopy.crs as ccrs
    _PC = ccrs.PlateCarree()
    _HAS_CARTOPY = True
except Exception:                                  # no cartopy -> plain lon/lat axes fallback
    _PC = None
    _HAS_CARTOPY = False

def _bind_hotkeys(mapping, uid):
    """Wire keyboard shortcuts to ipywidgets Buttons with no extra deps. `mapping` is
    {js_key_name: Button} (e.g. "k", "1", "ArrowLeft"). Tags each button with a unique CSS
    class and installs a document-level `keydown` listener (via injected JS) that `.click()`s
    the matching button -- so the on_click handler fires in the kernel exactly as a mouse
    click would. Works in Jupyter and Colab; no focus-first needed. No-op if not in IPython."""
    try:
        from IPython.display import Javascript, display
    except Exception:
        return
    import json
    cls_map = {}
    for i, (key, btn) in enumerate(mapping.items()):
        c = f"ek-{uid}-{i}"
        btn.add_class(c)
        cls_map[key] = c
    js = """
    (function(){
      var H='__eddyKey_%s';
      if(window[H]) document.removeEventListener('keydown', window[H], true);
      var map=%s;
      window[H]=function(e){
        if(e.target && /^(input|textarea|select)$/i.test(e.target.tagName)) return;
        var c=map[e.key]; if(!c) return;
        var b=document.querySelector('.'+c);
        if(b){ e.preventDefault(); e.stopPropagation(); b.click(); }
      };
      document.addEventListener('keydown', window[H], true);
    })();
    """ % (uid, json.dumps(cls_map))
    display(Javascript(js))


DECISION_COLS = [
    "cycle", "pass_name", "eddy", "polarity", "center_line", "center_col",
    "lon", "lat", "radius_km", "amp_m", "gap_reject", "nadir", "verdict", "ts",
]


class EddyReviewer:
    """One-eddy-at-a-time approval UI backed by a resumable decisions CSV."""

    def _pad_cols(self, arr):
        """Place a native (L x 69) array into the 128-col frame so it crops like SLA/ADT."""
        out = np.full((self.nL, self.nC), np.nan)
        w = min(arr.shape[1], self.nC - self.pad_left)
        out[:, self.pad_left:self.pad_left + w] = arr[:, :w]
        return out

    def __init__(self, dets, full, pad_left, cycle, pass_name, labels_path,
                 speed_a=None, uv=None, uv_full=None, edge_cols=None, geo=None,
                 labels=None,
                 sla_scale=(-10, 30), adt_scale=(-0.1, 0.8), speed_scale=(0, 0.6),
                 speedfull_scale=(0, 1.5)):
        # labels=None -> default Keep/Reject/Skip buttons (k/r/s). labels=(...)-> one button
        # per class numbered 1..N (hotkeys 1..N) + Skip (s); the verdict stored IS the label.
        self.labels = list(labels) if labels else None
        # geo = (glon, glat, gsla_cm, gadt_m): native lon/lat + fields for the geographic
        # panels (e) SSHA & (f) ADT on the REAL curved swath (with coastline). None -> skip.
        self.geo = geo
        self.dets = list(dets)
        self.sla = full[0] * 100.0                 # cm
        self.adt = full[1]                         # m
        self.nL, self.nC = self.sla.shape
        # Pixel panels use origin="lower" (line index up). On a descending pass latitude
        # DECREASES with line index, so north would end up at the bottom -- flip the y-axis so
        # the pixel zooms match the geographic (north-up) panels. Detect from the geo latitudes.
        # Likewise across-track: on a descending pass longitude DECREASES with column index, so
        # east ends up on the left. Flip both axes (a full 180deg view rotation) -> north-up,
        # east-right, AND the quiver vectors come out geographically correct again.
        self.north_down = self.east_down = False
        if geo is not None:
            lat = np.asarray(geo[1], dtype="float64")
            latcol = lat[:, lat.shape[1] // 2]
            idx = np.where(np.isfinite(latcol))[0]
            if idx.size > 2:
                self.north_down = latcol[idx[-1]] < latcol[idx[0]]
            lon = np.asarray(geo[0], dtype="float64")
            loncol = np.nanmean(lon, axis=0)
            cidx = np.where(np.isfinite(loncol))[0]
            if cidx.size > 2:
                self.east_down = loncol[cidx[-1]] < loncol[cidx[0]]
        self.pad_left = pad_left
        self.cycle = str(cycle)
        self.pass_name = str(pass_name)
        self.labels_path = Path(labels_path)
        self.sla_vmin, self.sla_vmax = sla_scale
        self.adt_vmin, self.adt_vmax = adt_scale
        self.speed_vmin, self.speed_vmax = speed_scale
        # geostrophic velocity anomaly (native L x 69) -> pad into the 128-col frame so it
        # crops like SLA/ADT. uv=(u,v) adds quiver vectors (anomaly direction = rotation);
        # speed_a alone gives just the magnitude background. VIZ only; detection unaffected.
        self.u = self.v = self.speed = None
        if uv is not None:
            self.u, self.v = self._pad_cols(uv[0]), self._pad_cols(uv[1])
            self.speed = np.hypot(self.u, self.v)
        elif speed_a is not None:
            self.speed = self._pad_cols(speed_a)
        # full geostrophic velocity (ugos/vgos, includes the mean flow) -> panel (d), like (c)
        self.speed_full_vmin, self.speed_full_vmax = speedfull_scale
        self.u2 = self.v2 = self.speed2 = None
        if uv_full is not None:
            self.u2, self.v2 = self._pad_cols(uv_full[0]), self._pad_cols(uv_full[1])
            self.speed2 = np.hypot(self.u2, self.v2)
        # 4 SWOT swath-edge lines (native cols -> padded frame); default to the pad bounds.
        self.edges = [pad_left + c for c in edge_cols] if edge_cols is not None else [pad_left, pad_left + 69]
        self.verdicts = self._load()               # {eddy_idx: verdict}
        self.i = self._first_unreviewed()

    # ---- persistence -------------------------------------------------------
    def _load(self) -> dict:
        import pandas as pd
        if not self.labels_path.exists():
            return {}
        df = pd.read_csv(self.labels_path)
        mine = df[(df["cycle"] == self.cycle) & (df["pass_name"] == self.pass_name)]
        return {int(r.eddy): str(r.verdict) for r in mine.itertuples()}

    def _save_row(self, eddy_idx, verdict):
        import pandas as pd
        d = self.dets[eddy_idx]
        row = {
            "cycle": self.cycle, "pass_name": self.pass_name, "eddy": eddy_idx,
            "polarity": d.polarity, "center_line": d.center_line, "center_col": d.center_col,
            "lon": None if d.center_lon is None else round(d.center_lon, 4),
            "lat": None if d.center_lat is None else round(d.center_lat, 4),
            "radius_km": round(d.radius_km, 2), "amp_m": round(d.amplitude_m, 4),
            "gap_reject": bool(d.gap_reject), "nadir": bool(d.center_in_nadir),
            "verdict": verdict, "ts": dt.datetime.now().isoformat(timespec="seconds"),
        }
        self.labels_path.parent.mkdir(parents=True, exist_ok=True)
        if self.labels_path.exists():
            df = pd.read_csv(self.labels_path)
            mask = (df["cycle"] == self.cycle) & (df["pass_name"] == self.pass_name) & (df["eddy"] == eddy_idx)
            df = df[~mask]
            df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        else:
            df = pd.DataFrame([row], columns=DECISION_COLS)
        df.to_csv(self.labels_path, index=False)

    def _first_unreviewed(self) -> int:
        for k in range(len(self.dets)):
            if k not in self.verdicts:
                return k
        return 0

    # ---- drawing -----------------------------------------------------------
    def _draw(self):
        import matplotlib.pyplot as plt
        d = self.dets[self.i]
        r_px = max(d.radius_km / 2.0, 6)
        half = int(round(max(2.5 * r_px, 18)))
        i0, i1 = max(0, d.center_line - half), min(self.nL, d.center_line + half + 1)
        j0, j1 = max(0, d.center_col - half),  min(self.nC, d.center_col + half + 1)
        col = "red" if d.polarity == "anticyclonic" else "blue"
        cls = "--" if d.gap_reject else "-"
        mk = "x" if d.center_in_nadir else ("s" if d.gap_reject else "o")
        cont = np.array(d.contour_pix) if d.contour_pix else None

        # wide context window (~4x the zoom) so eddy size is judged against its surroundings
        half_w = int(min(max(half * 4, 90), self.nL // 2))
        iw0, iw1 = max(0, d.center_line - half_w), min(self.nL, d.center_line + half_w + 1)
        jw0, jw1 = max(0, d.center_col - half_w),  min(self.nC, d.center_col + half_w + 1)

        panels = [
            dict(fld=self.sla, cmap="RdBu_r", vmin=self.sla_vmin, vmax=self.sla_vmax,
                 lbl="SLA (cm)", base="SLA (detection field)", qu=None, qv=None, qs=None, wide=False),
            dict(fld=self.adt, cmap="Spectral_r", vmin=self.adt_vmin, vmax=self.adt_vmax,
                 lbl="ADT (m)", base="ADT (reference)", qu=None, qv=None, qs=None, wide=False),
        ]
        if self.speed is not None:                     # (c) geostrophic velocity ANOMALY
            t = "geos. velocity anomaly" if self.u is not None else "geos. speed anomaly"
            panels.append(dict(fld=self.speed, cmap="turbo", vmin=self.speed_vmin, vmax=self.speed_vmax,
                               lbl="|u'| (m/s)", base=t, qu=self.u, qv=self.v, qs=0.06, wide=False))
        if self.speed2 is not None:                    # (d) FULL geostrophic velocity (ugos/vgos)
            panels.append(dict(fld=self.speed2, cmap="turbo", vmin=self.speed_full_vmin,
                               vmax=self.speed_full_vmax, lbl="|u| (m/s)",
                               base="geos. velocity (ugos/vgos)", qu=self.u2, qv=self.v2, qs=0.12, wide=False))
        panels.append(dict(fld=self.sla, cmap="RdBu_r", vmin=self.sla_vmin, vmax=self.sla_vmax,
                           lbl="SLA (cm)", base=f"wide context (±{half_w} px)",
                           qu=None, qv=None, qs=None, wide=True))
        for i, p in enumerate(panels):                 # letters follow panel order (a, b, c, d, e, ...)
            p["ttl"] = f"({chr(ord('a') + i)}) {p['base']}"
        n = len(panels)
        has_geo = self.geo is not None and d.center_lon is not None
        # geo row: (e) SSHA, (f) ADT, and (g) geos. speed if velocity is in the geo tuple
        ngeo = (3 if (has_geo and len(self.geo) >= 6 and self.geo[4] is not None) else 2) if has_geo else 0
        if has_geo:
            # top row = zoom panels (a..d); bottom row = wide context (pixel) + geo panels
            top_panels, wide_panel = panels[:-1], panels[-1]
            ntop, nbot = len(top_panels), 1 + ngeo
            fig = plt.figure(figsize=(4 * max(ntop, nbot), 9.6))
            outer = fig.add_gridspec(2, 1, height_ratios=[1.0, 1.4])
            top, bot = outer[0].subgridspec(1, ntop), outer[1].subgridspec(1, nbot)
            top_axes = [fig.add_subplot(top[0, c]) for c in range(ntop)]
            wide_ax = fig.add_subplot(bot[0, 0])                  # pixel axis (no map projection)
            geo_kw = {"projection": _PC} if _HAS_CARTOPY else {}  # GeoAxes (coastline) or plain
            geo_axes = [fig.add_subplot(bot[0, 1 + c], **geo_kw) for c in range(ngeo)]
            pix_axes, pix_panels = top_axes + [wide_ax], top_panels + [wide_panel]
        else:
            fig, pix_axes = plt.subplots(1, n, figsize=(4 * n, 4.6))
            pix_axes, pix_panels = list(np.atleast_1d(pix_axes)), panels
        for a, p in zip(pix_axes, pix_panels):
            fld, cmap, vmin, vmax, lbl, ttl, wide = (p["fld"], p["cmap"], p["vmin"], p["vmax"],
                                                     p["lbl"], p["ttl"], p["wide"])
            y0, y1, x0, x1 = (iw0, iw1, jw0, jw1) if wide else (i0, i1, j0, j1)
            im = a.imshow(fld[y0:y1, x0:x1], cmap=cmap, vmin=vmin, vmax=vmax,
                          origin="lower", aspect="equal", extent=[x0, x1, y0, y1])
            if p["qu"] is not None:                    # geographic (east, north) vectors drawn on the
                step = max(1, (x1 - x0) // 22)         # straightened col/line grid. The N-up/E-right
                ys, xs = np.arange(y0, y1, step), np.arange(x0, x1, step)   # axis flips would otherwise
                XX, YY = np.meshgrid(xs, ys)           # reverse the arrows -> counter them so an east
                sx = -1.0 if self.east_down else 1.0   # vector always points screen-right and a north
                sy = -1.0 if self.north_down else 1.0  # vector screen-up (matches the panel orientation).
                a.quiver(XX, YY, sx * p["qu"][y0:y1:step, x0:x1:step], sy * p["qv"][y0:y1:step, x0:x1:step],
                         color="k", angles="xy", scale_units="xy", scale=p["qs"], width=0.004, pivot="mid")
            for xx in self.edges:                  # 4 SWOT swath-edge lines (outer + nadir-gap)
                if x0 <= xx <= x1:
                    a.axvline(xx, color="k", lw=0.8)
            if cont is not None:
                a.plot(cont[:, 1], cont[:, 0], color=col, lw=1.5, ls=cls)
            a.plot(d.center_col, d.center_line, mk, color=col, ms=9, mfc="none", mew=1.8)
            if wide:                                   # outline the zoom window on the context panel
                from matplotlib.patches import Rectangle
                a.add_patch(Rectangle((j0, i0), j1 - j0, i1 - i0, fill=False,
                                      edgecolor="k", lw=1.0, ls="--"))
            a.set_xlim((x1, x0) if self.east_down else (x0, x1))    # east right
            a.set_ylim((y1, y0) if self.north_down else (y0, y1))   # north up
            a.set_xlabel("across-track px (E right)"); a.set_ylabel("along-track line (N up)")
            a.set_title(ttl, fontsize=10)
            fig.colorbar(im, ax=a, shrink=0.8, pad=0.02).set_label(lbl)
        pix_axes[0].title.set_color(col)
        if has_geo:                                # geo row letters continue after the pixel panels
            self._draw_geo(d, col, geo_axes, base=n)
        loc = "lon/lat n/a (pad)" if d.center_lon is None else f"{d.center_lon - 360:.2f}°E, {d.center_lat:.2f}°N"
        flags = []
        if d.gap_reject:      flags.append(f"GAP {d.gap_frac:.0%}")
        if d.center_in_nadir: flags.append("NADIR")
        ftag = ("  FLAGS: " + ", ".join(flags)) if flags else ""
        fig.suptitle(f"#{self.i} {d.polarity}  {loc}  R={d.radius_km:.0f}km  "
                     f"amp={d.amplitude_m*100:.1f}cm{ftag}", fontsize=10, y=1.02)
        plt.tight_layout()
        plt.show()                                 # ONE show -> a-f all render inside the Output

    def _coastline(self, ax, ext):
        """Best-effort black coastline on a plain lon/lat axis (cartopy data if available)."""
        try:
            import cartopy.feature as cfeature
            for geom in cfeature.COASTLINE.intersecting_geometries(ext):
                for ls in (geom.geoms if hasattr(geom, "geoms") else [geom]):
                    xy = np.asarray(ls.coords)
                    if len(xy):
                        ax.plot(xy[:, 0], xy[:, 1], color="k", lw=0.7, zorder=4)
        except Exception:
            pass                                              # no cartopy / no data -> skip coast

    def _draw_geo(self, d, col, axes, base=4):
        """Geo-row panels (lettered from `base`, i.e. continuing after the pixel panels): SSHA,
        ADT and -- if velocity is in the geo tuple -- geostrophic speed anomaly + vectors, on
        the FULL native swath across the whole crop box, with
        coastline + lat/lon gridlines (cartopy GeoAxes, like F1_v52_batch), the current eddy's
        contour + center marked -- so you see WHERE the eddy sits in the box relative to coast
        and front. Drawn into the provided `axes` (same figure as a-d) so one plt.show() renders
        everything (a separate figure was being dropped inside the ipywidgets Output). Falls
        back to plain lon/lat axes if cartopy is unavailable."""
        glon, glat, gsla, gadt = self.geo[:4]
        gu, gv = (self.geo[4], self.geo[5]) if len(self.geo) >= 6 else (None, None)
        fig = axes[0].figure
        ext = [float(np.nanmin(glon)), float(np.nanmax(glon)),          # whole crop box
               float(np.nanmin(glat)), float(np.nanmax(glat))]
        clon = ((d.center_lon + 180) % 360) - 180
        clat = d.center_lat
        cl = [p for p in d.contour_lonlat if p is not None]
        contour = np.array(cl) if cl else None
        geo = _HAS_CARTOPY and hasattr(axes[0], "set_extent")          # GeoAxes -> coastline path
        tkw = {"transform": _PC} if geo else {}
        L = lambda k: chr(ord("a") + base + k)                         # geo-row letters from `base`
        panels = [(gsla, "RdBu_r", -10, 30, "SLA (cm)", f"({L(0)}) SSHA — full swath in box", False),
                  (gadt, "Spectral_r", -0.1, 0.8, "ADT (m)", f"({L(1)}) ADT — full swath in box", False)]
        if gu is not None and len(axes) >= 3:                          # geostrophic speed anomaly
            panels.append((np.hypot(gu, gv), "turbo", 0, 0.6, "|u'| (m/s)",
                           f"({L(2)}) geos. speed anomaly", True))
        for k, (ax, (fld, cmap, vmin, vmax, lbl, ttl, quiver)) in enumerate(zip(axes, panels)):
            pm = ax.pcolormesh(glon, glat, fld, cmap=cmap, vmin=vmin, vmax=vmax,
                               shading="nearest", zorder=2, **tkw)
            if quiver:                                                 # anomaly velocity vectors
                s = max(1, glon.shape[0] // 45)
                ax.quiver(glon[::s, ::s], glat[::s, ::s], gu[::s, ::s], gv[::s, ::s],
                          color="k", scale=15, width=0.002, pivot="mid", zorder=4, **tkw)
            if geo:
                ax.set_extent(ext, crs=_PC)
                try:
                    ax.coastlines("10m", linewidth=0.6, color="k")
                except Exception:
                    pass                                               # coastline data missing -> skip
                gl = ax.gridlines(draw_labels=True, linewidth=0.3, color="gray",
                                  alpha=0.5, linestyle=":")
                gl.top_labels = gl.right_labels = False
            else:                                                      # plain lon/lat fallback
                self._coastline(ax, ext)
                ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3])
                ax.set_aspect(1 / np.cos(np.deg2rad(clat)))
                ax.set_xlabel("longitude"); ax.set_ylabel("latitude")
            if contour is not None:
                ax.plot(((contour[:, 0] + 180) % 360) - 180, contour[:, 1], color=col, lw=1.6,
                        zorder=5, **tkw)
            ax.plot(clon, clat, "o", color=col, ms=11, mfc="none", mew=2.2, zorder=6, **tkw)
            ax.set_title(ttl, fontsize=10, color=(col if k == 0 else "black"))
            fig.colorbar(pm, ax=ax, shrink=0.85, pad=0.02).set_label(lbl)

    def _counts(self):
        v = self.verdicts.values()
        return sum(x == "keep" for x in v), sum(x == "reject" for x in v), sum(x == "skip" for x in v)

    # ---- UI ----------------------------------------------------------------
    def show(self):
        try:
            import ipywidgets as widgets
            from IPython.display import display, clear_output
        except ImportError as e:
            raise ImportError("ipywidgets + IPython required (pip install ipywidgets)") from e
        if not self.dets:
            print("no detections to review"); return

        out = widgets.Output()
        status = widgets.HTML()

        def render():
            with out:
                clear_output(wait=True)
                self._draw()
            k, r, s = self._counts()
            cur = self.verdicts.get(self.i, "—")
            status.value = (f"<b>eddy {self.i+1}/{len(self.dets)}</b> "
                            f"(current verdict: <code>{cur}</code>) &nbsp;|&nbsp; "
                            f"reviewed {len(self.verdicts)}/{len(self.dets)} "
                            f"— keep {k}, reject {r}, skip {s} &nbsp;|&nbsp; "
                            f"<code>{self.labels_path}</code>")

        def record(verdict):
            self.verdicts[self.i] = verdict
            self._save_row(self.i, verdict)
            self.i = min(self.i + 1, len(self.dets) - 1)
            render()

        def go(delta):
            self.i = min(max(self.i + delta, 0), len(self.dets) - 1)
            render()

        b_back = widgets.Button(description="Back (left)")
        b_next = widgets.Button(description="Next (right)")
        b_back.on_click(lambda _: go(-1))
        b_next.on_click(lambda _: go(+1))
        hot = {"ArrowLeft": b_back, "ArrowRight": b_next}

        if self.labels:                                  # class buttons: one per label (1..N) + Skip
            styles = {"reject": "danger"}
            label_btns = []
            for i, opt in enumerate(self.labels):        # number each label -> hotkey 1..9
                b = widgets.Button(description=f"{opt} ({i+1})", button_style=styles.get(opt, "success"))
                b.on_click(lambda _w, o=opt: record(o))
                label_btns.append(b); hot[str(i + 1)] = b
            b_skip = widgets.Button(description="Skip (s)", button_style="warning")
            b_skip.on_click(lambda _: record("skip")); hot["s"] = b_skip
            row = [b_back] + label_btns + [b_skip, b_next]
        else:                                            # default Keep / Reject / Skip
            b_keep = widgets.Button(description="Keep (k)", button_style="success")
            b_rej  = widgets.Button(description="Reject (r)", button_style="danger")
            b_skip = widgets.Button(description="Skip (s)", button_style="warning")
            b_keep.on_click(lambda _: record("keep"))
            b_rej.on_click(lambda _: record("reject"))
            b_skip.on_click(lambda _: record("skip"))
            hot.update({"k": b_keep, "r": b_rej, "s": b_skip})
            row = [b_back, b_keep, b_rej, b_skip, b_next]

        display(widgets.VBox([status, widgets.HBox(row), out]))
        _bind_hotkeys(hot, format(id(self) & 0xFFFFFF, "x"))   # keyboard shortcuts
        render()


def kept_detections(dets, cycle, pass_name, labels_path):
    """Return only the detections marked 'keep' in the decisions CSV for this pass."""
    import pandas as pd
    labels_path = Path(labels_path)
    if not labels_path.exists():
        return []
    df = pd.read_csv(labels_path)
    keep = df[(df["cycle"] == str(cycle)) & (df["pass_name"] == str(pass_name))
              & (df["verdict"] == "keep")]
    idx = set(int(e) for e in keep["eddy"])
    return [d for k, d in enumerate(dets) if k in idx]


LABEL_COLS = ["cycle", "pass", "kind", "eddy", "polarity", "lon", "lat",
              "radius_km", "amplitude_m", "label", "ts"]


class CatalogLabeler:
    """Walk EVERY detection across ALL passes, one eddy at a time, and assign a class label
    (e.g. frontal / non-frontal / reject) for ML training. TWO PHASES: first all the KEPT
    (good) eddies across every pass, THEN all the auto-REJECTED ones (nadir/gap blobs) as
    negatives -- so you label clean data first and slog through false positives last.

    Polarity is NOT hand-labeled: PET assigns it from SLA sign (negative=cyclonic,
    positive=anticyclonic) and it is shown (red/blue + word) so you can eyeball sign flips.

    Resumable: labels -> one global CSV keyed (cycle, pass, kind, eddy). Re-launching jumps to
    the first unlabeled eddy and SKIPS fully-labeled (pass, kind) slots without re-detecting,
    using the run_all manifest's n_kept / n_rejected. Each pass is detected once and cached.

    Usage (notebook):
        from swot_eddy.review import CatalogLabeler
        CatalogLabeler(cfg, labels=("frontal", "non-frontal", "reject")).show()
    """

    def __init__(self, cfg, labels_path=None,
                 labels=("frontal", "coastal", "open ocean", "reject"),
                 cycles=None):
        from swot_eddy.aoi import load_aoi
        self.cfg = cfg
        self.region = cfg["region"]                    # AOI5 envelope (crop band)
        self.aoi = load_aoi(cfg)                        # polygon for the detection filter
        self.var_map = cfg["product"]["variables"][cfg["product"]["product_version"]]
        self.data_root = Path(cfg["data_root"])
        self.out_root = Path(cfg["out_root"])
        self.labels_path = (Path(labels_path) if labels_path
                            else self.out_root / "aoi5" / "labels" / "eddy_labels_aoi5.csv")
        self.label_opts = list(labels)
        self.passes = self._build_passlist(cycles)     # [(cycle, Path, n_kept, n_rejected), ...]
        # phase 1 = every pass's KEPT eddies, then phase 2 = every pass's REJECTED eddies
        self.slots = [(pi, "keep") for pi in range(len(self.passes))] \
            + [(pi, "reject") for pi in range(len(self.passes))]
        self.records = self._load()                    # {(cycle, pnum, kind, eddy): label}
        self._cache = (None, None)                     # (pass_index, payload)
        self.si, self.ei = 0, 0                        # slot index, eddy index
        self._goto_first_unlabeled()

    # ---- work list + persistence ------------------------------------------
    def _build_passlist(self, cycles):
        import pandas as pd
        man = self.out_root / "aoi5" / "catalog" / "passes_done.csv"
        pl = []
        if man.exists():                               # fast path: only passes that have eddies
            m = pd.read_csv(man, dtype=str)
            has_rej = "n_rejected" in m.columns
            keep = (m["n_kept"].astype(int) > 0)
            if has_rej:
                keep = keep | (m["n_rejected"].astype(int) > 0)
            m = m[keep]
            for i, (cyc, pas, nk) in enumerate(zip(m["cycle"], m["pass"], m["n_kept"])):
                if cycles and int(cyc.split("_")[1]) not in cycles:
                    continue
                fs = list((self.data_root / cyc).glob(f"*_{pas}_*.nc"))
                if fs:
                    nr = int(m["n_rejected"].iloc[i]) if has_rej else None
                    pl.append((cyc, fs[0], int(nk), nr))
        else:                                          # no manifest yet: walk every pass
            for cd in sorted(self.data_root.glob("cycle_*")):
                if cycles and int(cd.name.split("_")[1]) not in cycles:
                    continue
                for f in sorted(cd.glob("*.nc")):
                    pl.append((cd.name, f, None, None))
        return pl

    @staticmethod
    def _pnum(f):
        import re
        m = re.search(r"_(\d{3})_(\d{3})_", f.name)
        return m.group(2) if m else f.stem

    def _load(self):
        import pandas as pd
        if not self.labels_path.exists():
            return {}
        df = pd.read_csv(self.labels_path, dtype={"cycle": str, "pass": str})
        kinds = df["kind"] if "kind" in df.columns else ["keep"] * len(df)   # back-compat
        return {(str(c), str(p), str(k), int(e)): str(lab)
                for c, p, k, e, lab in zip(df["cycle"], df["pass"], kinds, df["eddy"], df["label"])}

    def _save_row(self, key, d, label):
        import pandas as pd
        cycle, pnum, kind, eddy = key
        row = {
            "cycle": cycle, "pass": pnum, "kind": kind, "eddy": eddy, "polarity": d.polarity,
            "lon": None if d.center_lon is None else round(((d.center_lon + 180) % 360) - 180, 4),
            "lat": None if d.center_lat is None else round(d.center_lat, 4),
            "radius_km": round(d.radius_km, 2), "amplitude_m": round(d.amplitude_m, 4),
            "label": label, "ts": dt.datetime.now().isoformat(timespec="seconds"),
        }
        self.labels_path.parent.mkdir(parents=True, exist_ok=True)
        if self.labels_path.exists():
            df = pd.read_csv(self.labels_path, dtype={"cycle": str, "pass": str})
            if "kind" not in df.columns:
                df["kind"] = "keep"
            mask = ((df["cycle"] == cycle) & (df["pass"].astype(str) == str(pnum))
                    & (df["kind"] == kind) & (df["eddy"] == eddy))
            df = pd.concat([df[~mask], pd.DataFrame([row])], ignore_index=True)
        else:
            df = pd.DataFrame([row], columns=LABEL_COLS)
        df.to_csv(self.labels_path, index=False)

    # ---- per-pass data (cached) -------------------------------------------
    def _in_box(self, d):
        """Detection center inside the GS_AOI5 polygon (point-in-polygon, not the bbox)."""
        return self.aoi.contains(d.center_lon, d.center_lat)

    def _load_pass(self, pi):
        if self._cache[0] == pi:
            return self._cache[1]
        import datetime as _dt, re
        import xarray as xr
        from swot_eddy import preprocess as pp
        from swot_eddy.detect import detect_pass_split
        cycle, f, _nk, _nr = self.passes[pi]
        m = re.search(r"_(\d{8})T(\d{6})_", f.name)
        date = _dt.datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S") if m else _dt.datetime(2023, 1, 1)
        tiles, summ, clon, clat = pp.preprocess_pass(str(f), self.cfg)
        kept_all, rejected_all, _ = detect_pass_split(tiles, self.cfg, clon, clat, date)
        kept = [d for d in kept_all if self._in_box(d)]
        rejected = [d for d in rejected_all if self._in_box(d)]
        C, _, P = tiles[0].data.shape
        full = np.full((C, summ["n_lines_cropped"], P), np.nan)
        for t in tiles:
            h = t.data.shape[1]; e = min(t.line_start + h, full.shape[1])
            full[:, t.line_start:e, :] = t.data[:, :e - t.line_start, :]
        with xr.open_dataset(str(f)) as ds:
            dc, _ = pp.crop_to_region(ds, self.region)
            ua, va = pp.geostrophic_anomaly_uv(dc, self.var_map)
            edges = pp.swath_edge_cols(dc)
            vm = self.var_map
            uf = dc[vm["ugos"]].values.astype("float64")    # full geos. velocity (ugos/vgos) -> panel (d)
            vf = dc[vm["vgos"]].values.astype("float64")
            geo = (((dc.longitude.values + 180) % 360) - 180, dc.latitude.values,
                   dc[vm["ssha"]].values * 100.0,
                   (dc[vm["ssha_adt_base"]] + dc[vm["mdt"]]).values,
                   ua, va)                                  # ugosa/vgosa for geo-row speed panel
        payload = {"cycle": cycle, "pnum": self._pnum(f), "keep": kept, "reject": rejected,
                   "full": full, "uv": (ua, va), "uv_full": (uf, vf), "edges": edges, "geo": geo,
                   "pad_left": tiles[0].pad_left, "rv": {},
                   "date": date.strftime("%b %d %Y %H:%M UTC")}   # from filename timestamp
        self._cache = (pi, payload)
        return payload

    def _rv(self, p, kind):
        """Cached EddyReviewer for one (pass, kind) to reuse the 4-panel drawing."""
        if kind not in p["rv"]:
            p["rv"][kind] = EddyReviewer(
                p[kind], p["full"], p["pad_left"], uv=p["uv"], uv_full=p["uv_full"],
                edge_cols=p["edges"], geo=p["geo"], cycle=p["cycle"], pass_name=p["pnum"],
                labels_path=self.labels_path.parent / "__none__.csv")
        return p["rv"][kind]

    def _slot_count(self, si):
        """Number of eddies in a (pass, kind) slot -- from the manifest if possible (no detect)."""
        pi, kind = self.slots[si]
        _c, _f, nk, nr = self.passes[pi]
        cached = nk if kind == "keep" else nr
        if cached is not None:
            return cached
        return len(self._load_pass(pi)[kind])

    def _slot_labeled(self, si):
        pi, kind = self.slots[si]
        cyc, f, _nk, _nr = self.passes[pi]
        pnum = self._pnum(f)
        return sum(1 for (c, p, k, _e) in self.records if c == cyc and p == pnum and k == kind)

    def _goto_first_unlabeled(self):
        for si in range(len(self.slots)):
            if self._slot_labeled(si) >= self._slot_count(si):
                continue                               # fully labeled -> skip (no detect if counted)
            pi, kind = self.slots[si]
            p = self._load_pass(pi)
            for ei in range(len(p[kind])):
                if (p["cycle"], p["pnum"], kind, ei) not in self.records:
                    self.si, self.ei = si, ei
                    return
        self.si, self.ei = max(0, len(self.slots) - 1), 0   # all done

    # ---- navigation --------------------------------------------------------
    def _slot_list(self, si):
        pi, kind = self.slots[si]
        return self._load_pass(pi)[kind], kind

    def _advance(self, step):
        lst, _ = self._slot_list(self.si)
        ei = self.ei + step
        while True:
            if 0 <= ei < len(lst):
                self.ei = ei; return
            if step > 0:
                if self.si >= len(self.slots) - 1:
                    self.ei = max(0, len(lst) - 1); return
                self.si += 1; lst, _ = self._slot_list(self.si); ei = 0
            else:
                if self.si <= 0:
                    self.ei = 0; return
                self.si -= 1; lst, _ = self._slot_list(self.si); ei = len(lst) - 1

    # ---- UI ----------------------------------------------------------------
    def show(self):
        try:
            import ipywidgets as widgets
            from IPython.display import display, clear_output
        except ImportError as e:
            raise ImportError("ipywidgets + IPython required (pip install ipywidgets)") from e
        if not self.passes:
            print("no passes found (run scripts/run_all.py first, or pass cycles=...)"); return

        out, status = widgets.Output(), widgets.HTML()
        n_keep = sum(nk for _, _, nk, _ in self.passes if nk) or None
        n_rej = sum(nr for _, _, _, nr in self.passes if nr) or None

        def render():
            pi, kind = self.slots[self.si]
            p = self._load_pass(pi)
            lst = p[kind]
            with out:
                clear_output(wait=True)
                if lst:
                    rv = self._rv(p, kind); rv.i = self.ei; rv._draw()
                else:
                    print(f"{p['cycle']}/{p['pnum']} [{kind}]: no eddies")
            cur = self.records.get((p["cycle"], p["pnum"], kind, self.ei), "—")
            counts = {}
            for v in self.records.values():
                counts[v] = counts.get(v, 0) + 1
            tally = "  ".join(f"{k}:{v}" for k, v in sorted(counts.items()))
            phase = "1/2 KEEP (good eddies)" if kind == "keep" else "2/2 REJECT (negatives)"
            d = lst[self.ei] if lst else None
            polinfo = f" &nbsp; <b style='color:{'red' if d and d.polarity=='anticyclonic' else 'blue'}'>" \
                      f"{d.polarity if d else ''}</b> (auto, from SLA sign)" if d else ""
            status.value = (
                f"<b>phase {phase}</b> &nbsp; pass {pi+1}/{len(self.passes)} ({p['cycle']}/{p['pnum']}) "
                f"&nbsp; <b>{p['date']}</b> "
                f"&nbsp; <b>eddy {self.ei+1}/{max(1,len(lst))}</b>{polinfo} &nbsp; label: <code>{cur}</code><br>"
                f"labeled {len(self.records)}"
                + (f" &nbsp; (~{n_keep} keep + ~{n_rej} reject est)" if n_keep else "")
                + f" &nbsp; [{tally}] &nbsp; <code>{self.labels_path}</code>")

        def record(label):
            pi, kind = self.slots[self.si]
            p = self._load_pass(pi)
            if p[kind]:
                key = (p["cycle"], p["pnum"], kind, self.ei)
                self.records[key] = label
                self._save_row(key, p[kind][self.ei], label)
            self._advance(+1); render()

        def go(step):
            self._advance(step); render()

        btns = []
        styles = {"reject": "danger"}
        for i, opt in enumerate(self.label_opts):          # number each label -> hotkey 1..9
            b = widgets.Button(description=f"{opt} ({i+1})", button_style=styles.get(opt, "success"))
            b.on_click(lambda _w, o=opt: record(o))
            btns.append(b)
        b_skip = widgets.Button(description="Skip (s)")
        b_back = widgets.Button(description="Back (left)")
        b_next = widgets.Button(description="Next (right)")
        b_skip.on_click(lambda _: record("skip"))
        b_back.on_click(lambda _: go(-1))
        b_next.on_click(lambda _: go(+1))

        hot = {str(i + 1): b for i, b in enumerate(btns)}  # 1..9 = labels, s = skip, arrows = nav
        hot.update({"s": b_skip, "ArrowLeft": b_back, "ArrowRight": b_next})
        display(widgets.VBox([status,
                              widgets.HBox([b_back] + btns + [b_skip, b_next]),
                              out]))
        _bind_hotkeys(hot, format(id(self) & 0xFFFFFF, "x"))
        render()
