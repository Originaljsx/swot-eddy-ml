"""Run the v2 frontal-eddy U-Net on a regridded SWOT GeoZarr store (swot-viewer data).

Reads pass slices from the pyramid store (zarr v3 over HTTP/S3; level-0 arrays
`0/ssha`, `0/adt`, 1-D `0/latitude` (ascending) / `0/longitude`, `slices.json` manifest),
builds the model's 4-channel input (ssha, adt, |grad adt|, validity), tiles 128x128 with
64-px overlap in BOTH dimensions, and writes detections as GeoJSON polygons for MapLibre
(feature properties: id, lon, lat, max_p, area_km2, radius_km, cycle, pass, date, time).

CAVEAT (documented distribution shift): the model was trained on L3 swath-frame passes
(along-track rows, 2 km pixels); this store is L2 regridded to a 0.02 deg lat/lon
lattice. Results are experimental until visually vetted.

Run with an environment that has zarr>=3 AND torch (the swot-250m-geozarr venv):

  # one pass:
  .../swot-250m-geozarr/.venv/bin/python scripts/predict_zarr.py --cycle 42 --pass 91 --png
  # every slice in the store (resumable; writes combined all_eddies.geojson):
  .../swot-250m-geozarr/.venv/bin/python scripts/predict_zarr.py --all

Outputs in out/aoi5/predictions/:
  zarr_c<cycle>_p<pass>_eddies.geojson   single-slice mode
  all_eddies.geojson                     --all mode (one FeatureCollection, filter by
                                         cycle/pass/date properties in the viewer)
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

DEFAULT_STORE = "https://d1ef9node0gwi2.cloudfront.net/c42_43_expert_pyramid_v2.zarr"
DEFAULT_BBOX = (-82.5, 23.0, -68.0, 41.0)          # GS_AOI5 envelope-ish
KM_PER_DEG_LAT = 110.574


def tile2d(x, tile=128, stride=64):
    """Yield (i0, j0, (C,tile,tile)) windows covering (C,H,W), edges snapped."""
    C, H, W = x.shape
    ii = list(range(0, max(H - tile, 0) + 1, stride)) or [0]
    jj = list(range(0, max(W - tile, 0) + 1, stride)) or [0]
    if ii[-1] != max(H - tile, 0):
        ii.append(max(H - tile, 0))
    if jj[-1] != max(W - tile, 0):
        jj.append(max(W - tile, 0))
    for i0 in ii:
        for j0 in jj:
            w = x[:, i0:i0 + tile, j0:j0 + tile]
            if w.shape[1] < tile or w.shape[2] < tile:
                w = np.pad(w, ((0, 0), (0, tile - w.shape[1]), (0, tile - w.shape[2])),
                           constant_values=np.nan)
            yield i0, j0, w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", default=DEFAULT_STORE)
    ap.add_argument("--slice", type=int, default=None, help="time index (see slices.json)")
    ap.add_argument("--cycle", type=int, default=None)
    ap.add_argument("--pass", dest="pnum", type=int, default=None)
    ap.add_argument("--all", action="store_true",
                    help="every slice in the store -> one combined all_eddies.geojson "
                         "(resumable via checkpoint)")
    ap.add_argument("--bbox", nargs=4, type=float, default=DEFAULT_BBOX,
                    metavar=("LON0", "LAT0", "LON1", "LAT1"))
    ap.add_argument("--weights", default=None)
    ap.add_argument("--thresh", type=float, default=None,
                    help="default: conformal lam_hat_v2")
    ap.add_argument("--png", action="store_true", help="also write a quicklook PNG")
    args = ap.parse_args()

    import torch
    import zarr
    from scipy import ndimage
    from skimage import measure
    from swot_eddy.unet import build_unet, normalize, TILE

    root = Path(__file__).resolve().parents[2]
    out_dir = root / "out/aoi5/predictions"
    out_dir.mkdir(parents=True, exist_ok=True)
    wpath = Path(args.weights) if args.weights else root / "out/aoi5/model/unet_frontal_v2.pt"
    thresh = args.thresh if args.thresh is not None else \
        float(np.load(root / "out/aoi5/model/lam_hat_v2.npy")[0])
    model = build_unet(in_ch=4)
    model.load_state_dict(torch.load(wpath, map_location="cpu", weights_only=True))
    model.eval()

    slices = json.load(urllib.request.urlopen(args.store + "/slices.json"))
    g = zarr.open_group(args.store, mode="r")
    lat = np.asarray(g["0/latitude"][:])
    lon = np.asarray(g["0/longitude"][:])
    lon0, lat0, lon1, lat1 = args.bbox
    y0, y1 = np.searchsorted(lat, [lat0, lat1])
    x0, x1 = np.searchsorted(lon, [lon0, lon1])
    lat_s, lon_s = lat[y0:y1], lon[x0:x1]
    dlat = float(np.median(np.diff(lat_s))) * KM_PER_DEG_LAT
    dlon = float(np.median(np.diff(lon_s))) * KM_PER_DEG_LAT * \
        float(np.cos(np.deg2rad(np.mean(lat_s))))

    def predict_slice(meta):
        """Features (+ debug fields) for one time slice; [] when swath misses the bbox."""
        t = meta["i"]
        ssha = np.asarray(g["0/ssha"][t, y0:y1, x0:x1], dtype="float64")
        finite = np.isfinite(ssha)
        if finite.mean() < 0.003:
            return [], None, None
        adt = np.asarray(g["0/adt"][t, y0:y1, x0:x1], dtype="float64")
        gy, gx = np.gradient(adt)
        grad = np.hypot(gy / dlat, gx / dlon)
        x = np.stack([ssha, adt, grad, finite.astype("float64")])

        prob = np.zeros(ssha.shape, np.float32)
        hits = np.zeros(ssha.shape, np.float32)
        with torch.no_grad():
            for i0, j0, w in tile2d(x, TILE, 64):
                if not np.isfinite(w[0]).any():
                    continue
                p = torch.sigmoid(model(torch.from_numpy(normalize(w))[None]))[0, 0].numpy()
                ie, je = min(i0 + TILE, prob.shape[0]), min(j0 + TILE, prob.shape[1])
                prob[i0:ie, j0:je] += p[:ie - i0, :je - j0]
                hits[i0:ie, j0:je] += 1
        prob /= np.maximum(hits, 1)
        mask = (prob > thresh) & finite

        lbl, n = ndimage.label(mask)
        feats = []
        for k in range(1, n + 1):
            blob = lbl == k
            area_km2 = float(blob.sum()) * dlat * dlon
            if area_km2 < 50:                      # sub-grid speckle
                continue
            cy, cx = (int(v) for v in ndimage.center_of_mass(blob))
            rings = []
            for c in measure.find_contours(blob.astype(float), 0.5):
                ring = [[round(float(np.interp(p[1], np.arange(len(lon_s)), lon_s)), 4),
                         round(float(np.interp(p[0], np.arange(len(lat_s)), lat_s)), 4)]
                        for p in c[::2]]
                if len(ring) >= 4:
                    ring.append(ring[0])
                    rings.append(ring)
            if not rings:
                continue
            feats.append({
                "type": "Feature",
                "geometry": {"type": "Polygon", "coordinates": [max(rings, key=len)]},
                "properties": {
                    "id": len(feats) + 1,
                    "lon": round(float(lon_s[cx]), 3), "lat": round(float(lat_s[cy]), 3),
                    "max_p": round(float(prob[blob].max()), 3),
                    "area_km2": round(area_km2, 0),
                    "radius_km": round(float(np.sqrt(area_km2 / np.pi)), 1),
                    "cycle": meta["cycle"], "pass": meta["pass"],
                    "date": meta["date"], "time": meta["time"],
                }})
        return feats, ssha, prob

    header = {"model": wpath.name, "threshold": round(thresh, 3), "store": args.store,
              "caveat": "model trained on L3 swath-frame data; "
                        "regridded-L2 results are experimental"}

    if args.all:
        ckpt_path = out_dir / "all_eddies_checkpoint.json"
        ck = json.loads(ckpt_path.read_text()) if ckpt_path.exists() else \
            {"done": [], "features": []}
        done = set(ck["done"])
        todo = [s for s in slices if s["i"] not in done]
        print(f"{len(slices)} slices; {len(done)} done, {len(todo)} left")
        for n, meta in enumerate(todo, 1):
            try:
                feats, _, _ = predict_slice(meta)
            except Exception as e:
                print(f"  ERROR slice {meta['i']} c{meta['cycle']}/p{meta['pass']}: "
                      f"{type(e).__name__}: {e}")
                continue                                  # retry on next run
            ck["features"].extend(feats)
            ck["done"].append(meta["i"])
            if n % 20 == 0 or n == len(todo):
                ckpt_path.write_text(json.dumps(ck))
                print(f"  [{n}/{len(todo)}] c{meta['cycle']}/p{meta['pass']}: "
                      f"{len(feats)} eddies (total {len(ck['features'])})")
        out = out_dir / "all_eddies.geojson"
        out.write_text(json.dumps({"type": "FeatureCollection", "properties": header,
                                   "features": ck["features"]}))
        print(f"{len(ck['features'])} eddies over {len(ck['done'])} slices -> {out}")
        return

    if args.slice is not None:
        meta = slices[args.slice]
    else:
        meta = next(s for s in slices
                    if s["cycle"] == args.cycle and s["pass"] == args.pnum)
    print(f"slice {meta['i']}: cycle {meta['cycle']} pass {meta['pass']} {meta['time']}")
    feats, ssha, prob = predict_slice(meta)
    if ssha is None:
        print("swath barely intersects bbox — nothing to do")
        return
    gj = {"type": "FeatureCollection", "properties": {**header, "slice": meta["i"]},
          "features": feats}
    out = out_dir / f"zarr_c{meta['cycle']:03d}_p{meta['pass']:03d}_eddies.geojson"
    out.write_text(json.dumps(gj))
    print(f"{len(feats)} eddies -> {out}")

    if args.png:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        finite = np.isfinite(ssha)
        fig, axes = plt.subplots(1, 2, figsize=(11, 7))
        v1, v2 = np.nanpercentile(ssha, 2), np.nanpercentile(ssha, 98)
        axes[0].pcolormesh(lon_s, lat_s, ssha, cmap="RdBu_r", vmin=v1, vmax=v2)
        axes[0].set_title(f"zarr SSHA c{meta['cycle']}/p{meta['pass']}")
        axes[1].pcolormesh(lon_s, lat_s, np.where(finite, prob, np.nan),
                           cmap="magma", vmin=0, vmax=1)
        axes[1].set_title(f"P(frontal) > {thresh:.2f}")
        for ax in axes:
            for f in feats:
                ring = np.array(f["geometry"]["coordinates"][0])
                ax.plot(ring[:, 0], ring[:, 1], color="#7b3294", lw=1.2)
            ax.set_aspect(1 / np.cos(np.deg2rad(float(np.mean(lat_s)))))
        fig.savefig(out.with_suffix(".png"), dpi=120, bbox_inches="tight")
        print(f"quicklook -> {out.with_suffix('.png')}")


if __name__ == "__main__":
    main()
