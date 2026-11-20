import os, numpy as np
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
def load_definition(path):
    d = {}
    with open(path, encoding='utf-8-sig', errors='replace') as f:
        for line in f:
            p = line.strip().split(';')
            if len(p) < 5 or not p[0].isdigit(): continue
            d[int(p[0])] = (int(p[1]), int(p[2]), int(p[3]), p[4])
    return d
def analyse(mapdir, cache):
    if os.path.exists(cache):
        z = np.load(cache, allow_pickle=True)
        return {k: z[k] for k in z.files}
    defs = load_definition(os.path.join(mapdir, 'definition.csv'))
    img = np.asarray(Image.open(os.path.join(mapdir, 'provinces.bmp')).convert('RGB'))
    H, W, _ = img.shape
    key = (img[..., 0].astype(np.int32) << 16) | (img[..., 1].astype(np.int32) << 8) | img[..., 2].astype(np.int32)
    ids = np.array(sorted(defs), dtype=np.int32)
    cols = np.array([(defs[i][0] << 16) | (defs[i][1] << 8) | defs[i][2] for i in ids], dtype=np.int32)
    order = np.argsort(cols); cs = cols[order]; ci = ids[order]
    pos = np.clip(np.searchsorted(cs, key), 0, len(cs) - 1)
    ok = cs[pos] == key
    pid = np.where(ok, ci[pos], 0).astype(np.int32)
    del key, img
    n = int(ids.max()) + 1
    flat = pid.ravel()
    area = np.bincount(flat, minlength=n)
    rows, cols_ = np.indices((H, W))
    sx = np.bincount(flat, weights=cols_.ravel(), minlength=n)
    sz = np.bincount(flat, weights=rows.ravel(), minlength=n)
    with np.errstate(invalid='ignore', divide='ignore'):
        cx = sx / area; cy = sz / area
    # nearest pixel to centroid per province
    dist = (cols_.ravel() - cx[flat]) ** 2 + (rows.ravel() - cy[flat]) ** 2
    o = np.lexsort((dist, flat))
    fs = flat[o]
    first = np.r_[True, fs[1:] != fs[:-1]]
    rep = np.full(n, -1, dtype=np.int64)
    rep[fs[first]] = o[first]
    # adjacency (with horizontal wrap)
    a1 = pid; a2 = np.roll(pid, -1, axis=1)
    m = a1 != a2
    e1 = np.stack([a1[m], a2[m]], 1)
    a2 = np.roll(pid, -1, axis=0)[:-1]; a1 = pid[:-1]
    m = a1 != a2
    e2 = np.stack([a1[m], a2[m]], 1)
    e = np.concatenate([e1, e2]); e = np.sort(e, 1)
    e = np.unique(e[:, 0].astype(np.int64) * 100000 + e[:, 1])
    edges = np.stack([e // 100000, e % 100000], 1).astype(np.int32)
    sea = np.zeros(n, bool); land = np.zeros(n, bool)
    for i, (r, g, b, t) in defs.items():
        sea[i] = t == 'sea'; land[i] = t == 'land'
    coastal = np.zeros(n, bool)
    for a, b in edges:
        if sea[a] and land[b]: coastal[b] = True
        if sea[b] and land[a]: coastal[a] = True
    hm = np.asarray(Image.open(os.path.join(mapdir, 'heightmap.bmp')).convert('L'))
    out = dict(pid=pid, area=area, rep=rep, edges=edges, coastal=coastal, sea=sea, land=land, hm=hm,
               order=o.astype(np.int32), H=np.array(H), W=np.array(W))
    np.savez(cache, **out)
    return out
if __name__ == '__main__':
    import time, sys
    t = time.time()
    M = analyse(os.path.expanduser('~/mnt/820260968/map'), os.path.expanduser('~/p2s/mapcache.npz'))
    print('done', time.time() - t, M['pid'].shape, 'land', M['land'].sum(), 'coastal', M['coastal'].sum(), 'unmapped px', (M['pid'] == 0).sum())
