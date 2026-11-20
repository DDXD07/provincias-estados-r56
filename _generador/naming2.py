#!/usr/bin/env python3
"""
Nombra cada provincia-estado del mod "Provincias como Estados (Road to 56)" con la ciudad/municipio
mas poblado que, en el mundo real, cae dentro de esa provincia (datos GeoNames cities500).

Uso:  python naming2.py <HOI4> <Road to 56> <carpeta del mod> <carpeta con p2s_cities500.zip y p2s_admin*.txt>
Variables opcionales: P2S_SPLIT=<split.pkl del generador>  P2S_LOC=<cache de localizacion>
"""
import os, re, sys, csv, time, pickle, collections
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pdx import Node, get, get_all, read
from maptools import analyse
from naming import norm, load_geonames, load_admin, load_vp_loc, hav, poly_feats, Model, unwrap

POS_TYPE = 9          # unitstacks: posicion de referencia de cada provincia
ALPHA = 10.0          # km de "bonus" por cada orden de magnitud de poblacion al emparejar VPs
SAMPLES = 64          # muestras Monte Carlo por ciudad
REGION_SHARE = 0.10   # una region real (pais+admin1) se acepta en un estado si aporta >=10% de su poblacion

def _tps_K(xq, yq, x, y):
    d = np.hypot(xq[:, None] - x[None], yq[:, None] - y[None])
    return np.where(d > 0, d ** 2 * np.log(d + 1e-12), 0)

class InvTPS:
    """(lon,lat) -> pixel: polinomio grado 3 + thin plate spline regularizada sobre los residuos."""
    def __init__(self, u, v, lon, lat, lam=1.0):
        self.B = lambda lo, la: poly_feats(lo / 360.0, la / 90.0, 3)
        B = self.B(lon, lat)
        self.cu = np.linalg.lstsq(B, u, rcond=None)[0]; self.cv = np.linalg.lstsq(B, v, rcond=None)[0]
        self.x = lon * np.cos(np.radians(lat)) / 10; self.y = lat / 10
        m = len(u); K = _tps_K(self.x, self.y, self.x, self.y)
        P = np.stack([np.ones(m), self.x, self.y], 1)
        A = np.zeros((m + 3, m + 3)); A[:m, :m] = K + lam * np.eye(m); A[:m, m:] = P; A[m:, :m] = P.T
        self.su = np.linalg.solve(A, np.concatenate([u - B @ self.cu, np.zeros(3)]))
        self.sv = np.linalg.solve(A, np.concatenate([v - B @ self.cv, np.zeros(3)]))
        self.m = m
    def __call__(self, lon, lat):
        ou = np.zeros(len(lon)); ov = np.zeros(len(lon))
        for s in range(0, len(lon), 4000):
            lo, la = lon[s:s + 4000], lat[s:s + 4000]
            B = self.B(lo, la); x = lo * np.cos(np.radians(la)) / 10; y = la / 10
            K = _tps_K(x, y, self.x, self.y); m = self.m
            ou[s:s + 4000] = B @ self.cu + K @ self.su[:m] + self.su[m] + self.su[m + 1] * x + self.su[m + 2] * y
            ov[s:s + 4000] = B @ self.cv + K @ self.sv[:m] + self.sv[m] + self.sv[m + 1] * x + self.sv[m + 2] * y
        return ou, ov

import unicodedata as _ud
_KEEP_MAP = {'ł': 'l', 'Ł': 'L', '’': "'", '‘': "'", 'ʻ': "'", 'ʼ': "'", '`': "'", 'ı': 'i', 'İ': 'I', 'đ': 'd', 'Đ': 'D', 'ŀ': 'l', 'ħ': 'h', 'ø': 'ø'}
_PREFIXES = ('Arrondissement de ', "Arrondissement d'", 'Municipio de ', 'Municipio ', 'Provincia de ', 'Distrito de ',
             'Partido de ', 'Departamento de ', 'Município de ', 'Comuna de ', 'Canton de ', 'Cantón ')
def clean_name(n):
    for p in _PREFIXES:
        if n.startswith(p) and len(n) > len(p) + 2: n = n[len(p):]
    out = []
    for ch in n:
        ch = _KEEP_MAP.get(ch, ch)
        if ord(ch) < 256: out.append(ch); continue
        d = ''.join(c for c in _ud.normalize('NFKD', ch) if not _ud.combining(c))
        out.append(''.join(c for c in d if ord(c) < 256))
    return re.sub(r'\s+', ' ', ''.join(out)).strip().replace('"', "'")
def good_name(n):
    return 0 < len(n) <= 28 and ',' not in n

def log_factory():
    lines = []
    def L(*a):
        s = ' '.join(map(str, a)); lines.append(s); print(s, flush=True)
    return L, lines

def main(van, r56, mod, gdir):
    t0 = time.time(); L, loglines = log_factory()
    roots = [van, r56]
    here = os.path.dirname(os.path.abspath(__file__))
    M = analyse(os.path.join(r56, 'map'), os.path.join(here, 'mapcache.npz'))
    pid = M['pid']; H, W = pid.shape; land = M['land']; rep = M['rep']
    pos = {}
    for line in open(os.path.join(r56, 'map', 'unitstacks.txt')):
        p = line.split(';')
        if len(p) > 4 and p[1] == str(POS_TYPE): pos[int(p[0])] = (float(p[2]), float(p[4]))
    # ---- estados ----
    sdir_r = os.path.join(r56, 'history', 'states'); sdir_m = os.path.join(mod, 'history', 'states')
    vp = {}; orig_state = {}; orig_name_key = {}
    def collect_vp(st):
        def walk(n):
            for k, _, v in n:
                if k == 'victory_points' and isinstance(v, Node):
                    vals = [x for _, _, x in v]
                    if len(vals) >= 2: vp[int(vals[0])] = max(vp.get(int(vals[0]), 0), float(vals[1]))
                elif isinstance(v, Node): walk(v)
        walk(get(st, 'history', Node()))
    for f in os.listdir(sdir_r):
        if not f.endswith('.txt'): continue
        st = get(read(os.path.join(sdir_r, f)), 'state'); sid = int(get(st, 'id')); collect_vp(st)
        for _, _, x in get(st, 'provinces'): orig_state[int(x)] = sid
    sp = os.environ.get('P2S_SPLIT')
    if sp and os.path.exists(sp):
        D = pickle.load(open(sp, 'rb')); state_of = dict(D['prov_state']); split_ids = set(D['pieces_map'])
    else:
        state_of = dict(orig_state); split_ids = set()
        for f in os.listdir(sdir_m):
            st = get(read(os.path.join(sdir_m, f)), 'state'); sid = int(get(st, 'id'))
            for _, _, x in get(st, 'provinces'): state_of[int(x)] = sid
            if sid in orig_state.values(): split_ids.add(sid)
    unsplit = set(orig_state.values()) - split_ids
    L(f'Estados finales: {len(set(state_of.values()))} | provincias con VP: {len(vp)}')
    # ---- localizacion y GeoNames ----
    lc = pickle.load(open(os.environ['P2S_LOC'], 'rb')) if os.environ.get('P2S_LOC') else {}
    def get_loc(lang):
        if lang not in lc: lc[lang] = load_vp_loc(roots, lang)
        return lc[lang]
    loc_en = get_loc('english')
    gcache = os.path.join(gdir, 'p2s_geonames_cache.pkl')
    if os.path.exists(gcache):
        C, admin, idx = pickle.load(open(gcache, 'rb'))
    else:
        C = load_geonames(gdir); admin = load_admin(gdir); idx = collections.defaultdict(set); nc = {}
        for i, (n, alt) in enumerate(zip(C['name'], C['alt'])):
            for a in set([n] + alt.split(',')):
                if a:
                    k = nc.get(a)
                    if k is None: k = nc[a] = norm(a)
                    idx[k].add(i)
        idx = dict(idx); pickle.dump((C, admin, idx), open(gcache, 'wb'))
    N = len(C['name']); pop = C['pop']; reg = np.array([f'{a}.{b}' for a, b in zip(C['cc'], C['a1'])])
    L(f'GeoNames: {N} lugares poblados [{time.time() - t0:.0f}s]')
    # ---- 1. calibrar la proyeccion con los puntos de victoria ----
    ctrl = []
    named = [q for q in state_of if land[q] and loc_en.get(f'VICTORY_POINTS_{q}')]
    for q in named:
        nm = loc_en.get(f'VICTORY_POINTS_{q}')
        if not nm or q not in pos or not land[q]: continue
        c = sorted(idx.get(norm(nm), ()), key=lambda i: -pop[i])[:25]
        if c: ctrl.append((q, nm, pos[q][0], pos[q][1], c))
    U = np.array([c[2] for c in ctrl]); V = np.array([c[3] for c in ctrl])
    seed = [(j, c[4][0]) for j, c in enumerate(ctrl) if pop[c[4][0]] >= 300000 and (len(c[4]) == 1 or pop[c[4][1]] < pop[c[4][0]] / 5)]
    # semillas por franjas de longitud (el mapa de HOI4 no es una proyeccion unica: America y Eurasia difieren)
    rng = np.random.default_rng(0)
    sj_all = np.array([s_[0] for s_ in seed]); si_all = np.array([s_[1] for s_ in seed])
    plon = np.full(len(ctrl), np.nan); plat = np.full(len(ctrl), np.nan); bestd = np.full(len(ctrl), np.inf)
    for b0 in range(0, W, 700):
        b1 = b0 + 1400
        inb = (U[sj_all] >= b0) & (U[sj_all] < b1)
        sj, si = sj_all[inb], si_all[inb]
        if len(sj) < 8: continue
        best = None
        for it in range(1500):
            pick = rng.choice(len(sj), 5, replace=False)
            try:
                a = np.polyfit(U[sj[pick]], C['lon'][si[pick]], 1); b = np.polyfit(V[sj[pick]], C['lat'][si[pick]], 2)
            except Exception:
                continue
            pl = np.polyval(a, U[sj]); pt = np.polyval(b, V[sj])
            inl = (hav(pt, pl, C['lat'][si], unwrap(C['lon'][si], pl)) < 300).sum()
            if best is None or inl > best[0]: best = (inl, a, b)
        if best is None or best[0] < 6: continue
        # aplicar a los puntos de control de la franja (preferir la franja mas centrada)
        cen = (b0 + b1) / 2
        sel = (U >= b0) & (U < b1) & (np.abs(U - cen) < bestd)
        plon[sel] = np.polyval(best[1], U[sel]); plat[sel] = np.polyval(best[2], V[sel]); bestd[sel] = np.abs(U[sel] - cen)
        L(f'  franja x {b0}-{min(b1, W)}: {len(sj)} semillas, {best[0]} coinciden')
    miss = np.isnan(plon)
    if miss.any():
        good = ~miss
        A = poly_feats(U[good] / W, V[good] / H, 2)
        plon[miss] = poly_feats(U[miss] / W, V[miss] / H, 2) @ np.linalg.lstsq(A, plon[good], rcond=None)[0]
        plat[miss] = poly_feats(U[miss] / W, V[miss] / H, 2) @ np.linalg.lstsq(A, plat[good], rcond=None)[0]
    thr = 400.0
    choice = np.zeros(len(ctrl), int); errs = np.full(len(ctrl), np.inf)
    for rnd in range(7):
        for j, c in enumerate(ctrl):
            d = hav(plat[j], plon[j], C['lat'][c[4]], unwrap(C['lon'][c[4]], plon[j]))
            k = int(np.argmin(d - ALPHA * np.log10(pop[c[4]] + 10))); choice[j] = c[4][k]; errs[j] = d[k]
        ok = errs < thr; lonc = unwrap(C['lon'][choice], plon)
        if False:
            A = poly_feats(U[ok] / W, V[ok] / H, 3); Aa = poly_feats(U / W, V / H, 3)
            plon = Aa @ np.linalg.lstsq(A, lonc[ok], rcond=None)[0]; plat = Aa @ np.linalg.lstsq(A, C['lat'][choice][ok], rcond=None)[0]
        else:
            mdl = Model(U[ok], V[ok], lonc[ok], C['lat'][choice][ok], W, H, deg=5, k=8)
            plon, plat = mdl.fwd(U, V, exclude_self=True)
        thr = max(60.0, 4 * np.median(errs[ok]))
    ok = errs < thr
    model = Model(U[ok], V[ok], unwrap(C['lon'][choice[ok]], plon[ok]), C['lat'][choice[ok]], W, H, deg=5, k=8)
    ctrl_ok = [ctrl[j] for j in np.nonzero(ok)[0]]; ctrl_city = choice[ok]
    pickle.dump(dict(u=model.u, v=model.v, lon=model.lon, lat=model.lat, prov=[c[0] for c in ctrl_ok], city=ctrl_city), open(os.path.join(here, 'ctrl.pkl'), 'wb'))
    flon, flat = model.fwd(model.u, model.v, exclude_self=True)
    cv_km = hav(flat, flon, model.lat, model.lon)
    # refinar: la ciudad de cada punto de control debe caer DENTRO de su provincia (no en el marcador)
    order = M['order']; fs_ = pid.ravel()[order]; starts = np.searchsorted(fs_, np.arange(len(M['area']) + 1))
    cprov_ = np.array([c[0] for c in ctrl_ok])
    def nearest_in_prov(q, u, v):
        pix = order[starts[q]:starts[q + 1]]
        py, px = np.divmod(pix, W); pu_ = px.astype(float); pv_ = (H - 1 - py).astype(float)
        du = (pu_ - u + W / 2) % W - W / 2; d = du ** 2 + (pv_ - v) ** 2
        j = int(np.argmin(d)); return u + du[j], pv_[j], d[j] == 0
    tu, tv = model.u.copy(), model.v.copy()
    for it in range(5):
        pu_, pv_ = InvTPS(tu, tv, model.lon, model.lat)(model.lon, model.lat)
        for j in range(len(tu)):
            a, b, inside = nearest_in_prov(cprov_[j], pu_[j], pv_[j])
            if not inside:
                du = a - pu_[j]; dv = b - pv_[j]; Ln = np.hypot(du, dv) + 1e-9
                a += du / Ln * 1.5; b += dv / Ln * 1.5
            tu[j], tv[j] = a, b
    fold = np.random.default_rng(0).integers(0, 10, len(tu)); iu = np.zeros(len(tu)); iv = np.zeros(len(tu))
    for f_ in range(10):
        tr = fold != f_; te = ~tr
        iu[te], iv[te] = InvTPS(tu[tr], tv[tr], model.lon[tr], model.lat[tr])(model.lon[te], model.lat[te])
    cv_px = np.hypot(iu - tu, iv - tv)
    _col = np.mod(np.round(iu).astype(int), W); _row = np.clip(H - 1 - np.round(iv).astype(int), 0, H - 1)
    hit = (pid[_row, _col] == cprov_).mean()
    model.u, model.v = tu, tv
    inv = InvTPS(tu, tv, model.lon, model.lat)
    L(f'Proyeccion: {ok.sum()} ciudades del juego como puntos de control; error de validacion cruzada mediano {np.median(cv_km):.0f} km '
      f'({np.median(cv_px):.1f} px), p90 {np.percentile(cv_km, 90):.0f} km ({np.percentile(cv_px, 90):.1f} px); ciudades del juego que caen en su propia provincia (validacion cruzada): {hit * 100:.0f}% [{time.time() - t0:.0f}s]')
    # ---- 2. ubicar cada ciudad en el mapa ----
    lon_ref = float(model.fwd(np.array([W / 2.0]), np.array([H / 2.0]))[0][0])
    cu, cv = inv(unwrap(C['lon'], lon_ref), C['lat'])
    # sigma calibrada: la probabilidad media predicha de "caer en su provincia" debe igualar el acierto real
    base_c = model._idw(iu, iv, model.u, model.v, [cv_px], exclude_self=True)[0]
    g0 = np.random.default_rng(2); Z = g0.normal(size=(len(iu), 2, 48))
    def mean_p(scale):
        sg = np.clip(base_c * scale, 1.0, 25.0)[:, None]
        uu = iu[:, None] + Z[:, 0] * sg; vv = iv[:, None] + Z[:, 1] * sg
        col = np.mod(np.round(uu).astype(int), W); row = np.clip(H - 1 - np.round(vv).astype(int), 0, H - 1)
        return (pid[row, col] == cprov_[:, None]).mean()
    def loglik(scale):
        sg = np.clip(base_c * scale, 1.0, 25.0)[:, None]
        uu = iu[:, None] + Z[:, 0] * sg; vv = iv[:, None] + Z[:, 1] * sg
        col = np.mod(np.round(uu).astype(int), W); row = np.clip(H - 1 - np.round(vv).astype(int), 0, H - 1)
        p = (pid[row, col] == cprov_[:, None]).mean(1)
        return np.log(np.maximum(p, 1 / 96)).mean()
    grid = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0]
    lls = [loglik(x) for x in grid]; sig_scale = grid[int(np.argmax(lls))]
    L(f'Incertidumbre calibrada (max. verosimilitud): escala {sig_scale}, prob. media {mean_p(sig_scale):.2f} vs acierto real {hit:.2f}')
    sig = np.clip(model._idw(cu, cv, model.u, model.v, [cv_px])[0] * sig_scale, 1.0, 25.0)
    prov_ok = np.zeros(len(land), bool)
    for q in state_of:
        if land[q]: prov_ok[q] = True
    def lookup(u, v):
        col = np.mod(np.round(u).astype(np.int64), W); row = H - 1 - np.round(v).astype(np.int64)
        good = (row >= 0) & (row < H)
        out = np.zeros(u.shape, np.int64)
        out[good] = pid[row[good], col[good]]
        out[~prov_ok[out]] = 0
        return out
    pt = lookup(cu, cv)
    # muestras Monte Carlo: probabilidad de que cada ciudad caiga en cada provincia
    g = np.random.default_rng(1)
    mapped = np.nonzero((cv > -30) & (cv < H + 30))[0]
    probs = {}  # ciudad -> {prov: p}
    S = np.zeros((len(mapped), SAMPLES), np.int64)
    for s in range(0, len(mapped), 20000):
        ii = mapped[s:s + 20000]
        du = g.normal(size=(len(ii), SAMPLES)) * sig[ii, None]; dv = g.normal(size=(len(ii), SAMPLES)) * sig[ii, None]
        S[s:s + len(ii)] = lookup(cu[ii, None] + du, cv[ii, None] + dv)
    for r, i in enumerate(mapped):
        row = S[r]; row = row[row > 0]
        if len(row) == 0: continue
        vals, cnt = np.unique(row, return_counts=True)
        probs[int(i)] = {int(a): b / len(row) for a, b in zip(vals, cnt)}
    L(f'Ciudades ubicadas en provincias: {len(probs)} [{time.time() - t0:.0f}s]')
    # ---- 3. regiones reales permitidas por estado original ----
    st_reg = collections.defaultdict(collections.Counter)
    for i, pr in probs.items():
        for q, p in pr.items(): st_reg[orig_state[q]][reg[i]] += p * max(pop[i], 500)
    allowed = {}
    for sid, cnt in st_reg.items():
        tot = sum(cnt.values()) or 1
        allowed[sid] = {r for r, v in cnt.items() if v / tot >= REGION_SHARE}
    for (q, nm, u, v, c), ci in zip(ctrl_ok, ctrl_city):
        allowed.setdefault(orig_state[q], set()).add(reg[ci])     # la region real de cada VP del estado
    admin_name_by_norm = collections.defaultdict(set)
    for code, nm in admin.items():
        if code.count('.') == 1: admin_name_by_norm[norm(nm)].add(code)
    for sid in set(orig_state.values()):
        sn = loc_en.get(f'STATE_{sid}')
        if sn: allowed.setdefault(sid, set()).update(admin_name_by_norm.get(norm(sn), ()))
    # ---- 4. asignacion: nombres de VP primero, luego ciudades por puntuacion ----
    names = {}; info = {}; used_city = set(); taken = set(); used_names = collections.defaultdict(set)
    vp_city = {q: int(ci) for (q, nm, u, v, c), ci in zip(ctrl_ok, ctrl_city)}
    ctrl_ok_prov = {c[0] for c in ctrl_ok}
    # nombres del juego: solo si la provincia tiene VP o si el nombre coincide con un lugar real cercano
    for q in sorted(named, key=lambda q: (-vp.get(q, 0), q not in ctrl_ok_prov, q)):
        if q in vp or q in ctrl_ok_prov:
            nmq = loc_en[f'VICTORY_POINTS_{q}']
            if norm(nmq) in used_names[orig_state[q]]: continue
            names[q] = ('juego', nmq); taken.add(q); used_names[orig_state[q]].add(norm(nmq))
            ci = vp_city.get(q)
            if ci is not None: used_city.add(ci)
            info[q] = (ci, probs.get(ci, {}).get(q, 0.0)) if ci is not None else (None, 0.0)
    allowed_cc = {sid_: {r_.split('.')[0] for r_ in rs} for sid_, rs in allowed.items()}
    truth_cc = collections.defaultdict(set)
    for (q_, nm_, u_, v_, c_), ci_ in zip(ctrl_ok, ctrl_city): truth_cc[orig_state[q_]].add(C['cc'][ci_])
    def cc_ok(i, osid, pstate):
        t_ = truth_cc.get(osid)
        return (not t_) or C['cc'][i] in t_ or pstate >= 0.9
    reg_states = collections.defaultdict(set)
    for sid_, rs in allowed.items():
        for r_ in rs: reg_states[r_].add(sid_)
    # provincias por estado original (para recuperar ciudades que se salieron por el borde)
    land_q = np.array([q for q in state_of if land[q]])
    qu = np.array([pos.get(int(q), (0, 0))[0] for q in land_q]); qv = np.array([pos.get(int(q), (0, 0))[1] for q in land_q])
    q_os = np.array([orig_state[int(q)] for q in land_q])
    pairs = []; recovered = 0
    for i, pr in probs.items():
        if i in used_city: continue
        okq = {}
        for q, p in pr.items():
            osid = orig_state[q]
            pst = sum(pp for qq, pp in pr.items() if orig_state[qq] == osid)
            if (reg[i] in allowed.get(osid, ()) or (C['cc'][i] in allowed_cc.get(osid, ()) and pst >= 0.95)) and cc_ok(i, osid, pst):
                okq[q] = p
        if not okq and pop[i] >= 5000:
            sts = reg_states.get(reg[i], set())
            if sts:
                du = (qu - cu[i] + W / 2) % W - W / 2; d = np.hypot(du, qv - cv[i])
                cand = np.nonzero(np.isin(q_os, list(sts)) & (d < 3 * sig[i] + 6))[0]
                if len(cand):
                    j = cand[np.argmin(d[cand])]; okq[int(land_q[j])] = 0.2; recovered += 1
        tot = sum(okq.values())
        for q, p in okq.items():
            if q in taken: continue
            pn = p / tot if tot > 0 else p
            pairs.append((max(pop[i], 1) * pn, i, q, pn))
    pairs.sort(key=lambda x: -x[0])
    for sc, i, q, p in pairs:
        if q in taken or i in used_city: continue
        nmc = clean_name(C['name'][i])
        if not good_name(nmc) or norm(nmc) in used_names[orig_state[q]]: continue
        names[q] = ('geonames', nmc); info[q] = (i, p); taken.add(q); used_city.add(i); used_names[orig_state[q]].add(norm(nmc))
    L(f'Provincias nombradas por ciudad interior: {sum(1 for v in names.values() if v[0] == "geonames")} (ciudades recuperadas del borde: {recovered}) [{time.time() - t0:.0f}s]')
    # ---- 5. provincias sin ciudad dentro: ciudad cercana de la misma region, o municipio/condado ----
    free = np.array(sorted(set(probs) - used_city)); fu, fv = cu[free], cv[free]
    nfall = collections.Counter()
    for q in sorted(state_of):
        if not land[q] or q in taken: continue
        osid = orig_state[q]
        u0, v0 = pos.get(q, (None, None))
        if u0 is None:
            ry, rx = divmod(int(rep[q]), W); u0, v0 = rx, H - 1 - ry
        du = (fu - u0 + W / 2) % W - W / 2; d = np.hypot(du, fv - v0)
        near = np.argsort(d)[:200]
        done = False
        for j in near:
            i = int(free[j])
            if d[j] > 45: break
            nmc = clean_name(C['name'][i])
            if i in used_city or reg[i] not in allowed.get(osid, ()) or not cc_ok(i, osid, 0) or not good_name(nmc) or norm(nmc) in used_names[osid]: continue
            names[q] = ('cercana', nmc); info[q] = (i, 0.0); used_city.add(i); taken.add(q); used_names[osid].add(norm(nmc))
            nfall['ciudad_cercana'] += 1; done = True; break
        if not done:
            for j in near:
                i = int(free[j])
                if d[j] > 120: break
                if reg[i] not in allowed.get(osid, ()) or not cc_ok(i, osid, 0): continue
                a2 = admin.get(f'{C["cc"][i]}.{C["a1"][i]}.{C["a2"][i]}')
                a2 = clean_name(a2) if a2 else None
                if a2 and good_name(a2) and norm(a2) not in used_names[osid]:
                    names[q] = ('condado', a2); info[q] = (i, 0.0); taken.add(q); used_names[osid].add(norm(a2))
                    nfall['municipio_condado'] += 1; done = True; break
        if not done:
            for j in np.argsort(d)[:2000]:
                i = int(free[j])
                if d[j] > 200: break
                nmc = clean_name(C['name'][i])
                if i in used_city or reg[i] not in allowed.get(osid, ()) or not cc_ok(i, osid, 0) or not good_name(nmc) or norm(nmc) in used_names[osid]: continue
                names[q] = ('lejana', nmc); info[q] = (i, 0.0); used_city.add(i); taken.add(q); used_names[osid].add(norm(nmc))
                nfall['ciudad_lejana'] += 1; done = True; break
        if not done: nfall['sin_datos'] += 1
    L('Provincias sin ciudad interior: ' + str(dict(nfall)))
    # ---- 6. verificacion ----
    report = []; conf = []
    reg_name = lambda r: admin.get(r, r)
    for q in sorted(state_of):
        if not land[q]: continue
        sid = state_of[q]; osid = orig_state[q]
        src, nm = names.get(q, ('sin_datos', ''))
        ci, p = info.get(q, (None, 0.0))
        checks = []
        if ci is not None and isinstance(ci, int):
            if reg[ci] not in allowed.get(osid, ()) : checks.append('fuera_de_la_region_principal')
            if src == 'geonames': conf.append(p)
        report.append(dict(estado=sid, estado_original=osid, nombre_estado_original=loc_en.get(f'STATE_{osid}', ''),
                           provincia=q, nombre=nm, fuente=src,
                           ciudad=C['name'][ci] if ci is not None else '', poblacion=int(pop[ci]) if ci is not None else '',
                           region=reg_name(reg[ci]) if ci is not None else '', pais=C['cc'][ci] if ci is not None else '',
                           lat=round(float(C['lat'][ci]), 4) if ci is not None else '', lon=round(float(C['lon'][ci]), 4) if ci is not None else '',
                           confianza=round(p, 2), avisos=' '.join(checks)))
    conf = np.array(conf)
    L(f'Confianza (prob. de que la ciudad este dentro de la provincia): mediana {np.median(conf):.2f}, '
      f'>=0.8: {(conf >= 0.8).mean() * 100:.0f}%, <0.5: {(conf < 0.5).mean() * 100:.0f}%')
    srcs = collections.Counter(r['fuente'] for r in report); L('Fuentes: ' + str(dict(srcs)))
    # nombres repetidos dentro del mismo estado original
    rep_cnt = collections.Counter((r['estado_original'], r['nombre']) for r in report if r['nombre'])
    L(f'Nombres repetidos dentro de un mismo estado original: {sum(c - 1 for c in rep_cnt.values() if c > 1)}')
    pickle.dump(dict(names=names, report=report, split_ids=split_ids, unsplit=unsplit, state_of=state_of),
                open(os.path.join(here, 'naming_result.pkl'), 'wb'))
    with open(os.path.join(mod, '_generador', 'nombres_provincias.csv'), 'w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(report[0].keys())); w.writeheader(); w.writerows(report)
    with open(os.path.join(mod, '_generador', 'nombres_log.txt'), 'w', encoding='utf-8') as f: f.write('\n'.join(loglines) + '\n')
    L(f'Listo [{time.time() - t0:.0f}s]')

if __name__ == '__main__':
    main(*sys.argv[1:5])
