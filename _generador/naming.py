#!/usr/bin/env python3
"""
Renombra cada provincia-estado del mod con la ciudad/municipio mas poblado que cae dentro
de la provincia en el mundo real (GeoNames cities500).
Uso: python naming.py <HOI4> <Road to 56> <carpeta del mod> <carpeta con cities500.zip/txt>
"""
import os, re, sys, zipfile, io, unicodedata, collections, math, csv, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pdx import Node, get, get_all, read
from maptools import analyse

BAD_FCODES = {'PPLX', 'PPLH', 'PPLQ', 'PPLW', 'PPLCH'}
R_EARTH = 6371.0

def norm(s):
    s = unicodedata.normalize('NFKD', s.lower())
    s = ''.join(c for c in s if not unicodedata.combining(c))
    s = re.sub(r'[^a-z0-9]+', ' ', s).strip()
    return s

def load_geonames(gdir):
    path = None
    for f in os.listdir(gdir):
        if re.match(r'(p2s_)?cities500\.(zip|txt)$', f): path = os.path.join(gdir, f)
    if path is None: raise SystemExit('No encuentro cities500.zip en ' + gdir)
    if path.endswith('.zip'):
        z = zipfile.ZipFile(path); raw = z.read('cities500.txt').decode('utf-8')
    else:
        raw = open(path, encoding='utf-8').read()
    C = dict(name=[], alt=[], lat=[], lon=[], fcode=[], cc=[], a1=[], a2=[], pop=[])
    for line in raw.splitlines():
        p = line.split('\t')
        if len(p) < 15 or p[6] != 'P' or p[7] in BAD_FCODES: continue
        C['name'].append(p[1]); C['alt'].append(p[2] + ',' + p[3])
        C['lat'].append(float(p[4])); C['lon'].append(float(p[5])); C['fcode'].append(p[7])
        C['cc'].append(p[8]); C['a1'].append(p[10]); C['a2'].append(p[11])
        C['pop'].append(int(p[14] or 0))
    for k in ('lat', 'lon', 'pop'): C[k] = np.array(C[k])
    return C

def load_admin(gdir):
    names = {}
    for fn in ('admin1CodesASCII.txt', 'admin2Codes.txt'):
        for cand in (fn, 'p2s_' + fn):
            p = os.path.join(gdir, cand)
            if os.path.exists(p):
                for line in open(p, encoding='utf-8'):
                    q = line.rstrip('\n').split('\t')
                    if len(q) >= 2: names[q[0]] = q[1]
    return names

LOC_RE = re.compile(r'^\s*([A-Za-z0-9_.\-]+):\d*\s*"(.*)"\s*(#.*)?$')
def load_vp_loc(roots, lang):
    out = {}
    for r in roots:
        d = os.path.join(r, 'localisation', lang)
        if not os.path.isdir(d): continue
        for dp, dn, fn in os.walk(d):
            for f in sorted(fn):
                if not f.endswith('.yml'): continue
                for line in open(os.path.join(dp, f), encoding='utf-8-sig', errors='replace'):
                    if 'VICTORY_POINTS_' in line or 'STATE_' in line:
                        m = LOC_RE.match(line)
                        if m: out[m.group(1)] = m.group(2)
    return out

def hav(lat1, lon1, lat2, lon2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dl = np.radians(lon2 - lon1); dp = p2 - p1
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * R_EARTH * np.arcsin(np.sqrt(np.clip(a, 0, 1)))

def poly_feats(a, b, deg):
    cols = []
    for i in range(deg + 1):
        for j in range(deg + 1 - i):
            cols.append((a ** i) * (b ** j))
    return np.stack(cols, 1)

class Model:
    """Mapeo pixel(u,v) <-> (lon,lat): polinomio + correccion local IDW con puntos de control."""
    def __init__(self, u, v, lon, lat, W, H, deg=5, k=8):
        self.W, self.H, self.deg, self.k = W, H, deg, k
        self.u, self.v, self.lon, self.lat = u, v, lon, lat
        A = poly_feats(u / W, v / H, deg)
        self.cf_lon = np.linalg.lstsq(A, lon, rcond=None)[0]
        self.cf_lat = np.linalg.lstsq(A, lat, rcond=None)[0]
        self.r_lon = lon - A @ self.cf_lon; self.r_lat = lat - A @ self.cf_lat
        B = poly_feats(lon / 360.0, lat / 90.0, deg)
        self.cf_u = np.linalg.lstsq(B, u, rcond=None)[0]
        self.cf_v = np.linalg.lstsq(B, v, rcond=None)[0]
        self.r_u = u - B @ self.cf_u; self.r_v = v - B @ self.cf_v
    def _idw(self, qa, qb, pa, pb, vals, exclude_self=False, scale_b=1.0):
        out = [np.zeros(len(qa)) for _ in vals]
        for s in range(0, len(qa), 2000):
            da = qa[s:s + 2000, None] - pa[None, :]; db = (qb[s:s + 2000, None] - pb[None, :]) * scale_b
            d2 = da ** 2 + db ** 2
            if exclude_self: d2[d2 < 1e-9] = np.inf
            idx = np.argpartition(d2, self.k, axis=1)[:, :self.k]
            dd = np.take_along_axis(d2, idx, 1)
            w = 1.0 / (dd + 1e-6)
            w /= w.sum(1, keepdims=True)
            for o, vv in zip(out, vals): o[s:s + 2000] = (w * vv[idx]).sum(1)
        return out
    def fwd(self, u, v, exclude_self=False):
        A = poly_feats(u / self.W, v / self.H, self.deg)
        cl, ct = self._idw(u, v, self.u, self.v, [self.r_lon, self.r_lat], exclude_self)
        return A @ self.cf_lon + cl, A @ self.cf_lat + ct
    def inv(self, lon, lat, exclude_self=False):
        B = poly_feats(lon / 360.0, lat / 90.0, self.deg)
        cu, cv = self._idw(lon, lat, self.lon, self.lat, [self.r_u, self.r_v], exclude_self)
        return B @ self.cf_u + cu, B @ self.cf_v + cv

def unwrap(lon, ref):
    return lon + 360.0 * np.round((ref - lon) / 360.0)

def main(van, r56, mod, gdir):
    t0 = time.time()
    roots = [van, r56]
    log = []
    def L(*a):
        s = ' '.join(map(str, a)); log.append(s); print(s, flush=True)
    M = analyse(os.path.join(r56, 'map'), os.path.join(os.path.dirname(os.path.abspath(__file__)), 'mapcache.npz'))
    pid = M['pid']; H, W = pid.shape; land = M['land']
    # posicion "ciudad" de cada provincia (unitstacks tipo 38) y representante
    pos = {}
    for line in open(os.path.join(r56, 'map', 'unitstacks.txt')):
        p = line.split(';')
        if len(p) > 4 and p[1] == '38': pos[int(p[0])] = (float(p[2]), float(p[4]))
    # estados del mod (resultado final): provincia -> estado, vp
    sdir_mod = os.path.join(mod, 'history', 'states'); sdir_r = os.path.join(r56, 'history', 'states')
    state_of = {}; vp = {}; main_ids = set(); unsplit = set()
    def collect_vp(st):
        def walk(n):
            for k, _, v in n:
                if k == 'victory_points' and isinstance(v, Node):
                    vals = [x for _, _, x in v]
                    if len(vals) >= 2: vp[int(vals[0])] = max(vp.get(int(vals[0]), 0), float(vals[1]))
                elif isinstance(v, Node): walk(v)
        walk(get(st, 'history', Node()))
    split_pkl = os.environ.get('P2S_SPLIT')
    if split_pkl and os.path.exists(split_pkl):
        import pickle
        D = pickle.load(open(split_pkl, 'rb'))
        state_of = dict(D['prov_state']); main_ids = set(D['pieces_map'])
        for f in os.listdir(sdir_r):
            st = get(read(os.path.join(sdir_r, f)), 'state'); collect_vp(st)
            sid = int(get(st, 'id'))
            if sid not in main_ids: unsplit.add(sid)
    else:
        files = {f: os.path.join(sdir_r, f) for f in os.listdir(sdir_r)}
        orig_files = set(files); split_mod_files = set(os.listdir(sdir_mod))
        files.update({f: os.path.join(sdir_mod, f) for f in split_mod_files})
        for f, p in files.items():
            st = get(read(p), 'state'); sid = int(get(st, 'id'))
            if f in orig_files and f not in split_mod_files: unsplit.add(sid)
            elif f in orig_files: main_ids.add(sid)
            for _, _, x in get(st, 'provinces'): state_of[int(x)] = sid
            collect_vp(st)
    L(f'Estados: {len(set(state_of.values()))}, provincias con VP: {len(vp)}')
    import pickle as _pk
    _lc = _pk.load(open(os.environ['P2S_LOC'], 'rb')) if os.environ.get('P2S_LOC') else {}
    def get_loc(lang):
        if lang not in _lc: _lc[lang] = load_vp_loc(roots, lang)
        return _lc[lang]
    loc_en = get_loc('english')
    import pickle
    gcache = os.path.join(gdir, 'p2s_geonames_cache.pkl')
    if os.path.exists(gcache):
        C, admin, idx = pickle.load(open(gcache, 'rb'))
    else:
        C = load_geonames(gdir); admin = load_admin(gdir)
        idx = collections.defaultdict(set)
        nc = {}
        for i, (n, alt) in enumerate(zip(C['name'], C['alt'])):
            for a in set([n] + alt.split(',')):
                if not a: continue
                k = nc.get(a)
                if k is None: k = nc[a] = norm(a)
                idx[k].add(i)
        idx = dict(idx)
        pickle.dump((C, admin, idx), open(gcache, 'wb'))
    L(f'GeoNames: {len(C["name"])} lugares poblados [{time.time() - t0:.0f}s]')
    # puntos de control: provincias con VP y nombre
    ctrl = []
    for q, val in vp.items():
        nm = loc_en.get(f'VICTORY_POINTS_{q}')
        if not nm or q not in pos: continue
        cands = sorted(idx.get(norm(nm), ()), key=lambda i: -C['pop'][i])[:20]
        if cands: ctrl.append((q, nm, pos[q][0], pos[q][1], cands))
    L(f'Puntos de control con coincidencia de nombre: {len(ctrl)} [{time.time() - t0:.0f}s]')
    U = np.array([c[2] for c in ctrl]); V = np.array([c[3] for c in ctrl])
    # semilla: ciudades grandes con nombre casi unico
    seed = []
    for j, c in enumerate(ctrl):
        pops = [C['pop'][i] for i in c[4]]
        if pops[0] >= 300000 and (len(pops) == 1 or pops[1] < pops[0] / 5): seed.append((j, c[4][0]))
    # modelo inicial simple: lon lineal en u, lat polinomio en v (RANSAC)
    rng = np.random.default_rng(0)
    sj = np.array([s[0] for s in seed]); si = np.array([s[1] for s in seed])
    best = None
    for it in range(3000):
        pick = rng.choice(len(seed), 6, replace=False)
        a = np.polyfit(U[sj[pick]], C['lon'][si[pick]], 1)
        b = np.polyfit(V[sj[pick]], C['lat'][si[pick]], 3)
        plon = np.polyval(a, U[sj]); plat = np.polyval(b, V[sj])
        err = hav(plat, plon, C['lat'][si], unwrap(C['lon'][si], plon))
        inl = (err < 400).sum()
        if best is None or inl > best[0]: best = (inl, a, b)
    L(f'Semilla: {len(seed)} ciudades, inliers RANSAC {best[0]}')
    plon = np.polyval(best[1], U); plat = np.polyval(best[2], V)
    choice = np.full(len(ctrl), -1); thr = 400.0
    for rnd in range(6):
        errs = np.full(len(ctrl), np.inf)
        for j, c in enumerate(ctrl):
            cl = C['lat'][c[4]]; co = unwrap(C['lon'][c[4]], plon[j])
            d = hav(plat[j], plon[j], cl, co)
            # preferir el candidato cercano; desempatar por poblacion
            score = d - 0.0 * np.log10(C['pop'][c[4]] + 10)
            k = int(np.argmin(score)); choice[j] = c[4][k]; errs[j] = d[k]
        ok = errs < thr
        lonc = unwrap(C['lon'][choice], plon)
        deg = 3 if rnd < 2 else 5
        A = poly_feats(U[ok] / W, V[ok] / H, deg)
        cfl = np.linalg.lstsq(A, lonc[ok], rcond=None)[0]; cft = np.linalg.lstsq(A, C['lat'][choice][ok], rcond=None)[0]
        Aall = poly_feats(U / W, V / H, deg)
        plon = Aall @ cfl; plat = Aall @ cft
        med = np.median(errs[ok]); thr = max(60.0, 4 * med)
        L(f'  ronda {rnd}: aceptados {ok.sum()}, error mediano {med:.1f} km, umbral {thr:.0f} km')
    ok = errs < thr
    cu, cv = U[ok], V[ok]
    clon = unwrap(C['lon'][choice[ok]], plon[ok]); clat = C['lat'][choice[ok]]
    model = Model(cu, cv, clon, clat, W, H, deg=5, k=8)
    # validacion cruzada (dejando fuera el propio punto)
    flon, flat = model.fwd(cu, cv, exclude_self=True)
    cv_km = hav(flat, flon, clat, clon)
    iu, iv = model.inv(clon, clat, exclude_self=True)
    cv_px = np.hypot(iu - cu, iv - cv)
    L(f'Validacion cruzada: error mediano {np.median(cv_km):.1f} km (p90 {np.percentile(cv_km, 90):.1f}), '
      f'en pixeles mediano {np.median(cv_px):.1f} (p90 {np.percentile(cv_px, 90):.1f})')
    lon_ref = model.fwd(np.array([W / 2.0]), np.array([H / 2.0]))[0][0]
    # ciudades -> pixel -> provincia
    clon_all = unwrap(C['lon'], lon_ref)
    cu_all, cv_all = model.inv(clon_all, C['lat'])
    col = np.mod(np.round(cu_all).astype(np.int64), W); row = (H - 1 - np.round(cv_all)).astype(np.int64)
    inside = (row >= 0) & (row < H)
    row = np.clip(row, 0, H - 1)
    cprov = pid[row, col].astype(np.int64)
    cprov[~inside] = 0
    # ciudades costeras que cayeron al mar: buscar tierra a <=4 px
    wet = ~land[cprov] & inside
    for i in np.nonzero(wet)[0]:
        r0, c0 = row[i], col[i]; best_p = 0; bd = 99
        for dr in range(-4, 5):
            for dc in range(-4, 5):
                rr, cc = r0 + dr, (c0 + dc) % W
                if 0 <= rr < H and land[pid[rr, cc]] and dr * dr + dc * dc < bd:
                    bd = dr * dr + dc * dc; best_p = pid[rr, cc]
        cprov[i] = best_p
    # error local estimado (px) para cada ciudad: IDW de los errores de validacion cruzada
    loc_err = model._idw(cu_all, cv_all, cu, cv, [cv_px])[0]
    by_prov = collections.defaultdict(list)
    for i in np.argsort(-C['pop']):
        if cprov[i] and cprov[i] in state_of: by_prov[int(cprov[i])].append(int(i))
    # paises de los puntos de control cercanos (para verificar)
    ctrl_cc = np.array([C['cc'][i] for i in choice[ok]])
    def near_cc(u, v, k=4):
        d = (cu - u) ** 2 + (cv - v) ** 2
        return set(ctrl_cc[np.argsort(d)[:k]])
    def border_frac(i, r):
        r = max(2, int(round(r)))
        p0 = cprov[i]; s = 0; n = 0
        for a in np.linspace(0, 2 * np.pi, 16, endpoint=False):
            rr = int(row[i] + r * np.sin(a)); cc = int(col[i] + r * np.cos(a)) % W
            if 0 <= rr < H: n += 1; s += pid[rr, cc] == p0
        return s / max(n, 1)
    # elegir nombres
    names = {}; report = []
    used_by_state = collections.defaultdict(set)
    rep = M['rep']
    prov_list = sorted(state_of)
    for q in prov_list:
        if not land[q]: continue
        sid = state_of[q]
        cands = by_prov.get(q, [])
        u0, v0 = pos.get(q, (None, None))
        if u0 is None:
            ry, rx = divmod(int(rep[q]), W); u0, v0 = rx, H - 1 - ry
        flag = ''
        vpname = loc_en.get(f'VICTORY_POINTS_{q}') if q in vp else None
        if vpname:
            src = 'vp'; nm = vpname; ci = cands[0] if cands else None
        elif cands:
            ci = cands[0]; nm = C['name'][ci]; src = 'geonames'
            conf = border_frac(ci, loc_err[ci] + 1)
            if conf < 0.5:
                # ciudad pegada al borde: si hay otra con >=1/3 de poblacion y bien adentro, preferirla
                for cj in cands[1:6]:
                    if C['pop'][cj] * 3 >= C['pop'][ci] and border_frac(cj, loc_err[cj] + 1) >= 0.75:
                        ci = cj; nm = C['name'][cj]; flag = 'cambiada_por_borde'; break
                else:
                    flag = 'cerca_del_borde'
            if C['cc'][ci] not in near_cc(u0, v0):
                flag = (flag + ' ' if flag else '') + 'pais_distinto_a_vecinos'
        else:
            # sin ciudad dentro: la mas cercana no usada en el mismo estado original, a <= 20 px
            lonq, latq = model.fwd(np.array([float(u0)]), np.array([float(v0)]))
            d = hav(latq[0], lonq[0], C['lat'], unwrap(C['lon'], lonq[0]))
            ci = None; nm = None; src = 'cercana'
            for cj in np.argsort(d)[:30]:
                if C['name'][cj] not in used_by_state[sid] and d[cj] < 250:
                    ci = int(cj); nm = C['name'][cj]; flag = f'sin_ciudad_dentro ({d[cj]:.0f} km)'; break
            if nm is None:
                src = 'sin_datos'; flag = 'sin_ciudad'
        if nm:
            nm = nm.replace('"', "'")
            if nm in used_by_state[sid] and src != 'vp':
                a2 = admin.get(f'{C["cc"][ci]}.{C["a1"][ci]}.{C["a2"][ci]}') if ci is not None else None
                nm2 = a2 if a2 and a2 not in used_by_state[sid] else None
                if nm2: nm = nm2; flag = (flag + ' ' if flag else '') + 'duplicado->condado'
                else: flag = (flag + ' ' if flag else '') + 'nombre_repetido'
            used_by_state[sid].add(nm)
            names[q] = (sid, nm, src)
        report.append(dict(estado=sid, provincia=q, nombre=nm or '', fuente=src,
                           pais=C['cc'][ci] if ci is not None else '', poblacion=int(C['pop'][ci]) if ci is not None else '',
                           lat=round(float(C['lat'][ci]), 4) if ci is not None else '', lon=round(float(C['lon'][ci]), 4) if ci is not None else '',
                           ciudades_dentro=len(cands), aviso=flag))
    cnt = collections.Counter(r['fuente'] for r in report); fl = collections.Counter()
    for r in report:
        for f in r['aviso'].split():
            if not f.startswith('('): fl[f] += 1
    L('Fuentes de nombres: ' + str(dict(cnt))); L('Avisos: ' + str(dict(fl)))
    # escribir localizacion
    langs = sorted(d for d in os.listdir(os.path.join(van, 'localisation')) if os.path.isdir(os.path.join(van, 'localisation', d)))
    sid_name = {}
    for q, (sid, nm, src) in names.items():
        sid_name.setdefault(sid, []).append((q, nm, src))
    for lang in langs:
        loc_l = {**loc_en, **get_loc(lang)} if lang != 'english' else loc_en
        d = os.path.join(mod, 'localisation', lang); os.makedirs(os.path.join(d, 'replace'), exist_ok=True)
        new_lines, rep_lines = [], []
        for sid, lst in sorted(sid_name.items()):
            if sid in unsplit: continue
            # una provincia por estado (los estados divididos tienen 1 tierra)
            q, nm, src = lst[0]
            if src == 'vp': nm = loc_l.get(f'VICTORY_POINTS_{q}', nm).replace('"', "'")
            line = f' STATE_{sid}:0 "{nm}"\n'
            (rep_lines if sid in main_ids else new_lines).append(line)
        with open(os.path.join(d, f'p2s_state_names_l_{lang}.yml'), 'w', encoding='utf-8-sig', newline='\n') as f:
            f.write(f'l_{lang}:\n' + ''.join(new_lines))
        with open(os.path.join(d, 'replace', f'p2s_state_names_replace_l_{lang}.yml'), 'w', encoding='utf-8-sig', newline='\n') as f:
            f.write(f'l_{lang}:\n' + ''.join(rep_lines))
    with open(os.path.join(mod, '_generador', 'nombres_provincias.csv'), 'w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(report[0].keys())); w.writeheader(); w.writerows(report)
    with open(os.path.join(mod, '_generador', 'nombres_log.txt'), 'w', encoding='utf-8') as f: f.write('\n'.join(log) + '\n')
    L(f'Listo en {time.time() - t0:.0f}s')

if __name__ == '__main__':
    main(*sys.argv[1:5])
