#!/usr/bin/env python3
"""
Provincias -> Estados para Hearts of Iron IV + The Road to 56
Genera un mod donde cada provincia terrestre es su propio estado.
Uso: python gen.py <carpeta HOI4> <carpeta Road to 56> <carpeta de salida del mod>
Requiere: Python 3.8+, numpy, pillow
"""
import os, re, sys, math, shutil, collections, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pdx import Node, parse, dump, get, get_all, read
from maptools import analyse, load_definition

# ---------------- configuracion ----------------
MIN_SLOTS_PER_PROVINCE = 0      # slots minimos por provincia (0 = conserva el total original de cada estado)
NEW_CATEGORY = 'p2s_village'    # categoria de 0 slots para provincias sin industria
SPECIAL_CATS = {'wasteland', 'enclave', 'tiny_island', 'small_island', 'large_island'}
EFFECT_KEYS = ('transfer_state', 'add_state_core', 'add_state_claim', 'remove_state_core', 'remove_state_claim')
SAFE_STATE_EFFECTS = {'add_core_of', 'add_claim_by', 'remove_core_of', 'remove_claim_by'}
DATE_RE = re.compile(r'^\d+\.\d+\.\d+$')
TAG_RE = re.compile(r'^[A-Z][A-Z0-9]{2}$')
PROV_LINE_TYPES = {'supply_node', 'special_project_facility_spawn', 'bunker', 'naval_supply_hub', 'naval_base_spawn',
                   'coastal_bunker', 'floating_harbor', 'naval_headquarters', 'dam_spawn', 'landmark_spawn', 'locks_spawn'}
log_lines = []
def log(*a):
    s = ' '.join(str(x) for x in a); log_lines.append(s); print(s)

def dhondt(total, weights, eligible=None):
    """Asignacion secuencial (monotona): total unidades entre piezas segun pesos."""
    n = len(weights); alloc = [0] * n
    idx = [i for i in range(n) if eligible is None or eligible[i]]
    if not idx: idx = [0]
    for _ in range(int(total)):
        best = max(idx, key=lambda i: (weights[i] / (alloc[i] + 1), -i))
        alloc[best] += 1
    return alloc

def hamilton(total, weights, minimum=0):
    n = len(weights)
    if total < minimum * n: minimum = 0
    rest = total - minimum * n
    sw = sum(weights) or 1
    raw = [rest * w / sw for w in weights]
    alloc = [int(math.floor(r)) for r in raw]
    left = int(rest - sum(alloc))
    order = sorted(range(n), key=lambda i: (-(raw[i] - alloc[i]), i))
    for i in order[:left]: alloc[i] += 1
    return [a + minimum for a in alloc]

def num(v):
    try: return float(v)
    except: return None

def fmt(v):
    return str(int(v)) if float(v).is_integer() else f'{v:.3f}'

# ---------------- carga de datos del juego ----------------
def effective_files(roots, sub, ext='.txt'):
    eff = {}
    for r in roots:
        d = os.path.join(r, sub)
        if not os.path.isdir(d): continue
        for dp, dn, fn in os.walk(d):
            for f in fn:
                if f.endswith(ext):
                    p = os.path.join(dp, f); eff[os.path.relpath(p, r).replace('\\', '/')] = p
    return eff

def load_building_types(roots):
    types = {}
    for rel, p in effective_files(roots, 'common/buildings').items():
        b = get(read(p), 'buildings')
        if b is None: continue
        for k, op, v in b:
            if not isinstance(v, Node): continue
            lc = get(v, 'level_cap', Node())
            types[k] = dict(shares=get(lc, 'shares_slots') == 'yes', province=get(lc, 'province_max') is not None,
                            coastal=get(v, 'only_costal') == 'yes', infra=get(v, 'infrastructure') == 'yes')
    return types

def load_categories(roots):
    cats = {}
    for rel, p in effective_files(roots, 'common/state_category').items():
        sc = get(read(p), 'state_categories')
        if sc is None: continue
        for k, op, v in sc:
            if isinstance(v, Node): cats[k] = int(float(get(v, 'local_building_slots', '0')))
    return cats

LOC_RE = re.compile(r'^\s*([A-Za-z0-9_.\-]+):\d*\s*"(.*)"\s*(#.*)?$')
def load_loc(roots, lang):
    out = {}
    for r in roots:
        d = os.path.join(r, 'localisation', lang)
        if not os.path.isdir(d): continue
        for dp, dn, fn in os.walk(d):
            for f in sorted(fn):
                if not f.endswith('.yml'): continue
                try: lines = open(os.path.join(dp, f), encoding='utf-8-sig', errors='replace').read().splitlines()
                except Exception: continue
                for line in lines:
                    if 'STATE_' not in line and 'VICTORY_POINTS_' not in line: continue
                    m = LOC_RE.match(line)
                    if m and (m.group(1).startswith('STATE_') or m.group(1).startswith('VICTORY_POINTS_')):
                        out[m.group(1)] = m.group(2)
    return out

# ---------------- division de un estado ----------------
class Splitter:
    def __init__(self, st, fname, M, btypes, cats, next_id):
        self.st, self.fname, self.M, self.bt, self.cats = st, fname, M, btypes, cats
        self.sid = int(get(st, 'id'))
        provs = [int(v) for _, _, v in get(st, 'provinces', Node())]
        land = M['land']
        self.land = [p for p in provs if land[p]]
        self.lakes = [p for p in provs if not land[p]]
        self.provs = provs
        hist = get(st, 'history', Node())
        self.vp = collections.Counter()
        for vp in get_all(hist, 'victory_points'):
            vals = [v for _, _, v in vp]
            if len(vals) >= 2: self.vp[int(vals[0])] += float(vals[1])
        area = M['area']
        if self.land:
            main = max(self.land, key=lambda p: (self.vp.get(p, 0), area[p], -p))
        else:
            main = provs[0]
        others = [p for p in self.land if p != main]
        others.sort(key=lambda p: (-self.vp.get(p, 0), -area[p], p))
        self.pieces = [main] + others                     # una provincia terrestre por pieza
        self.ids = [self.sid] + list(range(next_id, next_id + len(others)))
        self.piece_of = {p: i for i, p in enumerate(self.pieces)}
        # lagos -> pieza terrestre vecina
        nb = M['nbr']
        for lk in self.lakes:
            cand = [self.piece_of[q] for q in nb.get(lk, ()) if q in self.piece_of]
            self.piece_of[lk] = 0 if (not cand or 0 in cand) else min(cand)
        self.coastal = [bool(M['coastal'][p]) for p in self.pieces]
        self.w_fac = [1.0 + self.vp.get(p, 0) for p in self.pieces]
        self.w_free = [1.0 + self.vp.get(p, 0) / 50.0 for p in self.pieces]
        self.n = len(self.pieces)
        self.unknown = collections.Counter()

    def pidx(self, prov):
        try: return self.piece_of.get(int(prov), 0)
        except: return 0

    # ---- edificios ----
    def alloc_type(self, t, total):
        info = self.bt.get(t, {})
        elig = self.coastal if info.get('coastal') else None
        if elig is not None and not any(elig): elig = None
        return dhondt(total, self.w_fac, elig)

    def split_buildings(self, b, state):
        """state: dict type->alloc list (persistente entre bloques fechados). Devuelve lista de Node por pieza."""
        out = [Node() for _ in range(self.n)]
        for k, op, v in b:
            if k.isdigit() and isinstance(v, Node):
                i = self.pidx(k); out[i].append((k, op, v))
                for kk, _, vv in v:
                    if self.bt.get(kk, {}).get('shares') and num(vv) is not None:
                        self.prov_slot_use[i] = max(self.prov_slot_use[i], 0) + int(num(vv))
                continue
            info = self.bt.get(k)
            if info and info['infra'] or k == 'infrastructure':
                for i in range(self.n): out[i].append((k, op, v))
            elif info and info['shares'] and not info['province'] and num(v) is not None:
                new = self.alloc_type(k, int(num(v)))
                old = state.get(k)
                for i in range(self.n):
                    if old is None:
                        if new[i] > 0: out[i].append((k, '=', str(new[i])))
                    elif new[i] != old[i]:
                        out[i].append((k, '=', str(new[i])))
                state[k] = new
            else:
                out[0].append((k, op, v))
        return out

    def slot_usage(self, state):
        use = [0] * self.n
        for t, a in state.items():
            for i in range(self.n): use[i] += a[i]
        return [use[i] + self.prov_slot_use[i] for i in range(self.n)]

    # ---- historia ----
    def route(self, node, ctx):
        """Reparte una lista de sentencias de historia entre las piezas."""
        out = [Node() for _ in range(self.n)]
        for k, op, v in node:
            if DATE_RE.match(k) and isinstance(v, Node):
                sub = self.route(v, 'date')
                for i in range(self.n):
                    if sub[i]: out[i].append((k, op, sub[i]))
                self.usage_snapshots.append(self.slot_usage(self.bstate))
            elif k == 'buildings' and isinstance(v, Node):
                if ctx == 'base' and self.bstate == {}:
                    parts = self.split_buildings(v, self.bstate)
                else:
                    parts = self.split_buildings(v, self.bstate)
                for i in range(self.n):
                    if parts[i]: out[i].append((k, op, parts[i]))
            elif k == 'victory_points' and isinstance(v, Node):
                vals = [x for _, _, x in v]
                out[self.pidx(vals[0]) if vals else 0].append((k, op, v))
            elif k == 'add_to_array' and isinstance(v, Node) and get(v, 'state_victory_points') is not None:
                out[self.pidx(get(v, 'state_victory_points'))].append((k, op, v))
            elif k in ('strategic_province_location', 'strategic_state_location') and isinstance(v, Node):
                vals = [x for _, _, x in v]
                out[self.pidx(vals[0]) if vals else 0].append((k, op, v))
            elif k == 'remove_building' and isinstance(v, Node):
                t = get(v, 'type'); lvl = int(num(get(v, 'level', '1')) or 1)
                a = self.bstate.get(t)
                i = max(range(self.n), key=lambda j: (a[j], -j)) if a else 0
                if a: a[i] = max(0, a[i] - lvl)
                out[i].append((k, op, v))
            elif k == 'add_extra_state_shared_building_slots':
                self.extra_slots += int(num(v) or 0)
            elif k == 'save_global_event_target_as':
                out[0].append((k, op, v))
            elif TAG_RE.match(k) and isinstance(v, Node):
                rest = Node()
                for kk, oo, vv in v:
                    if kk == 'set_province_controller' and vv.isdigit() and int(vv) in self.piece_of:
                        tgt = out[self.pidx(vv)]
                        if ('controller', '=', k) not in tgt[-1:]: tgt.append(('controller', '=', k))
                    elif kk == 'set_province_controller':
                        out[0].append((k, op, Node([(kk, oo, vv)])))
                    else:
                        rest.append((kk, oo, vv))
                if rest:
                    for i in range(self.n): out[i].append((k, op, rest))
            elif k.lower() == 'if' and isinstance(v, Node):
                limit = [(kk, oo, vv) for kk, oo, vv in v if kk == 'limit']
                body = Node([(kk, oo, vv) for kk, oo, vv in v if kk != 'limit'])
                sub = self.route(body, ctx)
                for i in range(self.n):
                    if sub[i]: out[i].append((k, op, Node(limit + list(sub[i]))))
            else:
                for i in range(self.n): out[i].append((k, op, v))
        return out

    def run(self):
        st = self.st
        self.bstate = {}; self.prov_slot_use = [0] * self.n; self.extra_slots = 0; self.usage_snapshots = []
        hist = get(st, 'history', Node())
        hparts = self.route(hist, 'base')
        # el uso base es el estado justo despues de procesar la historia base; snapshots cubren fechas
        self.usage_snapshots.append(self.slot_usage(self.bstate))
        need = [max(s[i] for s in self.usage_snapshots) for i in range(self.n)]
        cats_all = get_all(st, 'state_category')
        cat = (cats_all[-1] if cats_all else 'rural').strip('"')
        cat_slots = self.cats.get(cat, 0)
        target = max(cat_slots + self.extra_slots, max(sum(s) for s in self.usage_snapshots))
        if MIN_SLOTS_PER_PROVINCE:
            need = [max(x, MIN_SLOTS_PER_PROVINCE) if cat not in SPECIAL_CATS else x for x in need]
        free = max(0, target - sum(need))
        freea = dhondt(free, self.w_free)
        slots = [need[i] + freea[i] for i in range(self.n)]
        # categorias
        ladder = sorted([(s, c) for c, s in self.cats.items() if c not in SPECIAL_CATS], key=lambda x: (x[0], x[1]))
        # una categoria por cantidad de slots (la primera alfabeticamente si hay empate)
        by_slots = {}
        for s, c in ladder: by_slots.setdefault(s, c)
        levels = sorted(by_slots)
        # manpower y recursos
        mp = int(num(get(st, 'manpower', '0')) or 0)
        mpw = [1.0 + 2.0 * self.vp.get(p, 0) for p in self.pieces]
        mpa = hamilton(mp, mpw, minimum=1)
        res = get(st, 'resources')
        resa = {}
        if res is not None:
            areaw = [float(self.M['area'][p]) for p in self.pieces]
            for k, op, v in res:
                val = num(v)
                if val is None: continue
                if float(val).is_integer():
                    resa[k] = hamilton(int(val), areaw)
                else:
                    a = hamilton(int(round(val * 1000)), areaw); resa[k] = [x / 1000 for x in a]
        prov_lists = [[] for _ in range(self.n)]
        for p in self.provs: prov_lists[self.pidx(p)].append(p)
        states = []
        for i in range(self.n):
            if cat in SPECIAL_CATS:
                c = cat; extra = max(0, slots[i] - cat_slots)
            else:
                lv = max([l for l in levels if l <= slots[i]] or [levels[0]])
                c = by_slots[lv]; extra = slots[i] - lv
            h = Node(hparts[i])
            if extra > 0: h.append(('add_extra_state_shared_building_slots', '=', str(extra)))
            s = Node()
            s.append(('id', '=', str(self.ids[i])))
            s.append(('name', '=', f'"STATE_{self.ids[i]}"'))
            s.append(('manpower', '=', str(mpa[i])))
            for k in ('buildings_max_level_factor', 'impassable', 'force_link_ownership_to', 'local_supplies'):
                v = get(st, k)
                if v is not None: s.append((k, '=', v))
            for v in get_all(st, 'impassable_ignored_links'): s.append(('impassable_ignored_links', '=', v))
            s.append(('state_category', '=', c))
            if resa:
                r = Node([(k, '=', fmt(a[i])) for k, a in resa.items() if a[i] > 0])
                if r: s.append(('resources', '=', r))
            s.append(('history', '=', h))
            s.append(('provinces', '=', Node([('', None, str(p)) for p in prov_lists[i]])))
            states.append(s)
        self.slots = slots; self.mpa = mpa; self.resa = resa
        return states

# ---------------- buildings.txt ----------------
def rebuild_buildings(path, M, prov_state, orig_of_state, state_provs_orig, pieces_info):
    pid = M['pid']; H, W = pid.shape; hm = M['hm']
    lines = []
    with open(path, encoding='utf-8-sig', errors='replace') as f:
        for line in f:
            p = line.strip().split(';')
            if len(p) >= 7 and p[0].lstrip('-').isdigit(): lines.append(p)
    per_state_type = collections.Counter()
    for p in lines: per_state_type[(int(p[0]), p[1])] += 1
    req = collections.Counter()
    for (s, t), c in per_state_type.items():
        if t not in PROV_LINE_TYPES: req[t] = max(req[t], c)
    has_type = collections.defaultdict(set)
    for (s, t) in per_state_type: has_type[s].add(t)
    out = []; have = collections.Counter(); moved = 0; fallback = 0; fb_types = collections.Counter()
    for p in lines:
        s = int(p[0]); x, z = float(p[2]), float(p[4])
        xi = int(x) % W; zi = min(max(int(z), 0), H - 1); row = H - 1 - zi
        prov = int(pid[row, xi])
        if orig_of_state.get(prov_state.get(prov)) != s:
            # buscar el pixel mas cercano perteneciente al estado original
            prov = None
            sp = state_provs_orig.get(s, set())
            for r in (2, 5, 10, 20, 40):
                r0, r1 = max(0, row - r), min(H, row + r + 1)
                cols = np.arange(xi - r, xi + r + 1) % W
                win = pid[r0:r1][:, cols]
                mask = np.isin(win, list(sp))
                if mask.any():
                    yy, xx = np.nonzero(mask)
                    d = (yy + r0 - row) ** 2 + (xx - r) ** 2
                    j = int(np.argmin(d)); prov = int(win[yy[j], xx[j]]); break
            if prov is None:
                fallback += 1; new_s = s; fb_types[p[1]] += 1
            else:
                new_s = prov_state[prov]
        else:
            new_s = prov_state[prov]
        if new_s != s: moved += 1
        p[0] = str(new_s); out.append(p); have[(new_s, p[1])] += 1
    # equilibrar edificios provinciales (una posicion por provincia)
    PROV_ONE = {'bunker': False, 'supply_node': False, 'special_project_facility_spawn': False,
                'naval_base_spawn': True, 'naval_supply_hub': True, 'coastal_bunker': True}
    by_orig = collections.defaultdict(list)
    for ns, (os_, pr, co) in pieces_info.items(): by_orig[os_].append(ns)
    lines_by = collections.defaultdict(list)
    for p in out: lines_by[(int(p[0]), p[1])].append(p)
    rebal = 0
    for t, need_coast in PROV_ONE.items():
        for os_, group in by_orig.items():
            if len(group) < 2: continue
            empty = [g for g in group if (pieces_info[g][2] or not need_coast) and not lines_by.get((g, t))]
            if not empty: continue
            spare = [p for g in group for p in lines_by.get((g, t), [])[1:]]
            for g in empty:
                if not spare: break
                ry, rx = divmod(int(M['rep'][pieces_info[g][1]]), W); rz = H - 1 - ry
                spare.sort(key=lambda p: (float(p[2]) - rx) ** 2 + (float(p[4]) - rz) ** 2)
                p = spare.pop(0)
                lines_by[(int(p[0]), t)].remove(p); have[(int(p[0]), t)] -= 1
                p[0] = str(g); lines_by[(g, t)].append(p); have[(g, t)] += 1; rebal += 1
    for t, need_coast in PROV_ONE.items():
        req[t] = 1
    log(f'buildings.txt: {rebal} posiciones provinciales reequilibradas')
    # generar posiciones que falten
    order = M['order']; flat_pid = pid.ravel()
    # indices de inicio por provincia en 'order'
    fs = flat_pid[order]
    starts = np.searchsorted(fs, np.arange(len(M['area']) + 1))
    gen = 0
    for new_s, (orig_s, prov, coastal) in pieces_info.items():
        types = has_type.get(orig_s, set())
        for t, need in req.items():
            if t not in types: continue
            if (t == 'dockyard' or PROV_ONE.get(t)) and not coastal: continue
            missing = need - have[(new_s, t)]
            if missing <= 0: continue
            pix = order[starts[prov]:starts[prov + 1]]
            if len(pix) == 0: continue
            rep = M['rep'][prov]; ry, rx = divmod(int(rep), W)
            py, px = np.divmod(pix, W)
            d = (py - ry) ** 2 + (px - rx) ** 2
            cand = pix[np.argsort(d, kind='stable')]
            for k in range(missing):
                j = int(cand[(sum(map(ord, t)) % 7 + k * 3) % max(1, min(len(cand), 60))])
                yy, xx = divmod(j, W)
                y = hm[yy, xx] / 10.0
                out.append([str(new_s), t, f'{xx:.2f}', f'{y:.2f}', f'{H - 1 - yy:.2f}', '0.00', '0'])
                gen += 1
    out.sort(key=lambda p: (int(p[0]), p[1]))
    if fb_types: log('  lineas sin provincia por tipo: ' + str(dict(fb_types)))
    return out, moved, fallback, gen

# ---------------- parches de scripts ----------------
TOKPOS = re.compile(r'\s+|#[^\n]*|"(?:[^"\\]|\\.)*"|<=|>=|!=|[{}=<>]|[^\s{}=<>#"]+')
def patch_script(text, pieces_map, max_orig):
    toks = [(m.start(), m.end(), m.group(0)) for m in TOKPOS.finditer(text)
            if not (m.group(0)[0].isspace() or m.group(0)[0] == '#')]
    edits = []  # (start, end, replacement)
    stack = []  # claves de bloque abiertas
    n = len(toks); i = 0; changed = 0
    def ids_for(x):
        return pieces_map.get(int(x)) if x.isdigit() else None
    while i < n:
        s, e, t = toks[i]
        if t == '{':
            key = toks[i - 2][2] if i >= 2 and toks[i - 1][2] == '=' else ''
            stack.append((key, i)); i += 1; continue
        if t == '}':
            if stack: stack.pop()
            i += 1; continue
        if i + 2 < n and toks[i + 1][2] == '=':
            v = toks[i + 2][2]
            if t in EFFECT_KEYS and v.isdigit() and ids_for(v):
                rep = ' '.join(f'{t} = {x}' for x in ids_for(v))
                edits.append((s, toks[i + 2][1], rep)); changed += 1; i += 3; continue
            if t == 'states' and v == '{':
                # lista numerica?
                j = i + 3; nums = []
                while j < n and toks[j][2] != '}' and toks[j][2].isdigit(): nums.append(toks[j][2]); j += 1
                if j < n and toks[j][2] == '}' and nums and any(ids_for(x) for x in nums):
                    allids = []
                    for x in nums: allids += ids_for(x) or [int(x)]
                    edits.append((toks[i + 2][0], toks[j][1], '{ ' + ' '.join(map(str, allids)) + ' }'))
                    changed += 1; i = j + 1; continue
            if t.isdigit() and v == '{' and ids_for(t):
                parent = stack[-1][0] if stack else ''
                if parent not in ('random_list', 'random', 'ai_will_do', 'ai_chance', 'modifier', 'random_events'):
                    # buscar cierre y comprobar contenido
                    depth = 0; j = i + 2; ok = True; keys = []
                    while j < n:
                        tt = toks[j][2]
                        if tt == '{': depth += 1
                        elif tt == '}':
                            depth -= 1
                            if depth == 0: break
                        elif depth == 1 and j + 1 < n and toks[j + 1][2] == '=':
                            keys.append(tt)
                            if j + 2 < n and toks[j + 2][2] == '{': ok = False
                        j += 1
                    if ok and keys and all(k in SAFE_STATE_EFFECTS for k in keys) and j < n:
                        body = text[toks[i + 2][0]:toks[j][1]]
                        rep = ' '.join(f'{x} = {body}' for x in ids_for(t))
                        edits.append((s, toks[j][1], rep)); changed += 1; i = j + 1; continue
        i += 1
    if not edits: return None, 0
    out = []; last = 0
    for s, e, r in sorted(edits):
        out.append(text[last:s]); out.append(r); last = e
    out.append(text[last:])
    return ''.join(out), changed

# ---------------- principal ----------------
import pickle
def read_text(p):
    raw = open(p, 'rb').read()
    bom = raw.startswith(b'\xef\xbb\xbf')
    try:
        return raw.decode('utf-8-sig'), bom, 'utf-8'
    except UnicodeDecodeError:
        return raw.decode('cp1252', errors='replace'), False, 'cp1252'

def phase_states(van, r56, out, work):
    t0 = time.time()
    roots = [van, r56]
    M = analyse(os.path.join(r56, 'map'), os.path.join(work, 'mapcache.npz'))
    nbr = collections.defaultdict(set)
    for a, b in M['edges']:
        nbr[int(a)].add(int(b)); nbr[int(b)].add(int(a))
    M['nbr'] = nbr
    btypes = load_building_types(roots)
    cats = load_categories(roots); cats[NEW_CATEGORY] = 0
    sdir = os.path.join(r56, 'history', 'states')
    parsed = []
    for fn in sorted(os.listdir(sdir)):
        if not fn.endswith('.txt'): continue
        for st in get_all(read(os.path.join(sdir, fn)), 'state'): parsed.append((fn, st))
    max_id = max(int(get(st, 'id')) for fn, st in parsed)
    next_id = max_id + 1
    os.makedirs(os.path.join(out, 'history', 'states'), exist_ok=True)
    D = dict(prov_state={}, orig_of_state={}, state_provs_orig={}, pieces_info={}, pieces_map={}, names={}, max_id=max_id)
    tot_mp_in = tot_mp_out = 0; nsplit = nnew = 0
    fac_in = collections.Counter(); fac_out = collections.Counter(); slots_in = slots_out = 0
    for fn, st in sorted(parsed, key=lambda x: int(get(x[1], 'id'))):
        sp = Splitter(st, fn, M, btypes, cats, next_id)
        sid = sp.sid
        D['state_provs_orig'][sid] = set(sp.provs)
        D['orig_of_state'][sid] = sid
        if sp.n <= 1:
            for p in sp.provs: D['prov_state'][p] = sid
            D['pieces_info'][sid] = (sid, sp.pieces[0], bool(M['coastal'][sp.pieces[0]]))
            continue
        nsplit += 1
        res_states = sp.run()
        next_id += sp.n - 1; nnew += sp.n - 1
        D['pieces_map'][sid] = sp.ids
        tot_mp_in += int(num(get(st, 'manpower', '0')) or 0); tot_mp_out += sum(sp.mpa)
        for k, a in sp.bstate.items(): fac_out[k] += sum(a)
        cat = get(st, 'state_category'); slots_in += cats.get(cat, 0); slots_out += sum(sp.slots)
        for i, s in enumerate(res_states):
            nid = sp.ids[i]
            D['orig_of_state'][nid] = sid
            D['pieces_info'][nid] = (sid, sp.pieces[i], sp.coastal[i])
            D['names'][nid] = (sid, sp.pieces[i], i + 1)
            for _, _, p in get(s, 'provinces'): D['prov_state'][int(p)] = nid
            if i == 0:
                fname = fn
            else:
                slug = re.sub(r'[^A-Za-z0-9_\- ]', '', fn.split('-', 1)[-1].rsplit('.', 1)[0])[:40].strip()
                fname = f'{nid}-{slug} {i + 1}.txt'
            with open(os.path.join(out, 'history', 'states', fname), 'w', encoding='utf-8', newline='\n') as f:
                f.write(dump(Node([('state', '=', s)])) + '\n')
    log(f'Estados originales: {len(parsed)} | divididos: {nsplit} | estados nuevos: {nnew} | total: {len(parsed) + nnew}')
    log(f'Manpower (estados divididos) original {tot_mp_in} -> generado {tot_mp_out}')
    log(f'Slots compartidos (estados divididos) categoria original {slots_in} -> generado {slots_out}')
    log('Fabricas (estado final tras fechas): ' + ', '.join(f'{k}={v}' for k, v in sorted(fac_out.items())))
    os.makedirs(os.path.join(out, 'common', 'state_category'), exist_ok=True)
    with open(os.path.join(out, 'common', 'state_category', 'p2s_village.txt'), 'w', encoding='utf-8') as f:
        f.write('state_categories = {\n\t%s = {\n\t\tlocal_building_slots = 0\n\t\tcolor = { 120 110 90 }\n\t}\n}\n' % NEW_CATEGORY)
    pickle.dump(D, open(os.path.join(work, 'split.pkl'), 'wb'))
    log(f'[estados] {time.time() - t0:.1f}s')

def phase_buildings(van, r56, out, work):
    t0 = time.time()
    D = pickle.load(open(os.path.join(work, 'split.pkl'), 'rb'))
    M = analyse(None, os.path.join(work, 'mapcache.npz'))
    bpath = os.path.join(r56, 'map', 'buildings.txt')
    if not os.path.exists(bpath): bpath = os.path.join(van, 'map', 'buildings.txt')
    blines, moved, fallback, gen = rebuild_buildings(bpath, M, D['prov_state'], D['orig_of_state'], D['state_provs_orig'], D['pieces_info'])
    os.makedirs(os.path.join(out, 'map'), exist_ok=True)
    with open(os.path.join(out, 'map', 'buildings.txt'), 'w', encoding='utf-8', newline='\n') as f:
        for p in blines: f.write(';'.join(p) + '\n')
    log(f'buildings.txt: {len(blines)} lineas, {moved} reasignadas, {gen} generadas, {fallback} sin provincia [{time.time() - t0:.1f}s]')

def phase_loc(van, r56, out, work):
    D = pickle.load(open(os.path.join(work, 'split.pkl'), 'rb'))
    roots = [van, r56]
    langs = sorted(d for d in os.listdir(os.path.join(van, 'localisation')) if os.path.isdir(os.path.join(van, 'localisation', d)))
    eng = load_loc(roots, 'english')
    for lang in langs:
        loc = eng if lang == 'english' else {**eng, **load_loc(roots, lang)}
        d = os.path.join(out, 'localisation', lang); os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, f'p2s_state_names_l_{lang}.yml'), 'w', encoding='utf-8-sig', newline='\n') as f:
            f.write(f'l_{lang}:\n')
            for nid, (sid, prov, idx) in sorted(D['names'].items()):
                if nid == sid: continue
                base = loc.get(f'STATE_{sid}', f'State {sid}')
                vpn = loc.get(f'VICTORY_POINTS_{prov}')
                nm = vpn if vpn else f'{base} ({idx})'
                f.write(f' STATE_{nid}:0 "{nm}"\n')
    log(f'Localizacion: {len(langs)} idiomas ({", ".join(langs)})')

def phase_patch(van, r56, out, work, part=0, nparts=1):
    t0 = time.time()
    D = pickle.load(open(os.path.join(work, 'split.pkl'), 'rb'))
    roots = [van, r56]
    files = []
    for sub in ('common', 'events'):
        files += sorted(effective_files(roots, sub).items())
    files = files[part::nparts]
    pc = pf = 0
    quick = re.compile(r'\b(?:' + '|'.join(EFFECT_KEYS) + r'|states)\s*=|(?<![\w.:@])\d+\s*=\s*\{')
    for rel, p in files:
        text, bom, enc = read_text(p)
        if not quick.search(text): continue
        new, c = patch_script(text, D['pieces_map'], D['max_id'])
        if new is None: continue
        dst = os.path.join(out, rel); os.makedirs(os.path.dirname(dst), exist_ok=True)
        with open(dst, 'wb') as f:
            f.write((b'\xef\xbb\xbf' if bom else b'') + new.encode(enc, errors='replace'))
        pc += c; pf += 1
    log(f'Scripts parcheados (parte {part + 1}/{nparts}): {pf} archivos, {pc} referencias expandidas [{time.time() - t0:.1f}s]')

def write_log(out):
    with open(os.path.join(out, 'p2s_log.txt'), 'a', encoding='utf-8') as f: f.write('\n'.join(log_lines) + '\n')

if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(description='Cada provincia de HOI4 + Road to 56 se vuelve un estado')
    ap.add_argument('hoi4'); ap.add_argument('r56'); ap.add_argument('out')
    ap.add_argument('--phase', default='all', choices=['all', 'states', 'buildings', 'loc', 'patch'])
    ap.add_argument('--part', type=int, default=0); ap.add_argument('--nparts', type=int, default=1)
    ap.add_argument('--work', default=None)
    a = ap.parse_args()
    import tempfile
    work = a.work or os.path.join(tempfile.gettempdir(), 'p2s_cache')
    os.makedirs(work, exist_ok=True); os.makedirs(a.out, exist_ok=True)
    if a.phase in ('all', 'states'): phase_states(a.hoi4, a.r56, a.out, work)
    if a.phase in ('all', 'buildings'): phase_buildings(a.hoi4, a.r56, a.out, work)
    if a.phase in ('all', 'loc'): phase_loc(a.hoi4, a.r56, a.out, work)
    if a.phase in ('all', 'patch'): phase_patch(a.hoi4, a.r56, a.out, work, a.part, a.nparts)
    write_log(a.out)
