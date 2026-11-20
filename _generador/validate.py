import os, sys, re, collections
sys.path.insert(0,'.')
from pdx import *
from gen import load_building_types, load_categories, num, DATE_RE
V = os.path.expanduser('~/mnt/Hearts of Iron IV'); R = os.path.expanduser('~/mnt/820260968')
OUT = sys.argv[1]
bt = load_building_types([V,R]); cats = load_categories([V,R]); cats['p2s_village'] = 0
SLOT = {k for k,v in bt.items() if v['shares'] and not v['province']}
def load(d):
    res = {}
    for fn in os.listdir(d):
        if fn.endswith('.txt'):
            st = get(read(os.path.join(d,fn)),'state'); res[fn] = st
    return res
orig = load(os.path.join(R,'history/states')); gen = load(os.path.join(OUT,'history/states'))
final = dict(orig); final.update(gen)   # override por nombre de archivo
ids = collections.Counter(int(get(s,'id')) for s in final.values())
dupids = [i for i,c in ids.items() if c>1]
prov = collections.Counter()
for s in final.values():
    for _,_,p in get(s,'provinces'): prov[int(p)] += 1
oprov = set(int(p) for s in orig.values() for _,_,p in get(s,'provinces'))
def eff_buildings(st):
    """devuelve (base_dict, final_dict) de edificios SLOT, y uso maximo"""
    h = get(st,'history', Node())
    cur = collections.Counter(); snaps = []
    def apply(b):
        for k,_,v in b:
            if k in SLOT and num(v) is not None: cur[k] = int(num(v))
    def walk(node, dated):
        for k,_,v in node:
            if k=='buildings' and isinstance(v,Node): apply(v)
            elif DATE_RE.match(k): walk(v, True); snaps.append(dict(cur))
            elif k.lower()=='if' and isinstance(v,Node): walk(v, dated)
    walk(h, False)
    snaps.append(dict(cur))
    return snaps
problems = collections.Counter(); ex = {}
def prob(k, msg):
    problems[k]+=1; ex.setdefault(k, msg)
# agrupar piezas por estado original
orig_by_id = {int(get(s,'id')): s for s in orig.values()}
groups = collections.defaultdict(list)
o_prov_state = {int(p): int(get(s,'id')) for s in orig.values() for _,_,p in get(s,'provinces')}
for s in final.values():
    ps = [int(p) for _,_,p in get(s,'provinces')]
    groups[o_prov_state[ps[0]]].append(s)
for oid, pieces in groups.items():
    o = orig_by_id[oid]
    mp_o = int(num(get(o,'manpower','0'))); mp_g = sum(int(num(get(s,'manpower','0'))) for s in pieces)
    if mp_o != mp_g: prob('manpower', (oid, mp_o, mp_g))
    ro = collections.Counter({k: num(v) for k,_,v in get(o,'resources',Node())})
    rg = collections.Counter()
    for s in pieces:
        for k,_,v in get(s,'resources',Node()): rg[k] += num(v)
    for k in set(ro)|set(rg):
        if abs(ro[k]-rg[k])>1e-6: prob('resources', (oid,k,ro[k],rg[k]))
    so = eff_buildings(o); sg = [eff_buildings(s) for s in pieces]
    for si in range(len(so)):
        for k in SLOT:
            a = so[si].get(k,0); b = sum(x[min(si,len(x)-1)].get(k,0) for x in sg)
            if a != b: prob('factories', (oid, si, k, a, b))
    for s in pieces:
        h = get(s,'history',Node())
        if get(h,'owner') is None: prob('owner', int(get(s,'id')))
        cat = get_all(s,'state_category')[-1].strip('"'); extra = int(num(get(h,'add_extra_state_shared_building_slots','0')))
        cap = cats.get(cat, None)
        if cap is None: prob('category', cat); continue
        use = max(sum(x.values()) for x in eff_buildings(s))
        if use > cap + extra: prob('slots', (int(get(s,'id')), use, cap, extra))
        if get(get(h,'buildings',Node()),'infrastructure') is None and get(get(o,'history'),'buildings') is not None and get(get(get(o,'history'),'buildings'),'infrastructure') is not None:
            prob('infra', int(get(s,'id')))
    vo = len(get_all(get(o,'history'),'victory_points')); vg = sum(len(get_all(get(s,'history'),'victory_points')) for s in pieces)
    if vo != vg: prob('vp', (oid, vo, vg))
print('estados finales', len(final), 'ids duplicados', dupids[:5])
print('provincias originales', len(oprov), 'asignadas', len(prov), 'repetidas', [p for p,c in prov.items() if c>1][:5], 'faltan', len(oprov-set(prov)))
print('problemas', dict(problems)); 
for k,v in ex.items(): print(' ej', k, v)
