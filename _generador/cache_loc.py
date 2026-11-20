import os, sys, pickle, time
sys.path.insert(0, '.')
from naming import load_vp_loc
V = os.path.expanduser('~/mnt/Hearts of Iron IV'); R = os.path.expanduser('~/mnt/820260968')
out = pickle.load(open('loc_cache.pkl', 'rb')) if os.path.exists('loc_cache.pkl') else {}
langs = sys.argv[1:]
for lang in langs:
    t = time.time(); out[lang] = load_vp_loc([V, R], lang); print(lang, len(out[lang]), f'{time.time()-t:.0f}s', flush=True)
    pickle.dump(out, open('loc_cache.pkl', 'wb'))
