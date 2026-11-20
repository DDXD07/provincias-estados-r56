#!/usr/bin/env python3
"""Escribe los nombres calculados por naming2.py en la localizacion del mod (10 idiomas)."""
import os, sys, re, pickle
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from naming import load_vp_loc
LOC_RE = re.compile(r'^\s*(STATE_\d+):\d*\s*"(.*)"\s*$')
def main(van, r56, mod):
    here = os.path.dirname(os.path.abspath(__file__))
    R = pickle.load(open(os.path.join(here, 'naming_result.pkl'), 'rb'))
    names, state_of, unsplit = R['names'], R['state_of'], R['unsplit']
    lc = pickle.load(open(os.environ['P2S_LOC'], 'rb')) if os.environ.get('P2S_LOC') else {}
    roots = [van, r56]
    langs = sorted(d for d in os.listdir(os.path.join(van, 'localisation')) if os.path.isdir(os.path.join(van, 'localisation', d)))
    en = lc.get('english') or load_vp_loc(roots, 'english')
    split_main = set(R['split_ids'])
    stats = {}
    for lang in langs:
        loc = {**en, **(lc.get(lang) or load_vp_loc(roots, lang))}
        d = os.path.join(mod, 'localisation', lang)
        old = {}
        p_old = os.path.join(d, f'p2s_state_names_l_{lang}.yml')
        if os.path.exists(p_old):
            for line in open(p_old, encoding='utf-8-sig'):
                m = LOC_RE.match(line)
                if m: old[m.group(1)] = m.group(2)
        new_lines, rep_lines = [], []
        per_state = {}
        for q, sid in state_of.items():
            if sid in unsplit: continue
            if q in names: per_state[sid] = q
            else: per_state.setdefault(sid, None)
        for sid in sorted(per_state):
            q = per_state[sid]
            if q is None:
                if sid not in split_main and f'STATE_{sid}' in old:
                    new_lines.append(f' STATE_{sid}:0 "{old[f"STATE_{sid}"]}"\n')
                continue
            src, nm = names[q]
            if src == 'juego': nm = loc.get(f'VICTORY_POINTS_{q}', nm)
            nm = nm.replace('"', "'")
            line = f' STATE_{sid}:0 "{nm}"\n'
            (rep_lines if sid in split_main else new_lines).append(line)
        os.makedirs(os.path.join(d, 'replace'), exist_ok=True)
        with open(p_old, 'w', encoding='utf-8-sig', newline='\n') as f:
            f.write(f'l_{lang}:\n' + ''.join(new_lines))
        with open(os.path.join(d, 'replace', f'p2s_state_names_replace_l_{lang}.yml'), 'w', encoding='utf-8-sig', newline='\n') as f:
            f.write(f'l_{lang}:\n' + ''.join(rep_lines))
        stats[lang] = (len(new_lines), len(rep_lines))
    print('Escritos (nuevos, reemplazos):', stats)
if __name__ == '__main__':
    main(*sys.argv[1:4])
