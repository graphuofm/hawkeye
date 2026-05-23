#!/usr/bin/env python3
"""Collect all Hawkeye sweep results into a mean +/- std summary table.
Run on iTiger: python3 collect_results.py"""
import json, glob, os, re, subprocess
from collections import defaultdict

ROOTS = [
    '/project/jding2/hawkeye/sota/DyGLib/saved_results/DyGFormer',
    '/project/jding2/hawkeye/sota/DyGLib_TGB/saved_results/DyGFormer',
]
# group[(dataset, channel, wf)][seed] = (test, newnode)
group = defaultdict(dict)
for root in ROOTS:
    for f in glob.glob(root + '/*/*.json'):
        ds = os.path.basename(os.path.dirname(f))
        name = os.path.basename(f)[:-5]
        m = re.match(r'DyGFormer_struct([a-z]+)(?:_wf([\d.]+)_[a-z]+)?_seed(\d+)$', name)
        if not m:
            continue
        ch, wf, seed = m.group(1), m.group(2), int(m.group(3))
        try:
            d = json.load(open(f))
        except Exception:
            continue
        tm = d.get('test metrics', {})
        nm = d.get('new node test metrics', {})
        test = tm.get('average_precision', tm.get('mrr'))
        nn = nm.get('average_precision', nm.get('mrr'))
        group[(ds, ch, wf or '-')][seed] = (test, nn)

def ms(vals):
    vals = [float(v) for v in vals if v is not None]
    if not vals:
        return '   -   '
    mean = sum(vals) / len(vals)
    if len(vals) == 1:
        return f'{mean:.4f}     '
    var = sum((x - mean) ** 2 for x in vals) / (len(vals) - 1)
    return f'{mean:.4f}+-{var**0.5:.4f}'

print('=' * 78)
print('HAWKEYE SWEEP RESULTS  (test / new-node ;  AP for DGB, MRR for tgbl-wiki)')
print('=' * 78)
order = ['none', 'cooccur', 'gev', 'both']
last_ds = None
for (ds, ch, wf) in sorted(group, key=lambda k: (k[0], k[2], order.index(k[1]) if k[1] in order else 9)):
    if ds != last_ds:
        print(f'\n[{ds}]')
        last_ds = ds
    seeds = group[(ds, ch, wf)]
    tag = ch + ('' if wf in ('-', '0.0') else f' wf{wf}')
    test = ms([v[0] for v in seeds.values()])
    nn = ms([v[1] for v in seeds.values()])
    print(f'  {tag:16s}  seeds={sorted(seeds)}  test={test}  new-node={nn}')

print('\n' + '=' * 78)
try:
    q = subprocess.run(['squeue', '-u', 'jding2', '-h', '-o', '%t'],
                       capture_output=True, text=True, timeout=20).stdout.split()
    from collections import Counter
    print('QUEUE:', dict(Counter(q)))
    fail = subprocess.run(['sacct', '-u', 'jding2', '-n', '-X', '-o', 'JobName%24,State',
                           '-S', '2026-05-22'], capture_output=True, text=True, timeout=20).stdout
    bad = sorted(set(l.strip() for l in fail.splitlines()
                     if 'FAILED' in l or 'TIMEOUT' in l or 'CANCEL' in l and 'hk_' in l))
    print(f'FAILED/CANCELLED jobs: {len(bad)}')
    for b in bad[:20]:
        print('  ', b)
except Exception as e:
    print('queue check skipped:', e)
