"""Reproducible sequential Mac validation; prepare_reference.py must run first."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import numpy as np

p = argparse.ArgumentParser()
p.add_argument('--executable', type=Path, required=True)
p.add_argument('--model', type=Path, required=True)
p.add_argument('--reference', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
a.output.mkdir(parents=True, exist_ok=False)
scripts = Path(__file__).resolve().parent
short = json.loads((a.reference/'multilingual-request.json').read_text())
long = json.loads((a.reference/'long-request.json').read_text())
summary = []

def run(name, request, fp32=False, reference=None, invalid=False):
    path = a.output/f'{name}-request.json'
    path.write_text(json.dumps(request, ensure_ascii=False))
    out = a.output/name
    cmd = [str(a.executable.resolve()), str(a.model.resolve()), str(path.resolve())]
    if fp32:
        cmd.append('--float32')
    result = subprocess.run([sys.executable, str(scripts/'guard.py'), '--output', str(out), '--timeout', '180', '--', *cmd], capture_output=True, text=True)
    receipt = json.loads((out/'receipt.json').read_text())
    if invalid:
        assert receipt['status']=='failed' and receipt['exit_code']==1, receipt
        assert 'Error:' in (out/'stderr.log').read_text()
        summary.append({'case':name, 'pass':True, 'expected_rejection':True})
        return
    if result.returncode:
        raise RuntimeError(result.stdout + result.stderr)
    if reference:
        subprocess.run([sys.executable, str(scripts/'audit.py'), str(a.reference/reference), str(out/'stdout.json'), '--output', str(out/'audit.json')], check=True)
        summary.append(json.loads((out/'audit.json').read_text()))
    return json.loads((out/'stdout.json').read_text())

batch = run('short-bf16', short, reference='multilingual-reference.json')
run('short-fp32', short, fp32=True, reference='multilingual-reference.json')
run('long-bf16', long, reference='long-reference.json')
run('long-fp32', long, fp32=True, reference='long-reference.json')
for dim in [128,256,512]:
    run(f'mrl-{dim}', dict(short, dimensions=dim), reference='multilingual-reference.json')
single = run('single', dict(short, texts=short['texts'][:1]))
x,y = np.array(batch['embeddings'][0]), np.array(single['embeddings'][0])
cos = float(x@y/(np.linalg.norm(x)*np.linalg.norm(y)))
error = float(abs(x-y).max())
assert cos>=.999 and error<=.01
summary.append({'case':'single-vs-padded-batch', 'minimum_cosine':cos, 'maximum_absolute_error':error, 'pass':True})
run('timing', dict(short, repetitions=6))
for name, request in [
    ('dimension', dict(short, dimensions=129)),
    ('empty-batch', {'texts':[]}),
    ('too-long', {'input_ids':[[2]*8193], 'attention_mask':[[1]*8193]}),
    ('all-padding', {'input_ids':[[2,3]], 'attention_mask':[[0,0]]}),
    ('media-token', {'input_ids':[[258880]], 'attention_mask':[[1]]}),
    ('unequal-shape', {'input_ids':[[2,3],[2]], 'attention_mask':[[1,1],[1,0]]}),
    ('double-prefix', {'query':'hello','documents':['world'],'task':'search'}),
]:
    run(f'reject-{name}', request, invalid=True)
(a.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(f'PASS: {len(summary)} numerical and input validation checks')
