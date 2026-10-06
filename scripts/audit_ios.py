"""Independently recompute physical-device parity from exported vectors."""
import argparse
import json
from pathlib import Path
import numpy as np

p = argparse.ArgumentParser()
p.add_argument('result', type=Path)
p.add_argument('--reference', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
result = json.loads(a.result.read_text())
assert result['platform']=='iOS' and result['precision']=='bfloat16'
assert [c['case'] for c in result['checks']]==['multilingual','long']
checks = []
for check in result['checks']:
    reference = json.loads((a.reference/f"{check['case']}-reference.json").read_text())
    x,y = np.array(check['embeddings']),np.array(reference['vectors']['768'])
    assert x.shape==y.shape and np.isfinite(x).all()
    cosine = np.sum(x*y,axis=-1)/(np.linalg.norm(x,axis=-1)*np.linalg.norm(y,axis=-1))
    error = float(abs(x-y).max())
    norm_error = float(abs(np.linalg.norm(x,axis=-1)-1).max())
    passed = bool(cosine.min()>=.999 and error<=.01 and norm_error<=1e-5 and check['tokenization_match'])
    assert passed and check['pass'], check['case']
    checks.append({'case':check['case'],'pass':passed,'minimum_cosine':float(cosine.min()),'maximum_absolute_error':error,'maximum_norm_error':norm_error,'tokenization_match_reported_by_device':check['tokenization_match'],'encode_seconds':check['seconds']})
output = {'pass':True,'platform':result['platform'],'os':result['os'],'precision':result['precision'],'load_seconds':result['load_seconds'],'memory':result['memory'],'checks':checks,'scope':'independent CPU recomputation from device vectors; token identity was checked inside app; not a quality benchmark'}
a.output.write_text(json.dumps(output,indent=2)+'\n')
print(json.dumps(output))
