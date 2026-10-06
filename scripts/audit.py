"""Compare Swift output to independent frozen upstream FP32 reference; fail closed."""
import argparse,json
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('reference',type=Path);p.add_argument('swift',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
r=json.loads(a.reference.read_text());s=json.loads(a.swift.read_text());dim=s['dimensions']
expected=np.array(r['vectors'][str(dim)]);actual=np.array(s['embeddings'])
assert expected.shape==actual.shape
assert np.isfinite(actual).all()
cos=(actual*expected).sum(-1)/(np.linalg.norm(actual,axis=-1)*np.linalg.norm(expected,axis=-1))
error=np.abs(actual-expected).max();norm_error=np.abs(np.linalg.norm(actual,axis=-1)-1).max()
# JSON CLI stores unpadded tokenization; compare to reference after removal of masked padding.
ref_ids=[[t for t,m in zip(row,mask) if m] for row,mask in zip(r['input_ids'],r['attention_mask'])]
ids=s['input_ids'];tokens_match=ids==ref_ids or ids==r['input_ids']
fp32=s['precision']=='float32';minimum=.99999 if fp32 else .999;maximum=.0001 if fp32 else .01
result={'case':r['case'],'precision':s['precision'],'dimensions':dim,'minimum_cosine':float(cos.min()),'maximum_absolute_error':float(error),'maximum_norm_error':float(norm_error),'tokenization_match':tokens_match,'thresholds':{'minimum_cosine':minimum,'maximum_absolute_error':maximum,'maximum_norm_error':1e-5},'pass':bool(cos.min()>=minimum and error<=maximum and norm_error<=1e-5 and tokens_match),'scope':'numerical parity on fixed inputs, not a retrieval benchmark'}
a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result));raise SystemExit(0 if result['pass'] else 1)
