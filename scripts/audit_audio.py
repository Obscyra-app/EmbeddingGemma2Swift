"""Independent NumPy audit against pinned official Transformers audio references."""
import argparse,json
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('reference',type=Path);p.add_argument('actual',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
r=json.loads(a.reference.read_text());s=json.loads(a.actual.read_text());pre=s['audio'][0] if 'audio' in s else s
metadata=all(pre[k]==r[k] for k in ['sample_count','frames','valid_frames','soft_tokens'])
result={'case':r['case'],'metadata_match':metadata}
if 'embeddings' in s:
 dim=s['dimensions'];x=np.array(s['embeddings']);y=np.array(r['vectors'][str(dim)])
 assert x.shape==y.shape and np.isfinite(x).all()
 tokens=s['input_ids']==r['input_ids'];cos=np.sum(x*y,axis=-1)/(np.linalg.norm(x,axis=-1)*np.linalg.norm(y,axis=-1));err=float(abs(x-y).max());norm=float(abs(np.linalg.norm(x,axis=-1)-1).max());fp32=s['precision']=='float32'
 passed=bool(metadata and tokens and cos.min()>=(.99999 if fp32 else .999) and err<=(.0001 if fp32 else .01) and norm<=1e-5)
 result.update(precision=s['precision'],dimensions=dim,minimum_cosine=float(cos.min()),maximum_absolute_error=err,maximum_norm_error=norm,tokenization_match=tokens)
else:
 passed=bool(metadata and s['mask_match'] and s['feature_max_error']<=2e-5)
 result.update(mask_match=s['mask_match'],feature_max_error=s['feature_max_error'])
result['pass']=passed;a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result));raise SystemExit(0 if passed else 1)
