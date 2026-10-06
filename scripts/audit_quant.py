"""Compare quantized outputs with BF16; retrieval quality is separate from kernel parity."""
import argparse,json
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--baseline',type=Path,required=True);p.add_argument('--actual',type=Path,required=True);p.add_argument('--dataset',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
b=json.loads(a.baseline.read_text());s=json.loads(a.actual.read_text());x=np.array(b['embeddings']);y=np.array(s['embeddings'])
assert x.shape==y.shape and np.isfinite(y).all() and b['input_ids']==s['input_ids']
norm=np.linalg.norm(y,axis=-1);assert max(abs(norm-1))<=1e-5
cos=(x*y).sum(-1)/(np.linalg.norm(x,axis=-1)*norm)
r={'baseline_weight_format':b.get('text_weight_format','unquantized'),'actual_weight_format':s['text_weight_format'],'minimum_cosine':float(cos.min()),'mean_cosine':float(cos.mean()),'maximum_absolute_error':float(abs(x-y).max()),'maximum_norm_error':float(max(abs(norm-1))),'tokenization_match':True}
if a.dataset:
 d=json.loads(a.dataset.read_text());n=len(d['documents']);truth=np.array([q['relevant'] for q in d['queries']]);bs=x[n:]@x[:n].T;qs=y[n:]@y[:n].T;bp=bs.argmax(1);qp=qs.argmax(1)
 lost=int(((bp==truth)&(qp!=truth)).sum());r.update(split=d['split'],queries=len(truth),baseline_correct=int((bp==truth).sum()),actual_correct=int((qp==truth).sum()),lost_correct=lost,changed_top1=int((bp!=qp).sum()),maximum_score_error=float(abs(bs-qs).max()),baseline_top1=bp.tolist(),actual_top1=qp.tolist(),dev_threshold_pass=bool(cos.min()>=.995 and lost<=1))
a.output.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r))
