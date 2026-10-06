"""Check selected packed iOS vectors against official reference and same-format Mac."""
import argparse,json,subprocess,sys,tempfile
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('actual',type=Path);p.add_argument('--reference',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();d=json.loads(a.actual.read_text());scripts=Path(__file__).resolve().parent
assert d['text_weight_format']=='affine8-g64'
with tempfile.TemporaryDirectory() as tmp:
 t=Path(tmp)
 subprocess.run([sys.executable,str(scripts/'audit_audio_device.py'),str(a.actual),'--reference',str(a.reference),'--output',str(t/'original.json')],check=True,capture_output=True)
 checks=[]
 for c in d['checks']+d['vision_checks']+d['audio_checks']:
  name=c['case'];mac=json.loads((a.reference/'quant'/f'parity-{name}-q8/stdout.json').read_text());x=np.array(c['embeddings']);y=np.array(mac['embeddings']);assert x.shape==y.shape and np.isfinite(x).all()
  error=float(abs(x-y).max());cos=float(((x*y).sum(-1)/(np.linalg.norm(x,axis=-1)*np.linalg.norm(y,axis=-1))).min());assert cos>=.999 and error<=.01
  checks.append({'case':name,'mac_q8_maximum_absolute_difference':error,'mac_q8_minimum_cosine':cos,'seconds':c['seconds'],'pass':True})
 original=json.loads((t/'original.json').read_text())
 # The older audio audit's Mac comparison describes its unquantized baseline, not a q8 match.
 for c in original['audio_checks']:c['unquantized_mac_bf16_maximum_absolute_difference']=c.pop('mac_bf16_maximum_absolute_difference')
 result={'pass':True,'text_weight_format':d['text_weight_format'],'checks':checks,'official_reference_audit':original,'memory':d['memory'],'load_seconds':d['load_seconds'],'vision_load_seconds':d['vision_load_seconds'],'audio_load_seconds':d['audio_load_seconds']}
 a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'pass':True,'checks':checks,'memory':d['memory']}))
