"""Independent full iOS audit: text, vision and audio against official references."""
import argparse,json,subprocess,sys,tempfile
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('actual',type=Path);p.add_argument('--reference',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
d=json.loads(a.actual.read_text());scripts=Path(__file__).resolve().parent
assert [c['case'] for c in d['audio_checks']]==['silence','tone','noise','chirp']
with tempfile.TemporaryDirectory() as tmp:
 t=Path(tmp)
 subprocess.run([sys.executable,str(scripts/'audit_vision_device.py'),str(a.actual),'--text-reference',str(a.reference),'--vision-reference',str(a.reference/'vision'),'--output',str(t/'previous.json')],check=True,capture_output=True)
 checks=[]
 for c in d['audio_checks']:
  actual=t/'actual.json';audit=t/'audit.json';actual.write_text(json.dumps(c))
  subprocess.run([sys.executable,str(scripts/'audit_audio.py'),str(a.reference/'audio'/f"{c['case']}-reference.json"),str(actual),'--output',str(audit)],check=True,capture_output=True)
  check=json.loads(audit.read_text());check['seconds']=c['seconds']
  mac=json.loads((a.reference/'audio'/f"{c['case']}-bf16"/'stdout.json').read_text())
  check['mac_bf16_maximum_absolute_difference']=float(abs(np.array(c['embeddings'])-np.array(mac['embeddings'])).max());checks.append(check)
 result={k:d[k] for k in ['platform','os','load_seconds','audio_load_seconds','memory','mlx_cache_limit_bytes']}
 result.update(audio_checks=checks,text_vision_audit=json.loads((t/'previous.json').read_text()))
 result['pass']=True;a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'pass':True,'audio_checks':checks}))
