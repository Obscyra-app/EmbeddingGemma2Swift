"""Audit a complete iOS validation export, including both text and images."""
import argparse,json,subprocess,sys,tempfile
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('actual',type=Path);p.add_argument('--text-reference',type=Path,required=True);p.add_argument('--vision-reference',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
d=json.loads(a.actual.read_text());scripts=Path(__file__).resolve().parent
assert d['platform']=='iOS' and d['precision']=='bfloat16'
assert [c['case'] for c in d['vision_checks']]==['landscape70','portrait140','square280','downscale70']
with tempfile.TemporaryDirectory() as tmp:
 t=Path(tmp);subprocess.run([sys.executable,str(scripts/'audit_ios.py'),str(a.actual),'--reference',str(a.text_reference),'--output',str(t/'text.json')],check=True,capture_output=True)
 checks=[]
 for c in d['vision_checks']:
  actual=t/'actual.json';audit=t/'audit.json';actual.write_text(json.dumps(c))
  subprocess.run([sys.executable,str(scripts/'audit_vision.py'),str(a.vision_reference/f"{c['case']}-reference.json"),str(actual),'--output',str(audit)],check=True,capture_output=True)
  check=json.loads(audit.read_text());check['seconds']=c['seconds'];checks.append(check)
 result={k:d[k] for k in ['platform','os','load_seconds','vision_load_seconds','memory','mlx_cache_limit_bytes']}
 result.update(checks=checks,text_audit=json.loads((t/'text.json').read_text()),pass_=True)
 result['pass']=result.pop('pass_');a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
