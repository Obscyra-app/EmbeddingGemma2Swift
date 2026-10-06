"""Reject incompatible packed metadata without changing any checkpoint."""
import argparse,json,subprocess,tempfile
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--executable',type=Path,required=True);p.add_argument('--model',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();checks=[]
for case in ['bits','shape','missing-scales','missing-metadata']:
 with tempfile.TemporaryDirectory() as temp:
  d=Path(temp)
  for name in ['model.safetensors','config.json','tokenizer.json','tokenizer_config.json']:(d/name).symlink_to((a.model/name).resolve())
  q=json.loads((a.model/'quantization.json').read_text());key=next(iter(q['shapes']))
  if case=='bits':q['bits']=3
  elif case=='shape':q['shapes'][key][0]+=1
  elif case=='missing-scales':q['shapes'].pop(key)
  if case!='missing-metadata':(d/'quantization.json').write_text(json.dumps(q))
  request=d/'request.json';request.write_text(json.dumps({'texts':['test']}))
  result=subprocess.run([str(a.executable.resolve()),str(d),str(request)],capture_output=True,text=True,timeout=30)
  assert result.returncode==1 and 'Error:' in result.stderr,(case,result.returncode,result.stderr)
  checks.append({'case':case,'pass':True,'expected_rejection':True})
a.output.write_text(json.dumps(checks,indent=2)+'\n');print(json.dumps(checks))
