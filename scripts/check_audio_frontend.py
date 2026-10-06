"""Check supported-length edges against Google's pinned audio frontend."""
import argparse,json,subprocess,wave
from pathlib import Path
import numpy as np
import torch
from safetensors.torch import save_file
from transformers.models.gemma4.feature_extraction_gemma4 import Gemma4AudioFeatureExtractor
p=argparse.ArgumentParser();p.add_argument('--executable',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
frontend=Gemma4AudioFeatureExtractor();checks=[]
for n in [161,255,256,319,320,321,639,640,641,480000]:
 pcm=np.random.default_rng(n).integers(-1000,1001,n,dtype=np.int16);path=a.output/f'{n}.wav'
 with wave.open(str(path),'wb') as f:f.setnchannels(1);f.setsampwidth(2);f.setframerate(16000);f.writeframes(pcm.astype('<i2').tobytes())
 result=frontend([pcm.astype(np.float32)/32768],return_tensors='pt');mask=result['input_features_mask'];features=a.output/f'{n}.safetensors'
 save_file({'features':result['input_features'],'mask':mask.to(torch.int32)},str(features))
 request=a.output/f'{n}-request.json';request.write_text(json.dumps({'audio':[str(path.resolve())],'feature_reference':str(features.resolve())}))
 s=json.loads(subprocess.check_output([str(a.executable.resolve()),'.',str(request),'--audio-preprocess-only'],text=True))
 assert s['feature_max_error']<=2e-5 and s['mask_match'] and s['frames']==mask.shape[1] and s['valid_frames']==int(mask.sum()) and s['soft_tokens']==int(mask[:,::4].sum()),s
 s['pass']=True;checks.append(s)
(a.output/'summary.json').write_text(json.dumps(checks,indent=2)+'\n');print(json.dumps(checks))
