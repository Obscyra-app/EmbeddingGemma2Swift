"""Independent NumPy unpack and upstream CPU FP32 forward of packed checkpoints."""
import argparse,json,gc
from pathlib import Path
import numpy as np
import torch
from safetensors import safe_open
from transformers import AutoTokenizer,EmbeddingGemma2Config,EmbeddingGemma2Model
from transformers.models.embedding_gemma2.modeling_embedding_gemma2 import EmbeddingGemma2RotaryEmbedding
from transformers.models.gemma4.modeling_gemma4 import Gemma4VisionRotaryEmbedding
p=argparse.ArgumentParser();p.add_argument('--text',type=Path,required=True);p.add_argument('--vision',type=Path,required=True);p.add_argument('--reference',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
torch.set_num_threads(2);weights={}
for directory,filename,prefix in [(a.text,'model.safetensors','language_model.'),(a.vision,'vision.safetensors','')]:
 q=json.loads((directory/'quantization.json').read_text());bits=q['bits'];group=q['groupSize'];shapes={prefix+k:v for k,v in q['shapes'].items()}
 with safe_open(directory/filename,framework='pt',device='cpu') as f:
  for key in f.keys():
   if key.endswith(('.scales','.biases')):continue
   if key not in shapes:weights[key]=f.get_tensor(key).float();continue
   rows,cols=shapes[key];packed=f.get_tensor(key).numpy();scales=f.get_tensor(key+'.scales').float().numpy();biases=f.get_tensor(key+'.biases').float().numpy();result=np.empty((rows,cols),dtype=np.float32)
   for start in range(0,rows,2048):
    end=min(start+2048,rows)
    integers=((packed[start:end,:,None]>>np.arange(0,32,bits,dtype=np.uint32))&((1<<bits)-1)).reshape(end-start,cols).astype(np.float32)
    result[start:end]=integers*np.repeat(scales[start:end],group,axis=1)+np.repeat(biases[start:end],group,axis=1)
   weights[key]=torch.from_numpy(result)
config=json.loads((a.vision/'config.json').read_text());config['audio_config']=None
with torch.device('meta'):model=EmbeddingGemma2Model(EmbeddingGemma2Config(**config))
model.load_state_dict(weights,strict=True,assign=True);del weights;gc.collect()
model.language_model.rotary_emb=EmbeddingGemma2RotaryEmbedding(model.config.text_config)
model.language_model.embed_tokens.embed_scale=torch.tensor(model.config.text_config.hidden_size**.5)
model.vision_tower.encoder.rotary_emb=Gemma4VisionRotaryEmbedding(model.config.vision_config)
assert not any(x.is_meta for x in list(model.parameters())+list(model.buffers()))
model.eval();model.set_attn_implementation('sdpa')
for case in ['multilingual','landscape70']:
 ref=json.loads((a.reference/('vision' if case=='landscape70' else '')/f'{case}-reference.json').read_text())
 inputs={'input_ids':torch.tensor(ref['input_ids']),'attention_mask':torch.tensor(ref['attention_mask'])}
 if case=='landscape70':
  with safe_open(a.reference/'vision'/f'{case}-pixels.safetensors',framework='pt',device='cpu') as f:inputs.update(pixel_values=f.get_tensor('pixels'),image_position_ids=f.get_tensor('positions').long())
 with torch.inference_mode():
  x=model(**inputs).last_hidden_state.float();mask=inputs['attention_mask'][...,None];pooled=(x*mask).sum(1)/mask.sum(1)
  ref['vectors']={str(d):torch.nn.functional.normalize(pooled[:,:d],dim=-1).tolist() for d in [128,256,512,768]}
 ref.update(reference_kind='independent numpy affine unpack -> official torch CPU FP32',bits=bits)
 (a.output/f'{case}-reference.json').write_text(json.dumps(ref,indent=2)+'\n');print(case,'reference saved',flush=True)
