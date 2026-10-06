"""Pinned unchanged vision extraction and independent official CPU FP32 fixtures."""
import argparse, hashlib, json, shutil, time
from pathlib import Path
import numpy as np
import torch
from PIL import Image,ImageDraw
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import AutoTokenizer,EmbeddingGemma2Config,EmbeddingGemma2Model
from transformers.models.gemma4.image_processing_pil_gemma4 import Gemma4ImageProcessorPil,get_aspect_ratio_preserving_size
p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--results',type=Path,required=True);a=p.parse_args()
a.output.mkdir(parents=True,exist_ok=True);a.results.mkdir(parents=True,exist_ok=True)
assert hashlib.file_digest((a.source/'model.safetensors').open('rb'),'sha256').hexdigest()=='197a32965d4b1105faf060417baa899e193fb73cd401f42ec9295234d5553d79'
config=json.loads((a.source/'config.json').read_text());config['audio_config']=None
(a.output/'config.json').write_text(json.dumps(config,indent=2)+'\n')
weights={}
with safe_open(a.source/'model.safetensors',framework='pt',device='cpu') as f:
 for key in f.keys():
  if key.startswith(('vision_tower.','embed_vision.')):weights[key]=f.get_tensor(key)
save_file(weights,str(a.output/'vision.safetensors'))
manifest={'original_revision':'914f7f89142e33e77833254d9c9b90c3cef7303b','parameters':sum(x.numel() for x in weights.values()),'tensors':len(weights),'sha256':hashlib.file_digest((a.output/'vision.safetensors').open('rb'),'sha256').hexdigest()}
(a.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print(manifest,flush=True)
torch.set_num_threads(2)
with torch.device('meta'):model=EmbeddingGemma2Model(EmbeddingGemma2Config(**config))
with safe_open(a.source/'model.safetensors',framework='pt',device='cpu') as f:
 for key in f.keys():
  if key.startswith('language_model.'):weights[key]=f.get_tensor(key)
model.load_state_dict({k:v.float() for k,v in weights.items()},strict=True,assign=True);del weights
# Meta construction leaves nonpersistent rotary buffers on meta; reconstruct just these buffers.
from transformers.models.embedding_gemma2.modeling_embedding_gemma2 import EmbeddingGemma2RotaryEmbedding
from transformers.models.gemma4.modeling_gemma4 import Gemma4VisionRotaryEmbedding
model.language_model.rotary_emb=EmbeddingGemma2RotaryEmbedding(model.config.text_config)
model.vision_tower.encoder.rotary_emb=Gemma4VisionRotaryEmbedding(model.config.vision_config)
model.language_model.embed_tokens.embed_scale=torch.tensor(model.config.text_config.hidden_size**0.5)
assert not any(x.is_meta for x in list(model.parameters())+list(model.buffers()))
model.eval();model.set_attn_implementation('sdpa')
tokenizer=AutoTokenizer.from_pretrained(a.source,local_files_only=True)
fixtures=[('landscape70',83,47,70),('portrait140',47,83,140),('square280',93,93,280),('downscale70',917,673,70)]
for name,w,h,budget in fixtures:
 y,x=np.mgrid[:h,:w];rgb=np.stack([(x*3+y*5)%256,(x*7+y*11)%256,(x*13+y*17)%256],-1).astype('uint8')
 im=Image.fromarray(rgb);ImageDraw.Draw(im).rectangle((w//4,h//4,3*w//4,3*h//4),fill=(240,30,20))
 im.save(a.results/f'{name}.png')
 processor=Gemma4ImageProcessorPil(max_soft_tokens=budget)
 inputs=processor(im,return_tensors='pt');n=int(inputs.pop('num_soft_tokens_per_image')[0])
 text=tokenizer.boi_token+tokenizer.image_token*n+tokenizer.eoi_token
 tokens=tokenizer(text,return_tensors='pt',add_special_tokens=True)
 th,tw=get_aspect_ratio_preserving_size(h,w,16,budget*9,3)
 resized=im.resize((tw,th),resample=Image.Resampling.BICUBIC)
 # Binary tensors allow direct preprocessing comparison without a JSON-sized pixel payload.
 save_file({'pixels':inputs['pixel_values'].contiguous(),'positions':inputs['image_position_ids'].to(torch.int32).contiguous()},str(a.results/f'{name}-pixels.safetensors'))
 t=time.perf_counter()
 with torch.inference_mode():
  out=model(**tokens,**inputs).last_hidden_state.float()
  pooled=(out*tokens['attention_mask'][...,None]).sum(1)/tokens['attention_mask'].sum(1)[:,None]
  vectors={str(d):torch.nn.functional.normalize(pooled[:,:d],dim=-1).tolist() for d in [128,256,512,768]}
 result={'case':name,'image':name+'.png','max_soft_tokens':budget,'soft_tokens':n,'resized_size':[tw,th],'resized_rgb_sha256':hashlib.sha256(resized.tobytes()).hexdigest(),'input_ids':tokens['input_ids'].tolist(),'attention_mask':tokens['attention_mask'].tolist(),'vectors':vectors,'reference_seconds':time.perf_counter()-t,'precision':'torch-float32-cpu-sdpa'}
 (a.results/f'{name}-reference.json').write_text(json.dumps(result,indent=2)+'\n');print(name,'PASS generated',result['reference_seconds'],flush=True)
