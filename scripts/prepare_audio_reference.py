"""Extract unchanged audio tensors and generate official CPU FP32 references."""
import argparse,json,hashlib,wave,time,gc
from pathlib import Path
import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import AutoTokenizer,EmbeddingGemma2Config,EmbeddingGemma2Model
from transformers.models.embedding_gemma2.modeling_embedding_gemma2 import EmbeddingGemma2RotaryEmbedding
from transformers.models.gemma4.modeling_gemma4 import Gemma4AudioRelPositionalEncoding
from transformers.models.gemma4.feature_extraction_gemma4 import Gemma4AudioFeatureExtractor
p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--results',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True);a.results.mkdir(parents=True,exist_ok=True)
assert hashlib.file_digest((a.source/'model.safetensors').open('rb'),'sha256').hexdigest()=='197a32965d4b1105faf060417baa899e193fb73cd401f42ec9295234d5553d79'
config=json.loads((a.source/'config.json').read_text());config['vision_config']=None
(a.output/'config.json').write_text(json.dumps(config,indent=2)+'\n');weights={}
with safe_open(a.source/'model.safetensors',framework='pt',device='cpu') as f:
 for k in f.keys():
  if k.startswith(('audio_tower.','embed_audio.')):weights[k]=f.get_tensor(k)
save_file(weights,str(a.output/'audio.safetensors'))
manifest={'revision':'914f7f89142e33e77833254d9c9b90c3cef7303b','tensor_count':len(weights),'tensor_elements':sum(x.numel() for x in weights.values()),'sha256':hashlib.file_digest((a.output/'audio.safetensors').open('rb'),'sha256').hexdigest()}
(a.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print(manifest,flush=True)
with safe_open(a.source/'model.safetensors',framework='pt',device='cpu') as f:
 for k in f.keys():
  if k.startswith('language_model.'):weights[k]=f.get_tensor(k)
torch.set_num_threads(2)
with torch.device('meta'):model=EmbeddingGemma2Model(EmbeddingGemma2Config(**config))
model.load_state_dict({k:v.float() for k,v in weights.items()},strict=True,assign=True);del weights;gc.collect()
model.language_model.rotary_emb=EmbeddingGemma2RotaryEmbedding(model.config.text_config)
model.language_model.embed_tokens.embed_scale=torch.tensor(model.config.text_config.hidden_size**.5)
model.audio_tower.rel_pos_enc=Gemma4AudioRelPositionalEncoding(model.config.audio_config)
for layer in model.audio_tower.layers:layer.self_attn.softcap=torch.tensor(model.config.audio_config.attention_logit_cap)
assert not any(x.is_meta for x in list(model.parameters())+list(model.buffers()))
model.eval();tokenizer=AutoTokenizer.from_pretrained(a.source,local_files_only=True);frontend=Gemma4AudioFeatureExtractor()
for name,n in [('silence',1600),('tone',16481),('noise',32049),('chirp',81971)]:
 t=np.arange(n)/16000
 if name=='silence':x=np.zeros(n)
 elif name=='tone':x=.3*np.sin(2*np.pi*440*t)
 elif name=='noise':x=np.random.default_rng(42).uniform(-.25,.25,n)
 else:x=.2*np.sin(2*np.pi*(150*t+250*t*t))
 pcm=np.round(x*32767).astype('<i2')
 with wave.open(str(a.results/f'{name}.wav'),'wb') as f:f.setnchannels(1);f.setsampwidth(2);f.setframerate(16000);f.writeframes(pcm.tobytes())
 samples=pcm.astype(np.float32)/32768
 inputs=frontend([samples],return_tensors='pt');mask=inputs['input_features_mask'];count=int(mask[:,::4].sum())
 text=tokenizer.boa_token+tokenizer.audio_token*count+tokenizer.eoa_token;tokens=tokenizer(text,return_tensors='pt',add_special_tokens=True)
 save_file({'features':inputs['input_features'].contiguous(),'mask':mask.to(torch.int32).contiguous()},str(a.results/f'{name}-features.safetensors'))
 start=time.perf_counter()
 with torch.inference_mode():
  states=model(**tokens,**inputs).last_hidden_state.float();pooled=states.mean(1)
  vectors={str(d):torch.nn.functional.normalize(pooled[:,:d],dim=-1).tolist() for d in [128,256,512,768]}
 result={'case':name,'sample_count':n,'frames':int(mask.shape[1]),'valid_frames':int(mask.sum()),'soft_tokens':count,'input_ids':tokens['input_ids'].tolist(),'attention_mask':tokens['attention_mask'].tolist(),'vectors':vectors,'reference_seconds':time.perf_counter()-start}
 (a.results/f'{name}-reference.json').write_text(json.dumps(result,indent=2)+'\n');print(name,result['frames'],count,result['reference_seconds'],flush=True)
