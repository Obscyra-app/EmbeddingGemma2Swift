"""Extract exact text weights and produce independent upstream PyTorch FP32 fixtures."""
import argparse,hashlib,json,shutil,time,importlib.metadata
from pathlib import Path
import torch
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import AutoTokenizer,EmbeddingGemma2TextConfig,EmbeddingGemma2TextModel
ROOT=Path(__file__).resolve().parents[1]
ap=argparse.ArgumentParser();ap.add_argument('--source',type=Path,default=ROOT/'models/embeddinggemma2/original');ap.add_argument('--output',type=Path,default=ROOT/'models/embeddinggemma2/text-bf16');ap.add_argument('--results',type=Path,default=ROOT/'runs/embeddinggemma2-swift');ap.add_argument('--extract-only',action='store_true');args=ap.parse_args()
source=args.source;out=args.output;results=args.results;out.mkdir(parents=True,exist_ok=True);results.mkdir(parents=True,exist_ok=True)
expected='197a32965d4b1105faf060417baa899e193fb73cd401f42ec9295234d5553d79'
assert hashlib.file_digest((source/'model.safetensors').open('rb'),'sha256').hexdigest()==expected,'Source weights differ from pinned Google checkpoint'
torch.set_num_threads(2)
config=json.loads((source/'config.json').read_text())
weights={}
with safe_open(source/'model.safetensors',framework='pt',device='cpu') as f:
 for key in f.keys():
  clean=key.removeprefix('model.')
  if clean.startswith('language_model.'):
   weights[clean]=f.get_tensor(key)
save_file(weights,str(out/'model.safetensors'))
for name in ['tokenizer.json','tokenizer_config.json','config_sentence_transformers.json']:
 shutil.copy2(source/name,out/name)
config['vision_config']=None;config['audio_config']=None
(out/'config.json').write_text(json.dumps(config,indent=2)+'\n')
manifest={'source':json.loads((source/'manifest.json').read_text()),'parameters':sum(v.numel() for v in weights.values()),'tensor_count':len(weights),'files':{}}
for path in out.iterdir():
 if path.is_file() and path.name!='manifest.json':manifest['files'][path.name]={'sha256':hashlib.file_digest(path.open('rb'),'sha256').hexdigest(),'bytes':path.stat().st_size}
(out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print('extracted',manifest['parameters'],len(weights),flush=True)
if args.extract_only:raise SystemExit(0)
model=EmbeddingGemma2TextModel(EmbeddingGemma2TextConfig(**config['text_config'])).float().eval()
model.load_state_dict({k.removeprefix('language_model.'):v.float() for k,v in weights.items()},strict=True)
del weights
model.config._attn_implementation='eager'
tokenizer=AutoTokenizer.from_pretrained(source,local_files_only=True)
texts=[
 'task: search result | query: Как проверить целостность скачанного файла?',
 'title: SHA256 | text: Вычислите SHA-256 файла и сравните его с контрольной суммой источника.',
 'title: погода | text: Сегодня идёт дождь, завтра ожидается солнечная погода.',
 'task: code retrieval | query: compute a file checksum',
 'title: checksum.py | text: import hashlib\ndef checksum(path):\n    with open(path, "rb") as f:\n        return hashlib.file_digest(f, "sha256").hexdigest()',
 'task: sentence similarity | query: Café naïve — Москва 😀\n你好 world!',
 'title: none | text: ',
]
cases=[('multilingual',texts),('long',['title: long | text: '+'Проверка контекста и целостности данных. '*95])]
receipts=[]
for name,items in cases:
 inputs=tokenizer(items,padding=True,return_tensors='pt',add_special_tokens=True)
 ids=inputs['input_ids'];mask=inputs['attention_mask']
 request={'texts':items,'task':'raw','dimensions':768,'repetitions':1}
 (results/f'{name}-request.json').write_text(json.dumps(request,ensure_ascii=False,indent=2))
 (results/f'{name}-tokens.json').write_text(json.dumps({'input_ids':ids.tolist(),'attention_mask':mask.tolist(),'dimensions':768,'repetitions':1}))
 t=time.perf_counter()
 with torch.inference_mode():
  states=model(input_ids=ids,attention_mask=mask).last_hidden_state
  pooled=(states.float()*mask[:,:,None]).sum(1)/mask.sum(1)[:,None]
  vectors={str(dim):torch.nn.functional.normalize(pooled[:,:dim],dim=-1).tolist() for dim in [128,256,512,768]}
 fixture={'case':name,'input_ids':ids.tolist(),'attention_mask':mask.tolist(),'vectors':vectors,'elapsed_seconds':time.perf_counter()-t,'tokens_shape':list(states.shape),'precision':'torch-float32-cpu','versions':{k:importlib.metadata.version(k) for k in ['torch','transformers','safetensors','numpy','tokenizers']}}
 (results/f'{name}-reference.json').write_text(json.dumps(fixture,ensure_ascii=False,indent=2))
 receipts.append({k:v for k,v in fixture.items() if k not in ['vectors','input_ids','attention_mask']})
 print(name,fixture['tokens_shape'],fixture['elapsed_seconds'],flush=True)
(results/'reference-summary.json').write_text(json.dumps(receipts,indent=2)+'\n')
