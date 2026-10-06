"""Download pinned public Google weights; verify known LFS SHA before use."""
import argparse,hashlib,json,urllib.request
from pathlib import Path
REVISION='914f7f89142e33e77833254d9c9b90c3cef7303b'
ROOT=Path(__file__).resolve().parents[1]
ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,default=ROOT/'models/embeddinggemma2/original');args=ap.parse_args()
out=args.output;out.mkdir(parents=True,exist_ok=True)
known={'config.json': 'b8f1e9931b57fbc054acdb445c41765d55b0074c58d145fa82839941ad1b5bb3', 'tokenizer.json': '4d777ef5bdc1aa36227abdfb77c3e49e7b9c892d16e1b6bda41c393504828be4', 'tokenizer_config.json': '17bd5d6e9364ca49a534e1502076593317c298d4a663623091ed45388f004874', 'config_sentence_transformers.json': '031e56a498d33c349ab489a21885bcfe25b4fcba841149dc99e1e90d4a7c28f5', 'README.md': 'b677a0ee2818c36f091b65fd3d9b4baea9f04d53b02b49cae916c74bf9682037', 'model.safetensors': '197a32965d4b1105faf060417baa899e193fb73cd401f42ec9295234d5553d79'}
manifest={'repo':'google/embeddinggemma-2','revision':REVISION,'files':{}}
for name in ['config.json','tokenizer.json','tokenizer_config.json','config_sentence_transformers.json','README.md','model.safetensors']:
 target=out/name
 if not target.exists():
  partial=target.with_suffix(target.suffix+'.partial')
  with urllib.request.urlopen(f'https://huggingface.co/google/embeddinggemma-2/resolve/{REVISION}/{name}',timeout=90) as response,partial.open('wb') as f:
   while b:=response.read(4*1024*1024):f.write(b)
  partial.rename(target)
 sha=hashlib.file_digest(target.open('rb'),'sha256').hexdigest()
 if name in known and sha!=known[name]:raise ValueError(f'SHA mismatch: {name}')
 manifest['files'][name]={'sha256':sha,'bytes':target.stat().st_size}
 print(name,target.stat().st_size,sha,flush=True)
(out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
