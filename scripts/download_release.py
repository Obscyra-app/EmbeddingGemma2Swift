"""Explicit public checkpoint download; standard library only, pinned SHA256."""
import argparse,hashlib,json,os,urllib.request,urllib.error
from pathlib import Path
REPOSITORY = "Nurymanau/EmbeddingGemma-2-MLX-Swift"
REVISION = "06b4e075e6b3772e3c29df9f2a2ace7bd4950bdd"
FILES = {'text-q8/model.safetensors': {'bytes': 288118885, 'sha256': 'eb8a067456a3cff750bebadc7b985420444e673508a2da5bcf56e726e52d62c9'}, 'text-q8/config.json': {'bytes': 2359, 'sha256': 'e090bf087c627155c55b481fca9d216461bdaf44d5a980e018ed3deb92fc7664'}, 'text-q8/tokenizer.json': {'bytes': 32170510, 'sha256': '4d777ef5bdc1aa36227abdfb77c3e49e7b9c892d16e1b6bda41c393504828be4'}, 'text-q8/tokenizer_config.json': {'bytes': 1599, 'sha256': '17bd5d6e9364ca49a534e1502076593317c298d4a663623091ed45388f004874'}, 'text-q8/config_sentence_transformers.json': {'bytes': 1565, 'sha256': '031e56a498d33c349ab489a21885bcfe25b4fcba841149dc99e1e90d4a7c28f5'}, 'text-q8/quantization.json': {'bytes': 16288, 'sha256': '9de7e788a89f36a07387c9758dd187b596675bfe9e0dbf9e56c3a09f1f410965'}, 'text-q8/manifest.json': {'bytes': 379, 'sha256': 'd9cdb30351293694b7343078970c72b3696eb12034ff06bff2af663fdc3c5cd8'}, 'vision-q8/vision.safetensors': {'bytes': 193095171, 'sha256': '9b18ee38272989696737fe3a0c4302dfa06f47cc37192b1b8f7583f8447da173'}, 'vision-q8/config.json': {'bytes': 3453, 'sha256': '6ece510cf4df3d5370011ea0e856486f971dbd8019f5587c6a059977d61571d4'}, 'vision-q8/quantization.json': {'bytes': 11258, 'sha256': '19c4f4d59a942f0829b4c5b6d8da12e24c5a5306434a22f0705e10ff2ced1186'}, 'vision-q8/manifest.json': {'bytes': 380, 'sha256': 'db92014f981b36cde052bc4759929d343d30b22f7888c623375b50427ffa11a4'}, 'audio-bf16/audio.safetensors': {'bytes': 611314600, 'sha256': 'a163c6dc2e47cb894a02fb7f8e6e4d7cc4edbd81a80a8506438fd42610a526e3'}, 'audio-bf16/config.json': {'bytes': 3361, 'sha256': '500326bb2a9b4372db5fa505298f046cff27a5f4961c02102a5f8bccdebe6eac'}, 'audio-bf16/manifest.json': {'bytes': 196, 'sha256': '39cb724fbd90a0b1b99ea64a643556e73fd18444f4b01a9b9fb835fdabd3ca5e'}}

def sha(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--audio',action='store_true',help='Also download the optional BF16 audio tower')
    p.add_argument('--text-only',action='store_true',help='Skip vision weights')
    p.add_argument('--verify-only',action='store_true',help='Verify existing files without network requests')
    a=p.parse_args();folders={'text-q8'}
    if not a.text_only:folders.add('vision-q8')
    if a.audio:folders.add('audio-bf16')
    for name,expected in FILES.items():
        if name.split('/')[0] not in folders:continue
        target=a.output/name
        if target.exists():
            if target.stat().st_size!=expected['bytes'] or sha(target)!=expected['sha256']:
                raise RuntimeError(f'Existing file differs from release: {name}; use a new output directory')
            print(name,'verified',flush=True);continue
        if a.verify_only:raise RuntimeError(f'Missing release file: {name}')
        target.parent.mkdir(parents=True,exist_ok=True);partial=target.with_name(target.name+'.download')
        try:
            request=urllib.request.Request(f'https://huggingface.co/{REPOSITORY}/resolve/{REVISION}/{name}',headers={'User-Agent':'EmbeddingGemma2Swift/0.1.0'})
            with urllib.request.urlopen(request,timeout=120) as response,partial.open('wb') as f:
                while chunk:=response.read(4*1024*1024):f.write(chunk)
            if partial.stat().st_size!=expected['bytes'] or sha(partial)!=expected['sha256']:raise RuntimeError(f'Checksum mismatch: {name}')
            os.replace(partial,target)
        except urllib.error.HTTPError as e:raise RuntimeError(f'Download failed for {name}: HTTP {e.code}') from None
        finally:
            if partial.exists():partial.unlink()
        print(name,'downloaded and verified',flush=True)
    print('Release ready:',', '.join(sorted(folders)))
if __name__=='__main__':main()
