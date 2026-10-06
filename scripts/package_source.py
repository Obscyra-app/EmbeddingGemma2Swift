"""Create an explicit source-only review archive. Does not upload or publish."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

p = argparse.ArgumentParser()
p.add_argument('--output',type=Path,required=True)
a = p.parse_args()
root = Path(__file__).resolve().parents[1]
files = ['Package.swift','Package.resolved','README.md','VALIDATION.md','CHANGELOG.md','LICENSE','NOTICE',
         'APACHE-2.0.txt','UPSTREAM_MLX_VLM_LICENSE','UPSTREAM_PILLOW_LICENSE','.gitignore',
         'requirements-reference.lock','Examples/iOS/DemoApp.swift',
         'Examples/iOS/project.yml','Examples/iOS/README.md','Examples/iOS/.gitignore']
for directory, suffix in [('Sources','.swift'),('Tests','.swift'),('scripts','.py'),('Validation','.json')]:
    files += [str(f.relative_to(root)) for f in sorted((root/directory).rglob('*'+suffix))]
manifest = {}
with zipfile.ZipFile(a.output,'x',compression=zipfile.ZIP_DEFLATED) as archive:
    for name in files:
        path = root/name
        if path.is_symlink() or not path.is_file():
            raise ValueError(f'Not a regular source file: {name}')
        data = path.read_bytes()
        manifest[name] = {'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}
        archive.writestr('EmbeddingGemma2Swift/'+name,data)
    archive.writestr('EmbeddingGemma2Swift/SOURCE_MANIFEST.json',json.dumps(manifest,indent=2)+'\n')
print(json.dumps({'source_files':len(files),'archive_bytes':a.output.stat().st_size,'sha256':hashlib.file_digest(a.output.open('rb'),'sha256').hexdigest()}))
