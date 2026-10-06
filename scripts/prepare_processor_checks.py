"""Non-personal decoder/resize fixtures independent of model weights."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from PIL import Image,ImageOps,ImageDraw
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
from transformers.models.gemma4.image_processing_pil_gemma4 import Gemma4ImageProcessorPil,get_aspect_ratio_preserving_size
from safetensors.torch import save_file
import torch
base=np.arange(29*17*3,dtype=np.uint32).reshape(17,29,3).astype(np.uint8)
checks=[]
for orientation in range(1,9):
 name=f'exif{orientation}';im=Image.fromarray(base);exif=Image.Exif();exif[274]=orientation;path=a.output/f'{name}.png';im.save(path,exif=exif)
 checks.append((name,path))
gray=a.output/'gray.png';Image.fromarray(base[:,:,0]).save(gray);checks.append(('gray',gray))
# Lossless sRGB/opaque PNG is the exact decode contract. JPEG may vary by platform decoder.
for name,path in checks:
 im=ImageOps.exif_transpose(Image.open(path)).convert('RGB')
 inputs=Gemma4ImageProcessorPil(max_soft_tokens=70)(im,return_tensors='pt')
 h,w=get_aspect_ratio_preserving_size(im.height,im.width,16,630,3)
 save_file({'pixels':inputs['pixel_values'],'positions':inputs['image_position_ids'].to(torch.int32)},str(a.output/f'{name}-pixels.safetensors'))
 ref={'case':name,'resized_size':[w,h],'soft_tokens':int(inputs['num_soft_tokens_per_image'][0]),'resized_rgb_sha256':hashlib.sha256(im.resize((w,h),Image.Resampling.BICUBIC).tobytes()).hexdigest()}
 (a.output/f'{name}-reference.json').write_text(json.dumps(ref))
 req={'images':[str(path.resolve())],'max_soft_tokens':70,'pixel_reference':str((a.output/f'{name}-pixels.safetensors').resolve())}
 (a.output/f'{name}-request.json').write_text(json.dumps(req))
transparent=a.output/'transparent.png';Image.new('RGBA',(8,8),(100,20,30,0)).save(transparent)
(a.output/'transparent-request.json').write_text(json.dumps({'images':[str(transparent.resolve())]}))
for name,color,kind in [('red-circle',(240,20,20),'circle'),('blue-square',(20,40,240),'square'),('green-triangle',(20,180,40),'triangle')]:
 im=Image.new('RGB',(256,256),'white');draw=ImageDraw.Draw(im)
 if kind=='circle':draw.ellipse((32,32,224,224),fill=color)
 elif kind=='square':draw.rectangle((40,40,216,216),fill=color)
 else:draw.polygon([(128,24),(232,232),(24,232)],fill=color)
 im.save(a.output/f'{name}.png')
print('Prepared 9 decoder cases, transparency rejection and 3 shape search smoke fixtures')
