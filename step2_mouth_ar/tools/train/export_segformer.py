#!/usr/bin/env python3
"""Export an already obtained, RGB dental-finetuned HF SegFormer to local ONNX.
No remote code/downloads. Requires torch, transformers, safetensors, onnx.
"""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint',type=Path,required=True,help='Local HF directory (safetensors + processor/config)')
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--tooth-classes',type=int,nargs='+',required=True)
    p.add_argument('--name',required=True)
    p.add_argument('--source-url',required=True)
    p.add_argument('--license',required=True,help='Actual checkpoint license, not merely the architecture license')
    p.add_argument('--rgb-intraoral',action='store_true',help='Confirm this checkpoint was trained for RGB intraoral photos')
    p.add_argument('--min-area',type=int,default=8)
    a=p.parse_args()
    if not a.rgb_intraoral:p.error('Do not export X-ray/3D/generic models as RGB dental detectors')
    if a.out.exists() or a.out.with_suffix('.json').exists():p.error('Output already exists')
    try:
        import torch
        from transformers import AutoImageProcessor, SegformerForSemanticSegmentation
    except ImportError as e:p.error(f'Missing export dependency: {e}')
    processor=AutoImageProcessor.from_pretrained(a.checkpoint,local_files_only=True,trust_remote_code=False)
    model=SegformerForSemanticSegmentation.from_pretrained(a.checkpoint,local_files_only=True,trust_remote_code=False,use_safetensors=True).eval()
    size=processor.size
    if set(size)!={'height','width'}:p.error('This exporter requires fixed height/width preprocessing')
    h,w=size['height'],size['width']
    if not processor.do_resize or int(processor.resample)!=2:p.error('Expected bilinear resize preprocessing')
    if not processor.do_normalize or not processor.do_rescale or abs(processor.rescale_factor-1/255)>1e-10:p.error('Unsupported processor normalization')
    if len(set(a.tooth_classes))!=len(a.tooth_classes) or any(c<0 or c>=model.config.num_labels for c in a.tooth_classes):p.error('Invalid tooth class IDs')
    if not all(isinstance(v,int) and 32<=v<=1024 for v in [w,h]) or not 1<=a.min_area<=w*h:p.error('Invalid input size or minimum area')
    class Wrapper(torch.nn.Module):
        def __init__(self):super().__init__();self.net=model
        def forward(self,pixel_values):return self.net(pixel_values=pixel_values).logits
    a.out.parent.mkdir(parents=True,exist_ok=True)
    torch.onnx.export(Wrapper(),torch.zeros(1,3,h,w),str(a.out),input_names=['pixel_values'],output_names=['logits'],opset_version=17,dynamo=False)
    card=dict(schema='dental-segformer-onnx/1',architecture='segformer',modality='rgb-intraoral',name=a.name,
        classes=model.config.num_labels,toothClasses=a.tooth_classes,minAreaPixels=a.min_area,
        input=dict(width=w,height=h,mean=processor.image_mean,std=processor.image_std),
        source=a.source_url,license=a.license,sha256=hashlib.sha256(a.out.read_bytes()).hexdigest(),
        validation='Not evaluated by export. Benchmark on independent reviewed labels before deployment.',
        instanceDecoding='Four-connected components per class. Same-class touching teeth remain merged.')
    a.out.with_suffix('.json').write_text(json.dumps(card,indent=2))
    print(a.out)
if __name__=='__main__':main()
