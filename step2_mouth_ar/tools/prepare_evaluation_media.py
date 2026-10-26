#!/usr/bin/env python3
"""Prepare native RGB crops for CLI model/video evaluation. Does not invent GT."""
import argparse
import hashlib
import json
from pathlib import Path
import cv2


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--video',type=Path)
    p.add_argument('--fps',type=int,choices=[1,2],default=1)
    p.add_argument('--crop',type=int,nargs=4,metavar=('X','Y','W','H'))
    p.add_argument('--photos',type=Path,help='Corrections source manifest with image/labels/crop exports')
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    if not a.video and not a.photos:p.error('Provide --video and/or --photos')
    a.out.mkdir(parents=True,exist_ok=True)
    records=[]
    def write(sid,image,**meta):
        h,w=image.shape[:2]
        if w*h>4_000_000 or min(w,h)<8:raise ValueError('Crop must be 8px or larger and at most 4 megapixels')
        rgb=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        # Exclusive writes protect any earlier annotation/evaluation pack.
        with (a.out/f'{sid}.rgb').open('xb') as f:f.write(rgb.tobytes())
        cv2.imwrite(str(a.out/f'{sid}.png'),image)
        records.append(dict(id=sid,rgb=f'{sid}.rgb',image=f'{sid}.png',width=w,height=h,**meta))
    if a.photos:
        for s in json.loads(a.photos.read_text())['samples']:
            image_path=(a.photos.parent/s['image']).resolve();label_path=(a.photos.parent/s['labels']).resolve()
            image=cv2.imread(str(image_path));doc=json.loads(label_path.read_text())
            if image is None or doc['cleanImage']!=image_path.name:raise ValueError('Photo and labels differ')
            if str(doc.get('annotationStatus','')).startswith('unreviewed'):raise ValueError('Review drafts before evaluating accuracy')
            x,y,w,h=doc['crop']
            if min(x,y)<0 or min(w,h)<8 or x+w>image.shape[1] or y+h>image.shape[0]:raise ValueError('Invalid crop')
            if not s['id'] or any(not(c.isascii() and (c.isalnum() or c in '-_')) for c in s['id']):raise ValueError('Unsafe sample ID')
            write(s['id'],image[y:y+h,x:x+w],group=s['group'],split=s['split'],labels=str(label_path),
                  cleanImage=image_path.name,crop=doc['crop'],sourceSha256=hashlib.sha256(image_path.read_bytes()).hexdigest())
    if a.video:
        cap=cv2.VideoCapture(str(a.video))
        try:
            if not cap.isOpened():raise ValueError('Cannot open video')
            fps=cap.get(cv2.CAP_PROP_FPS);count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if fps<=0 or count<=0:raise ValueError('Video has no usable timebase')
            w,h=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            crop=a.crop or [0,0,w,h];x,y,cw,ch=crop
            if min(x,y)<0 or min(cw,ch)<8 or x+cw>w or y+ch>h:raise ValueError('Crop outside video')
            import math
            n=math.ceil(count/fps*a.fps)
            if n>600:raise ValueError('Maximum 600 samples; shorten clip')
            sha=hashlib.sha256(a.video.read_bytes()).hexdigest()
            for i in range(n):
                time_ms=round(i*1000/a.fps);index=min(count-1,round(time_ms*fps/1000))
                cap.set(cv2.CAP_PROP_POS_FRAMES,index);ok,frame=cap.read()
                if not ok:raise ValueError(f'Failed to decode frame {index}')
                write(f'video_{sha[:12]}_{time_ms}ms',frame[y:y+ch,x:x+cw],group=sha,split='unassigned',
                      captureProtocol='opencv-frame-index/1',cleanImage=f'video_{sha[:12]}_{time_ms}ms.png',crop=[0,0,cw,ch],timeMs=time_ms,decodedFrameIndex=index,decodedTimeMs=index*1000/fps,sourceCrop=crop,videoSha256=sha,
                      sourceVideo=str(a.video.resolve()),labels=None)
        finally:cap.release()
    (a.out/'manifest.json').write_text(json.dumps(dict(schema='evaluation-media/1',samples=records),indent=2))
    print(f'Prepared {len(records)} native crops in {a.out}')

if __name__=='__main__':main()
