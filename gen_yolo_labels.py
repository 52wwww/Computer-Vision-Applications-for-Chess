import os
import json
from PIL import Image

piece_map = {'K':0,'Q':1,'R':2,'B':3,'N':4,'P':5,'k':6,'q':7,'r':8,'b':9,'n':10,'p':11}

def convert(img_dir, label_dir):
    imgs = [f for f in os.listdir(img_dir) if f.endswith('.png')]
    for img in imgs:
        json_path = os.path.join('test/test', img.replace('.png', '.json'))
        if not os.path.exists(json_path):
            continue
        with open(json_path, encoding='utf-8') as f:
            data = json.load(f)
        w, h = Image.open(os.path.join(img_dir, img)).size
        lines = []
        for p in data['pieces']:
            c = piece_map.get(p['piece'])
            if c is None: continue
            x, y, bw, bh = p['box']
            xc = (x + bw/2) / w
            yc = (y + bh/2) / h
            bw_ = bw / w
            bh_ = bh / h
            lines.append(f'{c} {xc:.6f} {yc:.6f} {bw_:.6f} {bh_:.6f}')
        with open(os.path.join(label_dir, img.replace('.png', '.txt')), 'w', encoding='utf-8') as f2:
            f2.write('\n'.join(lines))

convert('test/images/train', 'test/labels/train')
convert('test/images/val', 'test/labels/val')
print('YOLO标签已全部生成！')