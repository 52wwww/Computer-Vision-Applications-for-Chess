import os
import json
import cv2
import numpy as np
from PIL import Image

# 类别顺序与data.yaml一致
piece_map = {'K':0,'Q':1,'R':2,'B':3,'N':4,'P':5,'k':6,'q':7,'r':8,'b':9,'n':10,'p':11}

SRC_ROOT = 'test'
DST_ROOT = 'rectified_dataset'
IMG_SIZE = 800
PADDING = 150
DST_SIZE = IMG_SIZE + 2 * PADDING

splits = ['train', 'val']

for split in splits:
    img_dir = f'{SRC_ROOT}/images/{split}'
    label_dir = f'{DST_ROOT}/labels/{split}'
    out_img_dir = f'{DST_ROOT}/images/{split}'
    os.makedirs(label_dir, exist_ok=True)
    os.makedirs(out_img_dir, exist_ok=True)
    imgs = [f for f in os.listdir(img_dir) if f.endswith('.png')]
    for img in imgs:
        json_path = os.path.join(SRC_ROOT, 'test', img.replace('.png', '.json'))
        if not os.path.exists(json_path):
            continue
        with open(json_path, encoding='utf-8') as f:
            data = json.load(f)
        corners = np.array(data['corners'], dtype=np.float32)
        dst = np.array([
            [PADDING, DST_SIZE - PADDING - 1],      # 左下
            [PADDING, PADDING],                     # 左上
            [DST_SIZE - PADDING - 1, PADDING],      # 右上
            [DST_SIZE - PADDING - 1, DST_SIZE - PADDING - 1]  # 右下
        ], dtype=np.float32)
        M = cv2.getPerspectiveTransform(corners, dst)
        # 拉正图片
        img_path = os.path.join(img_dir, img)
        img_cv = cv2.imread(img_path)
        if img_cv is None:
            continue
        warped = cv2.warpPerspective(img_cv, M, (DST_SIZE, DST_SIZE))
        cv2.imwrite(os.path.join(out_img_dir, img), warped)
        # 拉正box并生成YOLO标签
        lines = []
        for p in data['pieces']:
            c = piece_map.get(p['piece'])
            if c is None: continue
            x, y, bw, bh = p['box']
            # 四角坐标
            pts = np.array([
                [x, y],
                [x+bw, y],
                [x, y+bh],
                [x+bw, y+bh]
            ], dtype=np.float32)
            pts = cv2.perspectiveTransform(pts[None, :, :], M)[0]
            x_min, y_min = pts.min(axis=0)
            x_max, y_max = pts.max(axis=0)
            # 裁剪到图片范围
            x_min = max(0, min(x_min, DST_SIZE - 1))
            x_max = max(0, min(x_max, DST_SIZE - 1))
            y_min = max(0, min(y_min, DST_SIZE - 1))
            y_max = max(0, min(y_max, DST_SIZE - 1))
            xc = (x_min + x_max) / 2 / DST_SIZE
            yc = (y_min + y_max) / 2 / DST_SIZE
            bw_ = (x_max - x_min) / DST_SIZE
            bh_ = (y_max - y_min) / DST_SIZE
            # 只保留完全在0~1范围内的box
            if 0 <= xc <= 1 and 0 <= yc <= 1 and 0 < bw_ <= 1 and 0 < bh_ <= 1:
                lines.append(f'{c} {xc:.6f} {yc:.6f} {bw_:.6f} {bh_:.6f}')
        with open(os.path.join(label_dir, img.replace('.png', '.txt')), 'w', encoding='utf-8') as f2:
            f2.write('\n'.join(lines))
print('批量拉正完成，结果在rectified_dataset目录！') 