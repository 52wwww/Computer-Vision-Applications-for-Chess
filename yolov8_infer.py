from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os

PADDING = 150  # 例：训练时每边加了150像素padding，按实际改

# 加载模型（假设你已训练好yolov8模型，模型文件名为best.pt，放在数据集文件夹下）
model = YOLO('rectified_dataset/runs/detect/train20/weights/best.pt')

def detect_chess(image_path):
    results = model(image_path)
    boxes = results[0].boxes.xyxy.cpu().numpy()
    classes = results[0].boxes.cls.cpu().numpy()
    names = results[0].names
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names.keys(), key=lambda x: int(x))]
    img = cv2.imread(image_path)
    h, w = img.shape[:2]
    grid_w, grid_h = w // 8, h // 8

    # 只画检测框和类别标签
    for box, cls in zip(boxes, classes):
        x1, y1, x2, y2 = map(int, box)
        class_name = names[int(cls)]
        color = (0, 255, 0)
        if 'black' in class_name:
            color = (0, 0, 255)
        elif 'white' in class_name:
            color = (255, 0, 0)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, class_name, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    output_path = 'static/detection_result.jpg'
    cv2.imwrite(output_path, img)
    board_detected = False
    if boxes is not None and len(boxes) > 0:
        for i in range(len(boxes)):
            if 'board' in names[int(classes[i])].lower():
                board_detected = True
                break
        if not board_detected:
            board_detected = True
    return boxes, classes, names, output_path, board_detected

def get_piece_positions(boxes, classes, names, image_shape, padding=PADDING):
    if len(boxes) == 0:
        return []
    h, w = image_shape[:2]
    # 去除padding后的棋盘区域
    x1, y1 = padding, padding
    x2, y2 = w - padding, h - padding
    grid_w = (x2 - x1) / 8
    grid_h = (y2 - y1) / 8
    positions = []
    coord_map = {}
    col_labels = ['A','B','C','D','E','F','G','H']
    for idx, (box, cls) in enumerate(zip(boxes, classes)):
        # 下边两个角
        left_x, left_y = box[0], box[3]
        right_x, right_y = box[2], box[3]
        # 坐标先减去padding
        left_x, left_y = left_x - x1, left_y - y1
        right_x, right_y = right_x - x1, right_y - y1
        left_col = int(left_x / grid_w)
        left_row = int(left_y / grid_h)
        right_col = int(right_x / grid_w)
        right_row = int(right_y / grid_h)
        # clip到[0,7]
        left_col = max(0, min(left_col, 7))
        left_row = max(0, min(left_row, 7))
        right_col = max(0, min(right_col, 7))
        right_row = max(0, min(right_row, 7))
        if (left_col == right_col) and (left_row == right_row):
            row, col = left_row, left_col
        else:
            # 用下边线中心
            center_x = (left_x + right_x) / 2
            center_y = (left_y + right_y) / 2
            col = int(center_x / grid_w)
            row = int(center_y / grid_h)
            col = max(0, min(col, 7))
            row = max(0, min(row, 7))
        chess_coord = f"{col_labels[col]}{8-row}"
        if chess_coord not in coord_map:
            positions.append({'piece': names[int(cls)], 'row': row, 'col': col, 'chess_coord': chess_coord})
            coord_map[chess_coord] = len(positions) - 1
    return positions

def draw_chessboard_with_pieces(positions, output_path='static/chessboard_result.jpg'):
    board_size = 480
    cell_size = board_size // 8
    board_img = Image.new("RGB", (board_size, board_size), (255, 255, 255))
    draw = ImageDraw.Draw(board_img)
    for row in range(8):
        for col in range(8):
            color = (240, 217, 181) if (row + col) % 2 == 0 else (181, 136, 99)
            draw.rectangle([col*cell_size, row*cell_size, (col+1)*cell_size, (row+1)*cell_size], fill=color)
    # 国际象棋符号映射
    piece_map = {
        "white-king": "♔", "white-queen": "♕", "white-rook": "♖", "white-bishop": "♗", "white-knight": "♘", "white-pawn": "♙",
        "black-king": "♚", "black-queen": "♛", "black-rook": "♜", "black-bishop": "♝", "black-knight": "♞", "black-pawn": "♟",
        "K": "♔", "Q": "♕", "R": "♖", "B": "♗", "N": "♘", "P": "♙",
        "k": "♚", "q": "♛", "r": "♜", "b": "♝", "n": "♞", "p": "♟",
    }
    # 构建8x8棋盘，填入棋子符号
    board = [[None for _ in range(8)] for _ in range(8)]
    print(positions)
    for p in positions:
        r, c = p['row'], p['col']
        piece = piece_map.get(p.get('piece', ''), p.get('piece', '?'))
        board[r][c] = piece
    for row in board:
        print(row)
    # 字体路径兼容
    font_path = "DejaVuSans.ttf"
    if not os.path.exists(font_path):
        font_path = "C:/Windows/Fonts/arialuni.ttf"
    try:
        font = ImageFont.truetype(font_path, 56)
    except Exception:
        font = ImageFont.load_default()
    for row in range(8):
        for col in range(8):
            piece = board[row][col]
            if piece:
                try:
                    bbox = draw.textbbox((0, 0), piece, font=font)
                    w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
                except AttributeError:
                    w, h = font.getsize(piece)
                x = col*cell_size + (cell_size-w)//2
                y = row*cell_size + (cell_size-h)//2
                draw.text((x, y-10), piece, fill=(0,0,0), font=font)
    board_img.save(output_path)
    return output_path

def get_fen_from_positions(positions):
    # 构建8x8棋盘，空格为None
    board = [[None for _ in range(8)] for _ in range(8)]
    piece_map = {
        "white-king": "K", "white-queen": "Q", "white-rook": "R", "white-bishop": "B", "white-knight": "N", "white-pawn": "P",
        "black-king": "k", "black-queen": "q", "black-rook": "r", "black-bishop": "b", "black-knight": "n", "black-pawn": "p",
        "K": "K", "Q": "Q", "R": "R", "B": "B", "N": "N", "P": "P",
        "k": "k", "q": "q", "r": "r", "b": "b", "n": "n", "p": "p",
    }
    for p in positions:
        r, c = p['row'], p['col']
        piece = piece_map.get(p.get('piece', ''), None)
        if piece:
            board[r][c] = piece
    fen_rows = []
    for row in board:
        fen_row = ''
        empty = 0
        for cell in row:
            if cell:
                if empty > 0:
                    fen_row += str(empty)
                    empty = 0
                fen_row += cell
            else:
                empty += 1
        if empty > 0:
            fen_row += str(empty)
        fen_rows.append(fen_row)
    fen = '/'.join(fen_rows)
    fen += ' w - - 0 1'  # 默认白方走棋，无易位、无吃过路
    return fen

def highlight_move_on_chessboard(img_path, move_uci, output_path='static/chessboard_result.jpg'):
    from PIL import Image, ImageDraw
    img = Image.open(img_path).convert('RGB')
    draw = ImageDraw.Draw(img)
    board_size = img.size[0]
    cell_size = board_size // 8
    # 解析uci走法，如e2e4
    if len(move_uci) < 4:
        img.save(output_path)
        return output_path
    col_labels = 'abcdefgh'
    try:
        from_sq = move_uci[:2]
        to_sq = move_uci[2:4]
        from_col = col_labels.index(from_sq[0])
        from_row = 8 - int(from_sq[1])
        to_col = col_labels.index(to_sq[0])
        to_row = 8 - int(to_sq[1])
        # 高亮起点终点格子
        for (row, col), color in [((from_row, from_col), (255, 0, 0)), ((to_row, to_col), (0, 200, 255))]:
            x1, y1 = col * cell_size, row * cell_size
            x2, y2 = (col + 1) * cell_size - 1, (row + 1) * cell_size - 1
            draw.rectangle([x1, y1, x2, y2], outline=color, width=6)
    except Exception:
        pass
    img.save(output_path)
    return output_path