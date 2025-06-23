from flask import Flask, render_template, request
from yolov8_infer import detect_chess, get_piece_positions, draw_chessboard_with_pieces, get_fen_from_positions, highlight_move_on_chessboard
import cv2
import chess
import chess.engine

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        error_message = None
        results = {
            'board_detected': False,
            'pieces_detected': False,
            'positions_calculated': False,
            'piece_count': 0
        }
        best_move_white = None
        best_move_black = None
        chessboard_path = None
        try:
            file = request.files['image']
            if not file:
                error_message = '未选择图片文件'
                return render_template('index.html', error=error_message, image_url=None, chessboard_url=None, results=results, best_move_white=None, best_move_black=None)
            path = 'static/upload.jpg'
            file.save(path)
            boxes, classes, names, detection_path, board_detected = detect_chess(path)
            positions = get_piece_positions(boxes, classes, names, cv2.imread(path).shape)
            results['board_detected'] = board_detected
            piece_count = 0
            if boxes is not None and classes is not None:
                for i in range(len(boxes)):
                    cls = int(classes[i])
                    if 'board' not in names[cls].lower():
                        piece_count += 1
            if piece_count > 0:
                results['pieces_detected'] = True
            results['piece_count'] = piece_count
            if positions and any(('row' in p and 'col' in p) for p in positions):
                results['positions_calculated'] = True
            # 生成FEN串
            fen = get_fen_from_positions(positions)
            # 分别预测白方和黑方的最佳走法
            if fen:
                board = chess.Board(fen)
                try:
                    with chess.engine.SimpleEngine.popen_uci('Stockfish-master/stockfish.exe') as engine:
                        # 白方
                        board_white = chess.Board(fen)
                        board_white.turn = chess.WHITE
                        info_white = engine.analyse(board_white, chess.engine.Limit(time=0.1))
                        move_white = info_white['pv'][0]
                        best_move_white = move_white.uci()
                        # 黑方
                        board_black = chess.Board(fen)
                        board_black.turn = chess.BLACK
                        info_black = engine.analyse(board_black, chess.engine.Limit(time=0.1))
                        move_black = info_black['pv'][0]
                        best_move_black = move_black.uci()
                except Exception as e:
                    best_move_white = None
                    best_move_black = None
            # 棋盘高亮当前轮到一方的推荐走法
            chessboard_path = draw_chessboard_with_pieces(positions, output_path='static/chessboard_result.jpg')
            if best_move_white:
                chessboard_path = highlight_move_on_chessboard(chessboard_path, best_move_white)
            return render_template('index.html', image_url=detection_path, chessboard_url=chessboard_path, error=error_message, results=results, best_move_white=best_move_white, best_move_black=best_move_black)
        except Exception as e:
            error_message = f"检测失败: {str(e)}"
            chessboard_path = draw_chessboard_with_pieces([])
            return render_template('index.html', error=error_message, image_url=None, chessboard_url=chessboard_path, results=results, best_move_white=None, best_move_black=None)
    empty_results = {
        'board_detected': False,
        'pieces_detected': False,
        'positions_calculated': False,
        'piece_count': 0
    }
    return render_template('index.html', image_url=None, chessboard_url=None, error=None, results=empty_results, best_move_white=None, best_move_black=None)

if __name__ == '__main__':
    app.run(debug=True)