# 棋盘与棋子识别Web应用
=======
# Computer-Vision-Applications-for-Chess
This is our group's final project for the Computer Vision course in the spring semester of 2025.
B站视频链接：
## 功能
- 识别棋盘
- 识别棋盘上的棋子
- 判断每个棋子在棋盘的哪个格子里
- Web界面上传图片并显示结果

## 环境安装

1. 安装依赖（用清华镜像）：
   ```
   pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt
   ```

2. 准备模型
   - 用你的数据集训练yolov8模型，得到`best.pt`，放到rectified_dataset目录下。

## 运行

```
python app.py
```

浏览器访问 http://127.0.0.1:5000

## 使用方法

1. 上传包含棋盘和棋子的图片
2. 系统自动识别棋盘、棋子及其所在格子
3. 页面显示识别结果

## 目录结构

- app.py：Web后端
- yolov8_infer.py：识别推理
- templates/index.html：前端页面
- static/style.css：样式
- requirements.txt：依赖
- chess(DST1535)/：数据集及模型

## 常见问题

- 若模型未训练，请先用yolov8训练，参考Ultralytics官方文档。
- 若依赖安装失败，请检查Python版本和网络。 

## linux云服务器训练
1. 开启后台训练
   ```
   nohup yolo detect train data=data.yaml model=yolov8n.pt epochs=50 imgsz=416 batch=10 > train.log 2>&1 &
   ```
2. 检查进程是否存在
   ```
   ps aux | grep "yolo detect train"
   ```
3. 查看日志文件
   ```
   tail -f train.log  # 实时监控日志
   ```

## 依赖安装

推荐使用清华镜像源安装依赖：

```bash
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt
```

## 运行说明

1. 启动Web服务：
```bash
python app.py
```
2. 上传棋盘图片，识别并获得AI推荐走法。



