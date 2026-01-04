import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO(model=r'ultralytics/cfg/models/11/yolo11-fc.yaml')
    model.train(data=r"CCPD.yaml",
                imgsz=640,  # 指定输入图像的尺寸（宽和高）。YOLO 模型会将所有输入图像缩放到这个尺寸进行处理。
                epochs=50,  # 模型会迭代 50 次训练数据。
                batch=16,  # 每次训练时，模型会处理 4 张图像。
                workers=4,
                device='',
                optimizer='SGD',
                resume=False,  # 指定是否从上次中断的地方继续训练。
                project='runs/train',
                name='exp',
                single_cls=False,  # 指定是否为单类别训练。
                cache=False,  # 指定是否缓存数据集。
                )
