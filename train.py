import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('')
    # model.load('yolov8n.pt') # loading pretrain weights
    model.train(data='',
                cache=False,
                imgsz=640,
                epochs=200,
                batch=1,
                close_mosaic=10,
                workers=0,
                device='0',
                optimizer='', # using SGD
                # resume='runs/train\exp3\weights\last.pt', # last.pt path
                # amp=False, # close amp
                # fraction=0.2,
                project='runs/train',
                name='exp',
                )