import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('')
    model.val(data='',
              split='',
              imgsz=640,
              batch=16,
              # rect=False,
              # save_json=True, # if you need to cal coco metriceP
              project='',
              name='exp',
              )
    