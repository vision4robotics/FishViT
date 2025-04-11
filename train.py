import warnings
warnings.filterwarnings('ignore')
from ultralytics import RTDETR
import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'

if __name__ == '__main__':
    model = RTDETR('ultralytics/cfg/models/rt-detr/rtdetr-r18-STattn_TII.yaml')#If you want to train your own model, please use this line
    # model.load('') # loading pretrain weights
    model.train(data='dataset/data.yaml',
                cache=False,
                imgsz=640,
                epochs=150,
                batch=16,
                workers=8,
                device='0',
                # resume='', # last.pt path
                project='TII/train',
                name='exp',
                )