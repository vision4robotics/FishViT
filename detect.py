import warnings
warnings.filterwarnings('ignore')
from ultralytics import RTDETR

if __name__ == '__main__':
    model = RTDETR('weights/ours.pt') # select your model.pt path
    model.predict(source='demo/fish2152.mp4',
                  project='runs/detect',
                  name='exp',
                  save=True,
                #   visualize=True # visualize model features maps
                  )