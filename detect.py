import warnings
warnings.filterwarnings('ignore')
from ultralytics import RTDETR

if __name__ == '__main__':
<<<<<<< HEAD
    model = RTDETR('path') # select your model.pt path
    model.predict(
                source=r'path/to/your/picture.mp4',  # video path
                project='runs/detect',
                name='exp',
                save=True,
                save_txt=True,  # save results to *.txt
=======
    model = RTDETR('weights/ours.pt') # select your model.pt path
    model.predict(source='demo/fish2152.mp4',
                  project='runs/detect',
                  name='exp',
                  save=True,
                #   visualize=True # visualize model features maps
>>>>>>> upstream/main
                  )