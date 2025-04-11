import warnings
warnings.filterwarnings('ignore')
from ultralytics import RTDETR

if __name__ == '__main__':
    model = RTDETR('path') # select your model.pt path
    model.predict(
                source=r'path/to/your/picture.mp4',  # video path
                project='runs/detect',
                name='exp',
                save=True,
                save_txt=True,  # save results to *.txt
                  )