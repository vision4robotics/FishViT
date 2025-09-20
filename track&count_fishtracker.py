import warnings
import cv2
from ultralytics import RTDETR
from ultralytics import YOLO
from src.V4R_sv.Line.Color_list import Color
from src.V4R_sv.Line.Line_counter import LineZone, LineZoneAnnotator
from src.V4R_sv.Line.Annotor import BoxAnnotator
from src.V4R_sv.Line.detectcore import Detections
from src.V4R_sv.Line.gemcore import Point
import sys
import os
import csv

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
warnings.filterwarnings('ignore')


def check_files_for_dash(directory):
    results = {}
    for filename in os.listdir(directory):
        if filename.endswith('.txt'):
            results[filename] = '-' in filename
    return results


if __name__ == '__main__':
    folder_path = './folder_path'  # Directory containing video files
    output_folder_path = './output_folder_path'  # Directory to save the txt files
    os.makedirs(output_folder_path, exist_ok=True)  # Create the output directory if it doesn't exist

    video_files = []
    fish_count = {}
    video_fish_frame = {}

    # Collect video fil
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(('.mp4', '.avi', '.mkv', '.flv', '.mov')):
                video_files.append(os.path.join(root, file))

    frame_interval = 15  # Set the frame interval

    # Process each video file
    for video_file in video_files:
        model = RTDETR(r'model path')  # Load the model

        if hasattr(model, 'predictor') and hasattr(model.predictor, 'trackers'):
            model.predictor.trackers = {}

        line_counter = LineZone(start=Point(800, 0), end=Point(800, 1080))  # Define line for crossing detection
        # Visualization setup
        line_annotator = LineZoneAnnotator(thickness=2, text_thickness=2, text_scale=2, color=Color(r=224, g=57, b=151))
        box_annotator = BoxAnnotator(thickness=3, text_thickness=1, text_scale=1, color=Color(r=255, g=0, b=0),
                                     text_color=Color(r=255, g=255, b=255))
        bias = 0

        cap = cv2.VideoCapture(video_file)  # Open video file
        frame_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = 0
        frame_counter = 0  # Initialize the frame counter
        txt_file_path = os.path.join(output_folder_path, os.path.splitext(os.path.basename(video_file))[0] + '.txt')
        output_video_path = os.path.join(output_folder_path,
                                         os.path.splitext(os.path.basename(video_file))[0] + '_output.mp4')

        m_bias = 0
        fish_count_frame = {}
        int_count = 0
        for result in model.track(source=video_file, conf=0.25, tracker='fishtracker.yaml', project='runs/track',
                                  name='fishvit', save=False):
            frame = result.orig_img  # Get original image
            detections = Detections.from_yolov8(result)  # Parse detection results
            if result.boxes.id is not None:
                detections.tracker_id = result.boxes.id.cpu().numpy().astype(int)  # Parse tracking ID
                # Get boxes, track IDs, class IDs, confidences
                boxes = result.boxes.xywh.cpu().numpy()
                track_ids = result.boxes.id.int().cpu().tolist()
                class_ids = detections.class_id
                confidences = detections.confidence
                tracker_ids = detections.tracker_id
                labels = ['#{} {:.1f}'.format(model.names[class_ids[i]], confidences[i]) for i in range(len(class_ids))]
                # Annotate frame
                frame = box_annotator.annotate(scene=frame, detections=detections, labels=labels)

                # Line crossing detection
                line_counter.trigger(detections=detections)
                line_annotator.annotate(frame=frame, line_counter=line_counter)
                if model.predictor.count_bias != 0:
                    bias = model.predictor.count_bias
                fish_num = line_counter.in_count - line_counter.out_count + bias

                if line_counter.in_count > int_count or m_bias != bias:
                    int_count = line_counter.in_count
                    m_bias = bias
                    fish_count_frame[int_count - 1] = {'frame_count': frame_count, 'in_count': line_counter.in_count,
                                                       'out_count': line_counter.out_count, 'bias': bias}

            frame_count += 1
            video_fish_frame[video_file] = fish_count_frame

        cap.release()

        # Display frame
        fish_count[video_file] = {'in_count': line_counter.in_count + bias, 'out_count': line_counter.out_count}
        print("fish_num: ", fish_num)

    # Save results to CSV
    # with open('', 'w', newline='') as csvfile:
    #     fieldnames = ['video_file', 'in_count', 'out_count']
    #     writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    #     writer.writeheader()
    #     for video_file, counts in fish_count.items():
    #         writer.writerow(
    #             {'video_file': video_file, 'in_count': counts['in_count'], 'out_count': counts['out_count']})
    # with open('./exp4bytetrack/fish_count_frame_FishVIT.csv', 'w', newline='') as csvfile:
    #     fieldnames = ['video_file', 'frame_count', 'in_count', 'out_count', 'bias']
    #     writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    #     writer.writeheader()
    #     for video_file, counts in video_fish_frame.items():
    #         for frame_count, frame_counts in counts.items():
    #             writer.writerow(
    #                 {'video_file': video_file, 'frame_count': frame_counts['frame_count'],
    #                  'in_count': frame_counts['in_count'],
    #                  'out_count': frame_counts['out_count'], 'bias': frame_counts['bias']})