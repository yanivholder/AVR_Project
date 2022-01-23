import cv2
import logging
from datetime import datetime
from cv2 import VideoCapture
from our_code.draw_faces import BoxConfig, Draw

from our_code.detect_picture import DetectImage
from our_code.recognition import DetectFace, DeepFaceModel

# This is a demo of running face recognition on a video file and saving the results to a new video file.
#
# PLEASE NOTE: This example requires OpenCV (the `cv2` library) to be installed only to read from your webcam.


class VideoDetection:
    def __init__(self, known_img_path: str,  frame_ratio: int = 2, resize_ratio: float = 1, increase_ratio=10,
                 detector_backend='mtcnn', distance_metric='cosine'):
        self.frame_ratio = frame_ratio
        self.resize_ratio = resize_ratio
        self.box_drawer = Draw()
        self.face_detector = DetectImage(tolerance=0.9, increase_ratio=increase_ratio)
        self.face_recognizer: DetectFace = DeepFaceModel(known_img_path, distance_metric=distance_metric,  detector_backend=detector_backend)

    @staticmethod
    def create_video_setting(input_movie, output_path: str):
        width = (int(input_movie.get(cv2.CAP_PROP_FRAME_WIDTH)))
        height = (int(input_movie.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        fps = (int(input_movie.get(cv2.CAP_PROP_FPS)))

        # Create an output movie file (make sure resolution/frame rate matches input video!)
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        return cv2.VideoWriter("output/" + output_path, fourcc, fps, (width, height))

    def mult_location_by_ratio(self, face_cor):
        return [int(cor / self.resize_ratio) for cor in face_cor]

    def create_movie(self, input_movie: VideoCapture, output_path: str,  box_config: BoxConfig):
        logging.basicConfig(filename="output/" + output_path[:-4] + "_logs.txt", filemode='w',  level=logging.INFO)
        # Initialize some variables
        frame_number = 0
        length = int(input_movie.get(cv2.CAP_PROP_FRAME_COUNT))
        output_movie = self.create_video_setting(input_movie, output_path)

        while True:
            # Grab a single frame of video
            ret, frame = input_movie.read()

            # # Quit when the input video file ends
            if not ret or frame_number > 1000:
                break

            if frame_number % self.frame_ratio == 0:
                # Resize frame of video to 1/4 size for faster face recognition processing
                small_frame = cv2.resize(frame, (0, 0), fx=self.resize_ratio, fy=self.resize_ratio)

                # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
                rgb_frame = small_frame[:, :, ::-1]

                time = datetime.now()
                face_locations, face_names, scores, face_detected = self.face_detector.detect(rgb_frame, self.face_recognizer)
                delta_time = (datetime.now() - time).total_seconds()
                logging.info("time {} frame {} faces found {} with scores {} detection_phase_number {} ".format(
                    delta_time, frame_number, face_names, scores, face_detected))
                face_locations = [self.mult_location_by_ratio(face_cor) for face_cor in face_locations]

            self.box_drawer.draw_faces(frame, box_config, face_locations, face_names, scores)
            # Write the resulting image to the output video file
            print("Writing frame {} / {}".format(frame_number, length))
            output_movie.write(frame)
            # cv2.imshow('Video', frame)
            frame_number += 1

        # All done!
        input_movie.release()
        cv2.destroyAllWindows()

