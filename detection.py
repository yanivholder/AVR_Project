import abc
import pandas as pd
import face_recognition
from typing import List
from cv2 import dnn, resize


class FaceDetector:
    @abc.abstractmethod
    def detect_picture(self, frame) -> List:
        pass


class FaceRecognition(FaceDetector):
    def __init__(self, model='hog', number_of_times_to_upsample: int = 2):
        self.model = model
        self.number_of_times_to_upsample = number_of_times_to_upsample

    def detect_picture(self, frame):
        return face_recognition.face_locations(frame, model=self.model, number_of_times_to_upsample=self.number_of_times_to_upsample)


class SSDDetection:
    def __init__(self, tolerance: float = 0.8):
        self.detector = dnn.readNetFromCaffe("deploy.prototxt.txt", "res10_300x300_ssd_iter_140000.caffemodel")
        self.target_size = (300, 300)
        self.tolerance = tolerance

    def detect_picture(self, frame) -> List:
        small_frame = resize(frame, self.target_size)
        imageBlob = dnn.blobFromImage(image=small_frame)
        self.detector.setInput(imageBlob)
        detections = self.detector.forward()
        detections_df = pd.DataFrame(detections[0][0],
                                     columns=["img_id", "is_face", "confidence", "left", "top", "right", "bottom"])
        detections_df = detections_df[detections_df["is_face"] == 1]  # 0: background, 1: face, 2: confidence
        detections_df = detections_df[detections_df["confidence"] >= self.tolerance]
        aspect_ratio_x, aspect_ratio_y = self.calc_resize_ratio(frame)
        faces = []

        for i, instance in detections_df.iterrows():
            left = int(instance["left"] * 300 * aspect_ratio_x)
            bottom = int(instance["bottom"] * 300 * aspect_ratio_y)
            right = int(instance["right"] * 300 * aspect_ratio_x)
            top = int(instance["top"] * 300 * aspect_ratio_y)

            faces.append((top, right, bottom, left))

        return faces

    def calc_resize_ratio(self, frame):
        original_size = frame.shape
        aspect_ratio_x = (original_size[1] / self.target_size[1])
        aspect_ratio_y = (original_size[0] / self.target_size[0])
        return aspect_ratio_x, aspect_ratio_y

