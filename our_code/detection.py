import abc
import pandas as pd
from typing import List
from cv2 import dnn, resize
from mtcnn_cv2 import MTCNN


class FaceDetector:
    def __init__(self, tolerance, increase_ratio):
        self.tolerance = tolerance
        self.increase_ratio = increase_ratio

    @abc.abstractmethod
    def detect_picture(self, frame) -> List:
        pass

    def max_0(self, pixel):
        return max(0, pixel - self.increase_ratio)

    def min_shape(self, pixel, shape):
        return min(pixel + self.increase_ratio, shape)
#
# class FaceRecognition(FaceDetector):
#     def __init__(self, model='hog', number_of_times_to_upsample: int = 2):
#         self.model = model
#         self.number_of_times_to_upsample = number_of_times_to_upsample
#
#     def detect_picture(self, frame):
#         return face_recognition.face_locations(frame, model=self.model, number_of_times_to_upsample=self.number_of_times_to_upsample)


class Mtcnn(FaceDetector):
    def __init__(self, tolerance: float, increase_ratio: int):
        super().__init__(tolerance, increase_ratio)
        self.detector = MTCNN()

    def detect_picture(self, frame) -> List:
        faces = []
        detection = self.detector.detect_faces(frame)
        for face in detection:
            if face['confidence'] > self.tolerance:
                top = self.max_0(face['box'][1])
                bottom = self.min_shape(top + face['box'][3], frame.shape[0])
                left = self.max_0(face['box'][0])
                right = self.min_shape(left + face['box'][2], frame.shape[1])
                faces.append((top, right, bottom, left))

            if len(faces) > 2:
                print(detection)
        return faces


class SSDDetection(FaceDetector):
    def __init__(self, tolerance: float = 0.5, increase_ratio=0):
        super().__init__(tolerance, increase_ratio)
        self.detector = dnn.readNetFromCaffe("deploy.prototxt.txt", "res10_300x300_ssd_iter_140000.caffemodel")
        self.target_size = (300, 300)

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
            left = self.max_0(int(instance["left"] * 300 * aspect_ratio_x))
            bottom = min(int((instance["bottom"] * 300 * aspect_ratio_y) + self.increase_ratio), frame.shape[0])
            right = min(int((instance["right"] * 300 * aspect_ratio_x) + self.increase_ratio), frame.shape[1])
            top = self.max_0(int(instance["top"] * 300 * aspect_ratio_y))

            faces.append((top, right, bottom, left))

        return faces

    def calc_resize_ratio(self, frame):
        original_size = frame.shape
        aspect_ratio_x = (original_size[1] / self.target_size[1])
        aspect_ratio_y = (original_size[0] / self.target_size[0])
        return aspect_ratio_x, aspect_ratio_y


