import abc
import recognition
import numpy as np
from deepface import DeepFace
from ImgParser import Parser
from abc import ABC
from typing import List


class DetectFace:
    def __init__(self, img_path):
        self.img_path = img_path

    @abc.abstractmethod
    def detect_face(self, target_face) -> (str, int):
        pass


class FaceRecognitionModel(DetectFace, ABC):
    def __init__(self, img_path,  tolerance: float = 0.6):
        super().__init__(img_path)
        self.known_faces = Parser(self.img_path).get_all_people()
        self.tolerance = tolerance

    def detect_face(self, target_face) -> (str, int):
        best_score = None
        best_name = None
        for name, imgs in self.known_faces.items():
            score = self.best_score_for_person(imgs, target_face)
            if best_score is None or score < best_score:
                best_score = score
                best_name = name
        if best_score < self.tolerance:
            return best_name, best_score
        return None, None

    @staticmethod
    def best_score_for_person(pictures: List, target_face):
        scores = recognition.face_distance(pictures, target_face)
        return np.amin(scores)

#
# class DeepFaceModel(DetectFace, ABC):
#     def __init__(self, img_path,  tolerance: float = 0.6, model_name: str ='VGG-Face'):
#         super().__init__(img_path)
#         self.tolerance = tolerance
#         self.model = DeepFace.build_model(model_name)
#
#     def detect_face(self, target_face) -> (str, int):
#         pass
#
#     @staticmethod
#     def crop(pictures: List, target_face):
#         scores = face_recognition.face_distance(pictures, target_face)
#         return np.amin(scores)
#
