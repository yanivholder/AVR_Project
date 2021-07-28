from typing import List
import recognition
import numpy as np


class Detect(object):
    def __init__(self, known_faces: List, known_faces_name: List):
        self.known_faces = known_faces
        self.known_faces_name = known_faces_name

    def get_closest(self, face_encoding):
        face_distances = recognition.face_distance(self.known_faces, face_encoding)
        best_match_index = np.argmin(face_distances)
        if recognition.compare_faces(self.known_faces, face_encoding, tolerance=0.6)[best_match_index]:
            return self.known_faces_name[best_match_index]
        return None

    def get_match(self, face_encoding):
        return recognition.compare_faces(self.known_faces, face_encoding, tolerance=0.6)