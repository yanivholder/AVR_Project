import unittest
from our_code.recognition import FaceRecognitionModel
import cv2
from our_code import recognition


class TestFaceRecognitionModel(unittest.TestCase):
    def setUp(self) -> None:
        self.face_model = FaceRecognitionModel('images')

    @staticmethod
    def get_encoding(frame):
        return recognition.face_encodings(frame)

    def test_same_picture_yaniv(self):
        img = cv2.imread(r'images/yaniv/yaniv.jpg')
        self.assertEqual("yaniv", self.face_model.detect_face(self.get_encoding(img)[0])[0])

    def test_same_picture_obama(self):
        img = cv2.imread(r'images/obama/obama.jpg')
        self.assertEqual("obama", self.face_model.detect_face(self.get_encoding(img)[0])[0])

    def test_random_pic_1(self):
        img = cv2.imread(r'images/random_pic1.jpg')
        self.assertEqual(None, self.face_model.detect_face(self.get_encoding(img)[0])[0])

    def test_yaniv_different_pic(self):
        img = cv2.imread(r'images/yaniv_pic.jpg')
        self.assertEqual("yaniv", self.face_model.detect_face(self.get_encoding(img)[0])[0])
