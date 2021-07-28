import unittest
from recognition import FaceRecognitionModel
import cv2
import recognition


class TestFaceRecognitionModel(unittest.TestCase):
    def setUp(self) -> None:
        self.face_model = FaceRecognitionModel('imgs_test')

    @staticmethod
    def get_encoding(frame):
        return recognition.face_encodings(frame)

    def test_same_picture_yaniv(self):
        img = cv2.imread(r'imgs_test/yaniv/yaniv.jpg')
        self.assertEqual("yaniv", self.face_model.detect_face(self.get_encoding(img)[0])[0])

    def test_same_picture_obama(self):
        img = cv2.imread(r'imgs_test/obama/obama.jpg')
        self.assertEqual("obama", self.face_model.detect_face(self.get_encoding(img)[0])[0])

    def test_random_pic_1(self):
        img = cv2.imread(r'imgs_test/random_pic1.jpg')
        self.assertEqual(None, self.face_model.detect_face(self.get_encoding(img)[0])[0])

    def test_yaniv_different_pic(self):
        img = cv2.imread(r'imgs_test/yaniv_pic.jpg')
        self.assertEqual("yaniv", self.face_model.detect_face(self.get_encoding(img)[0])[0])
