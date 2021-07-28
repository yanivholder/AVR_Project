import unittest
from detect_picture import DetectImage
from draw_faces import BoxConfig
from recognition import FaceRecognitionModel
import cv2
import recognition


box_config = BoxConfig()


class TestDetect(unittest.TestCase):
    def setUp(self) -> None:
        self.detector = DetectImage()
        self.face_model = FaceRecognitionModel('imgs_test')

    def test_same_picture_yaniv(self):
        img = cv2.imread(r'imgs_test/yaniv/yaniv.jpg')
        locations, names, scores = self.detector.detect(img, self.face_model)
        self.assertEqual(1, len(locations))
        self.assertEqual(1, len(names))
        self.assertEqual("yaniv", names[0])

    def test_same_picture_obama(self):
        img = cv2.imread(r'imgs_test/obama/obama.jpg')
        locations, names, scores = self.detector.detect(img, self.face_model)
        self.assertEqual(1, len(locations))
        self.assertEqual(1, len(names))
        self.assertEqual("obama", names[0])

    def test_multiple(self):
        img = cv2.imread(r'imgs_test/yaniv_adi_and_other.jpg')
        locations, names, scores = self.detector.detect(img, self.face_model)
        self.assertEqual(4, len(locations))
        self.assertEqual(4, len(names))
        self.assertEqual(1, names.count("yaniv"))
        self.assertEqual(1, names.count("adi"))


class TestTime(unittest.TestCase):
    def setUp(self) -> None:
        self.obama_img = cv2.imread(r'imgs_test/obama/obama.jpg')
        self.biden_img = cv2.imread(r'imgs_test/biden/biden.jpg')
        self.obama_encoding = recognition.face_encodings(self.obama_img)[0]
        self.biden_encoding = recognition.face_encodings(self.biden_img)[0]

    def test_get_encoding_single(self):
        img = cv2.imread(r'imgs_test/yaniv/yaniv.jpg')
        recognition.face_encodings(img)

    def test_detect_face_single(self):
        recognition.face_distance([self.obama_encoding], self.biden_encoding)

