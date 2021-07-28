from draw_faces import BoxConfig
from video import VideoDetection
from detection import SSDDetection
import cv2
import os
from deepface import DeepFace
import time
import recognition

def save_photo(url: str, output: str):
    detector = DetectImage('imgs')
    img = cv2.imread(url)
    detector.detect(img)
    cv2.imwrite(output, img)


def save_video(video_path: str, output: str):
    input_movie = cv2.VideoCapture(video_path)
    v = VideoDetection('imgs', frame_ratio=5, resize_ratio=1
                       # model='cnn'
                       )
    v.create_movie(input_movie, output, BoxConfig())


save_video("videos/ofir_amit_high_res.mp4", "high_res_two_sample_ratio_ratio_5_tolerence_0.7_gpu.avi")
# save_video("imgs/ofir_amit.mp4", "ofir_amit.avi")
# save_bphoto(r'imgs_test/yaniv_adi_and_other.jpg', 'temp.jpg')
# names = ["retinaface",  "opencv", "ssd", "dlib"]
# names = []
# for m_name in names:
#     start_time = time.clock()
#     df = DeepFace.detectFace("test.jpg", detector_backend=m_name)
#     print("{} seconds with {}".format(time.clock() - start_time, m_name))
#
#
#
path = "test.jpg"
s = SSDDetection()


# def crop(path):
    # pic = cv2.imread(path)
    # faces = s.detect_picture(pic)
    # face_1_cor = faces[0]
    # face_1 = pic[face_1_cor[0]:face_1_cor[2], face_1_cor[3]: face_1_cor[1]]
    # cv2.imwrite('image4.jpg',  face_1)
    # for m in ["OpenFace", "DeepID", "Dlib"]:
    #     start_time = time.clock()
    #     DeepFace.find('image4.jpg', db_path="imgs", model_name=m, enforce_detection=False, align=False,
    #                   detector_backend="opencv")
    #     print("{} seconds with {}".format(time.clock() - start_time, m))
    #

# TODO - deepid with ssd on tesla v100 gpu 0.26sec not on gpu 0.51
# TODO - deepid with opencv on tesla v100 gpu 0.18sec not on gpu 0.298
# TODO - Dlib with opencv on tesla v100 gpu 0.23sec not on gpu 0.45

