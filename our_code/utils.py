from draw_faces import BoxConfig
from video import VideoDetection
# # from detect_picture import DetectImage
from our_code.detection import SSDDetection
import cv2
import os
from deepface import DeepFace
import time
# import recognition

increase_ratio = 10
metric = 'cosine'  # ['cosine', 'euclidean', 'euclidean_l2']
detector_backend = 'mtcnn'
video = 'wed'


def save_video(video_path: str, output: str):
    input_movie = cv2.VideoCapture(video_path)
    v = VideoDetection(known_img_path='../tests/images', frame_ratio=5, increase_ratio=increase_ratio,
                       detector_backend=detector_backend, distance_metric=metric,
                       )
    v.create_movie(input_movie, output, BoxConfig())


save_video("videos/{}.mp4.".format(video), video + "+{}_{}_{}_Facenet512.avi".format(increase_ratio, metric, detector_backend))
# save_video("imgs/ofir_amit.mp4", "ofir_amit.avi")
# save_bphoto(r'images/yaniv_adi_and_other.jpg', 'temp.jpg')
# names = ["retinaface",  "opencv", "ssd", "dlib"]
# for m_name in names:
#     start_time = time.clock()
#     df = DeepFace.detectFace("test.jpg", detector_backend=m_name)
#     print("{} seconds with {}".format(time.clock() - start_time, m_name))
#
#

s = SSDDetection()


def crop(path):
    pic = cv2.imread(path)
    faces = s.detect_picture(pic)
    face_1_cor = faces[0]
    face_1 = pic[face_1_cor[0]:face_1_cor[2], face_1_cor[3]: face_1_cor[1]]
    cv2.imwrite('image4.jpg',  face_1)
    for m in  ["OpenFace", "DeepID"]:
        start_time = time.process_time()
        z = DeepFace.find('image4.jpg', db_path="imgs", model_name=m, enforce_detection=False, align=False,
                      detector_backend="ssd")
        print("{} seconds with {}".format(time.process_time() - start_time, m))

# crop(path)
# # TODO - deepid with ssd on tesla v100 gpu 0.26sec not on gpu 0.51
# # TODO - deepid with opencv on tesla v100 gpu 0.18sec not on gpu 0.298
# # TODO - Dlib with opencv on tesla v100 gpu 0.23sec not on gpu 0.45
#
