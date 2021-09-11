import os.path
import socketserver
import time
from threading import Thread

import cv2
import logging
import camera_stream.camera_streamer as cam_streamer
from datetime import datetime

from detect_picture import DetectImage
from draw_faces import BoxConfig, Draw
from recognition import DeepFaceModel, DetectFace
from multiprocessing.pool import ThreadPool, ApplyResult
import server_config

HOST, PORT = "127.0.0.1", 9879
os.makedirs("servers/logs", exist_ok=True)
logging.basicConfig(filename="servers/logs/{}.txt".format(datetime.now().strftime("%m_%d_%H_%M")), filemode='w', level=logging.INFO)


class MyTCPHandler(socketserver.BaseRequestHandler):
    def handle(self) -> None:
        detection = DetectImage(tolerance=server_config.tolerance, increase_ratio=server_config.increase_ratio)
        recognition = DeepFaceModel(server_config.img_folder, distance_metric=server_config.distance_metric,
                                    detector_backend=server_config.detector_backend)
        box_config = BoxConfig(box_thickness=server_config.box_thickness)
        box_drawer = Draw()
        frame_number = 0
        face_locations, face_names, scores = [], [], []
        pool = ThreadPool()
        first_run = True

        while True:
            # Receive the data from TCP socket
            frame = cam_streamer.get_frame_from_socket(self.request)

            # Manipulate the
            if first_run:
                result: ApplyResult = pool.apply_async(self.find_faces, args=(frame, frame_number, detection, recognition))
                first_run = False

            elif result.ready():
                face_locations, face_names, scores = result.get()
                result = pool.apply_async(self.find_faces, args=(frame, frame_number, detection, recognition))

            box_drawer.draw_faces(frame, box_config, face_locations, face_names, scores, print_scores=False)
            # Write the resulting image to the output video file
            frame_number += 1

            # Send the data back
            message = cam_streamer.pickle_data_for_packet(frame)
            self.request.sendall(message)

    def find_faces(self, frame, frame_number: int, face_detector: DetectImage, face_recognizer: DetectFace,
                   resize_ratio: float = 1):

        # TODO: we can resize picture to better preformance
        small_frame = cv2.resize(frame, (0, 0), fx=resize_ratio, fy=resize_ratio)

        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_frame = small_frame[:, :, ::-1]

        start = datetime.now()
        face_locations, face_names, scores, face_detected = face_detector.detect(rgb_frame, face_recognizer)
        delta_time = (datetime.now() - start).total_seconds()
        logging.info("time {} frame {} faces {} scores {} detection_phase_number {}".format(
            delta_time,frame_number,  face_names, scores, face_detected, ))
        face_locations = [self.mult_location_by_ratio(face_cor, resize_ratio) for face_cor in face_locations]
        return face_locations, face_names, scores

    def mult_location_by_ratio(self, face_cor, resize_ratio):
        return [int(cor / resize_ratio) for cor in face_cor]


if __name__ == "__main__":
    # Create the server and bind it to localhost on port 9879
    with socketserver.TCPServer((HOST, PORT), MyTCPHandler) as server:
        # This will run forever until you interrupt the program
        # with Ctrl-C
        server.serve_forever()
