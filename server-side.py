import os.path
import pickle
import struct

from kivy.support import install_twisted_reactor

install_twisted_reactor()

from twisted.internet.protocol import Factory, Protocol
from twisted.internet import reactor
from twisted.internet.endpoints import TCP4ServerEndpoint

from kivy.app import App
from kivy.uix.label import Label

import cv2
import logging
from multiprocessing.pool import ThreadPool, ApplyResult
from datetime import datetime

from our_code.detect_picture import DetectImage
from our_code.draw_faces import BoxConfig, Draw
from our_code.recognition import DeepFaceModel, DetectFace
import our_code.camera_stream.camera_streamer as cam_streamer
import our_code.server_config as server_config

payload_size = struct.calcsize("Q")
RECV_SIZE = 4 * 1024  # 4 KB

HOST, PORT = "127.0.0.1", 9879
# HOST, PORT = '', 9879

os.makedirs("logs", exist_ok=True)
logging.basicConfig(filename="logs/{}.txt".format(datetime.now().strftime("%m_%d_%H_%M")), filemode='w', level=logging.INFO)


class ServerAnswer(Protocol):
    """This class will be instantiated for each server connection"""

    def dataReceived(self, data: bytes):
        response = self.factory.app.handle_message(data)
        if response:
            self.transport.write(response)


class ServerAnswerFactory(Factory):
    protocol = ServerAnswer

    def __init__(self, app):
        self.app = app


class ServerApp(App):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.data = b''
        self.msg_size = None
        # self.detection = DetectImage(tolerance=server_config.tolerance, increase_ratio=server_config.increase_ratio)
        # self.recognition = DeepFaceModel(server_config.img_folder, distance_metric=server_config.distance_metric,
        #                             detector_backend=server_config.detector_backend)
        # self.box_config = BoxConfig(box_thickness=server_config.box_thickness)
        # self.box_drawer = Draw()
        # self.frame_number = 0
        # self.face_locations, face_names, scores = [], [], []
        # self.pool = ThreadPool()
        # self.first_run = True
        self.label = None

    def build(self):
        self.label = Label(text="server started\n")
        reactor.listenTCP(PORT, ServerAnswerFactory(self))
        return self.label

    def handle_message(self, msg):
        if len(self.data) < payload_size:
            self.data += msg
            packed_msg_size = self.data[:payload_size]
            self.data = self.data[payload_size:]
            self.msg_size = struct.unpack("Q", packed_msg_size)[0]
            return

        self.data += msg

        if len(self.data) >= self.msg_size:
            frame = pickle.loads(self.data)
            #  do something
            self.data = b''
            self.msg_size = 0
            return cam_streamer.pickle_data(frame)

        #     # Manipulate the
        #     if first_run:
        #         result: ApplyResult = pool.apply_async(self.find_faces, args=(frame, frame_number, detection, recognition))
        #         first_run = False
        #
        #     elif result.ready():
        #         face_locations, face_names, scores = result.get()
        #         result = pool.apply_async(self.find_faces, args=(frame, frame_number, detection, recognition))
        #
        #     box_drawer.draw_faces(frame, box_config, face_locations, face_names, scores, print_scores=False)
        #     # Write the resulting image to the output video file
        #     frame_number += 1
        #
        #     # Send the data back
        #     message = cam_streamer.pickle_data_for_packet(frame)
        #     self.request.sendall(message)
        #
        # def find_faces(self, frame, frame_number: int, face_detector: DetectImage, face_recognizer: DetectFace,
        #                resize_ratio: float = 1):
        #
        #     # TODO: we can resize picture to better preformance
        #     small_frame = cv2.resize(frame, (0, 0), fx=resize_ratio, fy=resize_ratio)
        #
        #     # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        #     rgb_frame = small_frame[:, :, ::-1]
        #
        #     start = datetime.now()
        #     face_locations, face_names, scores, face_detected = face_detector.detect(rgb_frame, face_recognizer)
        #     delta_time = (datetime.now() - start).total_seconds()
        #     logging.info("time {} frame {} faces {} scores {} detection_phase_number {}".format(
        #         delta_time, frame_number, face_names, scores, face_detected, ))
        #     face_locations = [self.mult_location_by_ratio(face_cor, resize_ratio) for face_cor in face_locations]
        #     return face_locations, face_names, scores
        #
        # def mult_location_by_ratio(self, face_cor, resize_ratio):
        #     return [int(cor / resize_ratio) for cor in face_cor]


if __name__ == "__main__":
    ServerApp().run()
    # # Create the server and bind it to current IP on port 9879
    # endpoint = TCP4ServerEndpoint(reactor, PORT)
    # # Listens to the protocol made by the factory
    # endpoint.listen(Factory.forProtocol(ServerAnswer))
