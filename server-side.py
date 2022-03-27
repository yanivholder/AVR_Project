import os.path
import pickle
import random
import shutil
import struct
from time import sleep

from kivy.support import install_twisted_reactor

install_twisted_reactor()

from twisted.internet.protocol import Factory, Protocol
from twisted.internet import reactor
from twisted.internet.endpoints import TCP4ServerEndpoint

from kivy.app import App
from kivy.uix.label import Label

import cv2
import logging
from datetime import datetime

from our_code.detect_picture import DetectImage
from our_code.recognition import DeepFaceModel, DetectFace
import our_code.camera_stream.camera_streamer as cam_streamer
import our_code.server_config as server_config

payload_size = struct.calcsize("Q")
RECV_SIZE = 4 * 1024  # 4 KB

HOST, PORT = "127.0.0.1", 9879
# HOST, PORT = '', 9879

os.makedirs("logs", exist_ok=True)
# logging.basicConfig(filename="server_logs/{}.txt".format(datetime.now().strftime("%m_%d_%H_%M")), filemode='w', level=logging.INFO)

logging.basicConfig(level=logging.DEBUG)

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
    server_db = 'server_db'
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.imgs_folder = f'{self.server_db}/server_{random.getrandbits(128)}'
        os.makedirs(self.server_db, exist_ok=True)
        self.data = b''
        self.msg_size = None

        self.detection = DetectImage(tolerance=server_config.tolerance, increase_ratio=server_config.increase_ratio)
        self.recognition = DeepFaceModel(self.imgs_folder, distance_metric=server_config.distance_metric,
                                    detector_backend=server_config.detector_backend)
        self.label = None

    def build(self):
        self.label = Label(text="server started\n")
        reactor.listenTCP(PORT, ServerAnswerFactory(self))
        return self.label

    def handle_message(self, msg):
        logging.debug("server receive data")
        sleep(0.03)
        if len(self.data) < payload_size:
            self.data += msg
            packed_msg_size = self.data[:payload_size]
            self.data = self.data[payload_size:]
            self.msg_size = struct.unpack("Q", packed_msg_size)[0]
            return None

        self.data += msg

        if len(self.data) >= self.msg_size:
            logging.debug("server receive a whole message end")
            data = pickle.loads(self.data)
            self.data = b''
            self.msg_size = 0
            #imges
            if len(data) == 2 and data[0] == 'imgs':
                self.handle_new_imgs(data[1])
                result = cam_streamer.pickle_data('images loaded')
            #single frame
            else:
                result = self.handle_single_frame(data)

            return result

    def handle_new_imgs(self, imgs):
        shutil.rmtree(self.imgs_folder, ignore_errors=True)
        os.mkdir(self.imgs_folder)
        for i, (name, img) in enumerate(imgs):
            os.makedirs(os.path.join(self.imgs_folder, name), exist_ok=True)
            cv2.imwrite(os.path.join(self.imgs_folder, name, f'{i}.png'), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        self.recognition.crate_representation_from_model()

    def handle_single_frame(self, frame):
        result = self.find_faces(frame, self.detection, self.recognition, resize_ratio=0.5)
        pickle_result = cam_streamer.pickle_data(result)
        return pickle_result

    def find_faces(self, frame, face_detector: DetectImage, face_recognizer: DetectFace,
                   resize_ratio: float = 1):

        # TODO: we can resize picture to better preformance
        small_frame = cv2.resize(frame, (0, 0), fx=resize_ratio, fy=resize_ratio)

        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_frame = small_frame[:, :, ::-1]

        start = datetime.now()
        face_locations, face_names, scores, face_detected = face_detector.detect(rgb_frame, face_recognizer)
        delta_time = (datetime.now() - start).total_seconds()
        logging.info("time {} faces {} scores {} detection_phase_number {}".format(
            delta_time, face_names, scores, face_detected, ))
        face_locations = [self.mult_location_by_ratio(face_cor, resize_ratio) for face_cor in face_locations]
        return face_locations, face_names, scores

    def mult_location_by_ratio(self, face_cor, resize_ratio):
        return [int(cor / resize_ratio) for cor in face_cor]


if __name__ == "__main__":
    ServerApp().run()
    # # Create the server and bind it to current IP on port 9879
    # endpoint = TCP4ServerEndpoint(reactor, PORT)
    # # Listens to the protocol made by the factory
    # endpoint.listen(Factory.forProtocol(ServerAnswer))
