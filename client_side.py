import pickle
import struct

from kivy.support import install_twisted_reactor

from our_code import server_config
from our_code.draw_faces import BoxConfig, Draw

install_twisted_reactor()

from twisted.internet import reactor, protocol
from twisted.protocols.basic import LineReceiver

from kivy.app import App
from kivy.uix.image import Image
from kivy.uix.boxlayout import BoxLayout
from kivy.graphics.texture import Texture
from kivy.clock import Clock

import cv2
from datetime import datetime
import time

import our_code.camera_stream.camera_streamer as cam_streamer

payload_size = struct.calcsize("Q")
RECV_SIZE = 4 * 1024  # 4 KB

HOST, PORT = "localhost", 9879
# HOST, PORT = "132.68.39.159", 9879

from kivy.lang import Builder

Builder.load_string('''
<CameraClick>:
    orientation: 'vertical'
    Camera:
        id: camera
        resolution: (640, 480)
        play: False
    ToggleButton:
        text: 'Play'
        on_press: camera.play = not camera.play
        size_hint_y: None
        height: '48dp'
''')


class AVRClient(protocol.Protocol):
    def connectionMade(self):
        self.factory.app.on_connection(self.transport)

    def dataReceived(self, data):
        self.factory.app.data_received(data)


class AVRClientFactory(protocol.ClientFactory):
    protocol = AVRClient

    def __init__(self, app):
        self.app = app

    def startedConnecting(self, connector):
        print('Started to connect.')

    def clientConnectionLost(self, connector, reason):
        print('Lost connection.')

    def clientConnectionFailed(self, connector, reason):
        print('Connection failed.')


class CameraClick(BoxLayout):
    def capture(self):
        '''
        Function to capture the images and give them the names
        according to their captured time and date.
        '''
        camera = self.ids['camera']
        timestr = time.strftime("%Y%m%d_%H%M%S")
        camera.export_to_png("IMG_{}.png".format(timestr))
        print("Captured")


class ClientApp(App):
    connection = None
    textbox = None
    label = None
    transport = None

    def __init__(self, **kwargs):
        super(ClientApp, self).__init__(**kwargs)

        # init drawer objects
        self.box_config = BoxConfig(box_thickness=server_config.box_thickness)
        self.box_drawer = Draw()
        self.location = []
        self.names = []
        self.scores = []

        # data receive
        self.counter = 1
        self.current_msg_size = 0
        self.current_data_received = b''
        self.got_response = True

    def build(self):
        # Define a video capture object
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise IOError("Cannot open webcam")

        root = self.setup_gui()
        self.connect_to_server()
        return root

    def setup_gui(self):
        # return Image(source=)
        self.img1 = Image()
        layout = BoxLayout()
        layout.add_widget(self.img1)
        # opencv2 stuffs
        self.cap = cv2.VideoCapture(0)
        return layout

    def connect_to_server(self):
        reactor.connectTCP(HOST, PORT, AVRClientFactory(self))

    def on_connection(self, transport):
        print("Connected successfully!")
        self.transport = transport
        Clock.schedule_interval(self.show_video, 1.0 / 32)

    def show_video(self, dt):
        # display image from cam in opencv window
        ret, frame = self.cap.read()

        # Send the frame to the server
        if self.current_data_received == b'' and self.got_response:
            frame_bytes = cam_streamer.pickle_data(frame)
            self.transport.write(frame_bytes)
            self.got_response = False
            print("data send")

        self.box_drawer.draw_faces(frame, self.box_config, self.location, self.names, self.scores, print_scores=False)
        self.show_image(frame)

    def data_received(self, msg):
        print('client recive')
        if len(self.current_data_received) < payload_size:
            packed_msg_size = msg[:payload_size]
            self.current_data_received = msg[payload_size:]
            self.current_msg_size = struct.unpack("Q", packed_msg_size)[0]
        else:
            self.current_data_received += msg

        if len(self.current_data_received) >= self.current_msg_size:
            assert len(self.current_data_received) == self.current_msg_size
            self.location, self.names, self.scores = pickle.loads(self.current_data_received)
            self.current_data_received = b''
            self.current_msg_size = 0
            self.got_response = True

    def show_image(self, frame):
        # Show the frames in a screen.
        buf = cv2.flip(frame, 0).tostring()
        texture1 = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
        texture1.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')

        # display image from the texture
        self.img1.texture = texture1
        self.counter += 1
    #
    # def generate_msg_to_server(self):
    #     # Capture one frame
    #     ret, frame = self.cap.read()
    #     if ret is False:
    #         # TODO: think what to do here
    #         pass
    #
    #     # Send the data
    #     self.time = datetime.now()
    #     message = frame
    #     self.transport.write(message)
    #     # logging.info("frame {}  time {} ".format(counter,  (datetime.now() - time).total_seconds()))

if __name__ == '__main__':
    ClientApp().run()
    # cap.release()
    cv2.destroyAllWindows()
