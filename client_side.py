from kivy.support import install_twisted_reactor

install_twisted_reactor()

from twisted.internet import reactor, protocol

from kivy.app import App
from kivy.uix.image import Image
from kivy.uix.boxlayout import BoxLayout
from kivy.graphics.texture import Texture

import cv2
from datetime import datetime

import our_code.camera_stream.camera_streamer as cam_streamer

HOST, PORT = "localhost", 9879
# HOST, PORT = "132.68.39.159", 9879


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


class ClientApp(App):
    connection = None
    textbox = None
    label = None
    transport = None

    def build(self):
        # Define a video capture object
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise IOError("Cannot open webcam")
        self.counter = 1

        root = self.setup_gui()
        self.connect_to_server()
        return root

    def setup_gui(self):
        self.img1 = Image()
        layout = BoxLayout()
        layout.add_widget(self.img1)
        return layout

    def connect_to_server(self):
        reactor.connectTCP(HOST, PORT, AVRClientFactory(self))

    def on_connection(self, transport):
        print("Connected successfully!")
        self.transport = transport
        self.transport.write('Hello world'.encode('utf-8'))

    def data_received(self, data):
        print(data)
        # # Receive data back from the server
        # frame = cam_streamer.get_frame_from_socket(data)
        # (datetime.now() - self.time).total_seconds()
        #
        # # Show the frames in a screen.
        # buf1 = cv2.flip(frame, 0)
        # buf = buf1.tostring()
        # texture1 = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
        # texture1.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
        #
        # # display image from the texture
        # self.img1.texture = texture1
        # self.counter += 1

    def generate_msg_to_server(self):
        # Capture one frame
        ret, frame = self.cap.read()
        if ret is False:
            # TODO: think what to do here
            pass

        # Send the data
        self.time = datetime.now()
        message = frame
        self.transport.write(message)
        # logging.info("frame {}  time {} ".format(counter,  (datetime.now() - time).total_seconds()))



if __name__ == '__main__':
    ClientApp().run()


#     cap.release()
#     cv2.destroyAllWindows()
