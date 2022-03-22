import struct

from kivy.support import install_twisted_reactor

install_twisted_reactor()

from twisted.internet import reactor, protocol

from kivy.app import App

import cv2

from app_layout import Recognize


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

    def __init__(self, **kwargs):
        super(ClientApp, self).__init__(**kwargs)
        self.root = Recognize()

    def build(self):
        # Define a video capture object
        # if not self.cap.isOpened():
        #     raise IOError("Cannot open webcam")
        self.connect_to_server()
        return self.root

    def connect_to_server(self):
        reactor.connectTCP(HOST, PORT, AVRClientFactory(self))

    def on_connection(self, transport):
        print("Connected successfully!")
        self.transport = transport
        self.root.transport = transport

    def data_received(self, msg):
        self.root.data_received(msg)


if __name__ == '__main__':
    ClientApp().run()
    # cap.release()
    cv2.destroyAllWindows()
