import pickle
import struct

from kivy.support import install_twisted_reactor

install_twisted_reactor()

from kivy.uix.boxlayout import BoxLayout
from kivy.graphics.texture import Texture
from kivy.clock import Clock
from kivy.lang import Builder
from kivy.core.image import Image


import cv2

import our_code.camera_stream.camera_streamer as cam_streamer
from our_code import server_config
from our_code.draw_faces import BoxConfig, Draw

payload_size = struct.calcsize("Q")
RECV_SIZE = 4 * 1024  # 4 KB


Builder.load_string('''
<Recognize>:
    orientation: 'vertical'
    padding: 0
    spacing: 1

    BoxLayout:
        padding: 0
        spacing: 1

        Image:
            id: video

    ToggleButton:
        id: play_or_stop_but
        text: 'Play'
        on_press: root.play_or_stop()
        size_hint_y: None
        height: '48dp'
''')


class Recognize(BoxLayout):
    def __init__(self, **kwargs):
        super(Recognize, self).__init__(**kwargs)

        self.play_state = False
        self.cap = cv2.VideoCapture(0)

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

    def play_or_stop(self):
        '''
        Function to capture the images and give them the names
        according to their captured time and date.
        '''
        self.play_state = not self.play_state

        if self.play_state:
            self.ids['play_or_stop_but'].text = "stop"
            Clock.schedule_interval(self.show_video, 1.0)
        else:
            self.ids['play_or_stop_but'].text = "play"
            Clock.unschedule(self.show_video)
            self.flush_video()

    def flush_video(self):
        self.current_data_received = b''
        self.current_msg_size = 0
        self.location = []
        self.names = []
        self.scores = []
        self.got_response = True
        self.ids['video'].texture = Image('images/yaniv/yaniv2.jpg').texture

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
        texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
        texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')

        video = self.ids['video']
        video.texture = texture
        self.counter += 1

