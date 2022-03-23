import os
import pickle
import struct

from kivy.support import install_twisted_reactor

install_twisted_reactor()

from kivy.uix.boxlayout import BoxLayout
from kivy.graphics.texture import Texture
from kivy.clock import Clock
from kivy.lang import Builder
from kivy.core.image import Image
from kivy.uix.popup import Popup
from kivy.properties import ObjectProperty
from kivy.uix.floatlayout import FloatLayout

import cv2
import numpy as np
import PIL

import our_code.camera_stream.camera_streamer as cam_streamer
from our_code import server_config
from our_code.draw_faces import BoxConfig, Draw

payload_size = struct.calcsize("Q")
RECV_SIZE = 4 * 1024  # 4 KB

# error_msg = "folder must be in the following format\nroot ->\n  name1\n    img1\n    img2\n  name2\n    img1"
# pop up

Builder.load_string('''
<P>:
    id: pop
    size_hint: .4, .2
    auto_dismiss: False
    title: ""
    Button:
        text: "ok"
        pos_hint: {"x":0.1, "y":0.1}
        on_press: pop.dismiss()
''')

# pick folder
Builder.load_string('''
<LoadDialog>:
    BoxLayout:
        size: root.size
        pos: root.pos
        orientation: 'vertical'
        FileChooserListView:
            id: filechooser
            path: "./"

        BoxLayout:
            size_hint_y : None
            height : 30
            Button:
                text: 'Cancel'
                on_release: root.cancel()

            Button:
                text: 'Load'
                on_release: root.load(filechooser.path, filechooser.selection)
''')

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
    
    BoxLayout:
        padding: 0
        spacing: 1
        size_hint: (1.0, 0.1)
        
        TextInput:
            id: txtFName
            text: 'cur dir - null'
            multiline: False
            
        Button:
            text: 'file load'
            on_press: root.file_select()
        
    ToggleButton:
        id: play_or_stop_but
        text: 'Play'
        on_press: root.play_or_stop()
        size_hint_y: None
        height: '48dp'
''')


class P(Popup):
    pass

#Video selection pop-up
class LoadDialog(FloatLayout):
    load = ObjectProperty(None)
    cancel = ObjectProperty(None)


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
            self.flush_video()
            Clock.schedule_interval(self.show_video, 1.0 / 8)
        else:
            self.ids['play_or_stop_but'].text = "play"
            Clock.unschedule(self.show_video)
            self.ids['video'].texture = Image('images/yaniv/yaniv2.jpg').texture

    def flush_video(self):
        self.current_data_received = b''
        self.current_msg_size = 0
        self.location = []
        self.names = []
        self.scores = []
        self.got_response = True

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
        """

        :param msg: msg can be one of two - faces location
        :return:
        """
        print('client recive')
        if len(self.current_data_received) < payload_size:
            packed_msg_size = msg[:payload_size]
            self.current_data_received = msg[payload_size:]
            self.current_msg_size = struct.unpack("Q", packed_msg_size)[0]
        else:
            self.current_data_received += msg

        if len(self.current_data_received) >= self.current_msg_size:
            assert len(self.current_data_received) == self.current_msg_size

            data = pickle.loads(self.current_data_received)
            if len(data) == 3:
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

    #########################################load images ####################################
    #Pop-up to load video
    
    def file_select(self):
        if self.play_state:
            self.show_pop_up("can not load images while playing")
            return
        
        content = LoadDialog(load=self.load, cancel=self.dismiss_popup)
        self._popup = Popup(title="File Select", content=content, size_hint=(0.9, 0.9))
        self._popup.open()

    def load(self, path, _):
        self.dismiss_popup()

        if self.send_images_to_server(path):
            txtFName = self.ids['txtFName']
            txtFName.text = f'cur dir - {path}'
        else:
            self.show_pop_up("wrong folder format")

    def show_pop_up(self, msg: str):
        popupWindow = P()  # Create a new instance of the P classPopup
        # Create the popup window
        popupWindow.title = msg
        popupWindow.open()  # show the popup

    def send_images_to_server(self, folder_root):
        imgs = []
        for fname in os.listdir(folder_root):
            name_path = os.path.join(folder_root, fname)
            if not os.path.isdir(name_path):
                return False

            for img in os.listdir(os.path.join(name_path)):
                if not img.endswith('jpg') and not img.endswith('png'):
                    return False
                img_path = os.path.join(name_path, img)
                imgs.append((fname, np.asarray(PIL.Image.open(img_path))))

        if not imgs:
            return False

        frame_bytes = cam_streamer.pickle_data(['imgs', imgs])
        self.transport.write(frame_bytes)
        self.got_response = False
        return True

    def dismiss_popup(self):
        self._popup.dismiss()



