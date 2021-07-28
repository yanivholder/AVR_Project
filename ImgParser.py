import os
import recognition
import re


class Parser(object):
    def __init__(self, root: str):
        self.root = root

    def get_all_people(self):
        img_dict = {}
        for d in os.scandir(self.root):
            if not d.is_dir():
                continue
            img_dict[d.name] = self.scan_person_folder(d.path)
        return img_dict

    def get_person(self, folder_name: str):
        if folder_name in os.scandir():
            try:
                return {folder_name: self.scan_person_folder(os.path.join(os.getcwd(), folder_name))}
            except FileNotFoundError:
                raise Exception('no such folder {}'.format(folder_name))

    def scan_person_folder(self, folder: str):
        imgs = [os.path.join(folder, f) for f in os.listdir(folder) if re.match(r'.*\.(jpg|jpeg|png)', f, flags=re.I)]
        return [self.embad_img(path) for path in imgs]

    @staticmethod
    def embad_img(path: str):
        image = recognition.load_image_file(path)
        return recognition.face_encodings(image)[0]

