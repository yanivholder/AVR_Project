import abc
import logging
import os
import pickle
import pandas as pd
from tqdm import tqdm
from os import path
from abc import ABC

from deepface.commons import distance as dst
from deepface import DeepFace
from deepface.basemodels import Facenet512

from our_code.ImgParser import Parser
from our_code.server_config import recognition_threshold


class DetectFace:
    def __init__(self, img_path):
        self.img_path = img_path

    @abc.abstractmethod
    def detect_face(self, target_face) -> (str, int):
        pass


class FaceRecognitionModel(DetectFace, ABC):
    def __init__(self, img_path, tolerance: float = 0.6):
        super().__init__(img_path)
        self.known_faces = Parser(self.img_path).get_all_people()
        self.tolerance = tolerance

    def detect_face(self, target_face) -> (str, float):
        best_score = None
        best_name = None
        for name, imgs in self.known_faces.items():
            score = self.best_score_for_person(imgs, target_face)
            if best_score is None or score < best_score:
                best_score = score
                best_name = name
        if best_score < self.tolerance:
            return best_name, best_score
        return None, None

    # @staticmethod
    # def best_score_for_person(pictures: List, target_face):
    #     scores = recognition.face_distance(pictures, target_face)
    #     return np.amin(scores)


class DeepFaceModel(DetectFace, ABC):
    def __init__(self,
                 img_path,
                 distance_metric: str,
                 model_name: str = 'Facenet512',
                 detector_backend: str = 'mtcnn'
                 ):
        super().__init__(img_path)
        self.model_name = model_name
        self.model = Facenet512.loadModel()
        self.detector_backend = detector_backend
        self.distance_metric = distance_metric
        self.representation_path = "{}/representations_{}_{}.pkl".format(img_path, self.model_name, self.detector_backend)
        self.crate_representation_from_model()

    def detect_face(self, target_face) -> (str, float):
        return self.deep_face_find_rewrite(target_face)

    def deep_face_find_rewrite(self, img_path):

        if not os.path.isfile(self.representation_path):
            return None, None

        representations = pickle.load(open(self.representation_path, "rb"))
        df = pd.DataFrame(representations, columns=["identity", "%s_representation" % (self.model_name)])

        # we can enforce detection by throwing Value error if there is no face
        target_representation = DeepFace.represent(img_path=img_path,
                                                   model_name=self.model_name,
                                                   model=self.model,
                                                   enforce_detection=False,
                                                   detector_backend=self.detector_backend,
                                                   align=True,
                                                   )

        distances = []
        for index, instance in df.iterrows():
            source_representation = instance["%s_representation" % (self.model_name)]
            if self.distance_metric == 'cosine':
                distance = dst.findCosineDistance(source_representation, target_representation)
            elif self.distance_metric == 'euclidean':
                distance = dst.findEuclideanDistance(source_representation, target_representation)
            else: # using l2
                distance = dst.findEuclideanDistance(dst.l2_normalize(source_representation),
                                                 dst.l2_normalize(target_representation))

            distances.append(distance)

        df["score"] = distances
        threshold = dst.findThreshold(self.model_name, self.distance_metric)
        best = df.nsmallest(1, "score")

        # threshold is a const from the deepface library, we add a const to improve
        if float(best['score']) < threshold + recognition_threshold:
            return list(best['identity'])[0], float(best['score'])
        else:
            return None, None

    def crate_representation_from_model(self, representation_exists_ok=False):

        if path.exists(self.representation_path) and not representation_exists_ok:
            logging.warning("WARNING: Representations for images in ", self.img_path, " folder were previously stored in ", self.representation_path,
                  ". If you added new instances after this file creation, then please delete this file and call find function again. It will create it again.")
            return

        # create representation.pkl from scratch
        employees = []

        for r, d, f in os.walk(self.img_path):  # r=root, d=directories, f = files
            for file in f:
                if ('.jpg' in file.lower()) or ('.png' in file.lower()):
                    exact_path = os.path.join(r, file)
                    employees.append(exact_path)

        if len(employees) > 0:

            # ------------------------
            # find representations for db images

            representations = []

            pbar = tqdm(range(0, len(employees)), desc='Finding representations')

            # for employee in employees:
            for index in pbar:
                employee = employees[index]
                instance = []
                instance.append(employee.split(os.sep)[-2])
                representation = DeepFace.represent(img_path=employee,
                                                    model_name=self.model_name,
                                                    model=self.model,
                                                    enforce_detection=False,
                                                    detector_backend=self.detector_backend,
                                                    align=True
                                                    )

                instance.append(representation)
            # -------------------------------

                representations.append(instance)

            f = open(self.representation_path, "wb")
            pickle.dump(representations, f)
            f.close()

        print("Representations stored in ", self.img_path, "/", self.representation_path,
              " file. Please delete this file when you add new identities in your database.")


