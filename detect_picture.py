from recognition import DetectFace
from detection import SSDDetection, Mtcnn
from multiprocessing.pool import ThreadPool


class DetectImage:
    def __init__(self, tolerance: float, increase_ratio: int = 0):
        # self.detection_model = SSDDetection(tolerance=tolerance, increase_ratio=increase_ratio)
        self.detection_model = Mtcnn(tolerance=tolerance, increase_ratio=increase_ratio)

    def detect(self, frame, face_detector: DetectFace):

        def detect_face_wrap(args):
            """ wrap detect_face func because in threads we want to save the index"""
            face, index = args
            try:
                m_name, m_score = face_detector.detect_face(face)
                return m_name, m_score, index
            except (ZeroDivisionError, ValueError)as e:
                return "error", "error", "error"

        # Find all the faces and face encodings in the current frame of video
        face_suggestion = self.detection_model.detect_picture(frame)
        faces = [frame[face_cor[0]: face_cor[2], face_cor[3]: face_cor[1]] for face_cor in face_suggestion]

        face_names = []
        scores = []
        face_locations = []

        if len(faces) > 0:
            pool = ThreadPool(len(faces))
            res = pool.map(detect_face_wrap, zip(faces, range(len(faces))))

            for match_name, match_score, i in res:
                if match_name != "error":
                # See if the face is a match for the known face(s)
                    face_names.append(match_name)
                    scores.append(match_score)
                    face_locations.append(face_suggestion[i])

        return face_locations, face_names, scores, len(face_suggestion)
