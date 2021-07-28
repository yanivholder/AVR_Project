import face_recognition
from recognition import DetectFace
from detection import SSDDetection


class DetectImage:
    def __init__(self, model: str = 'hog', number_of_times_to_upsample: int = 2):
        self.model = model
        self.number_of_times_to_upsample = number_of_times_to_upsample
        self.detection_model = SSDDetection(tolerance=0.7)

    def detect(self, frame, face_detector: DetectFace):
        # Find all the faces and face encodings in the current frame of video
        # face_locations = face_recognition.face_locations(frame, model=self.model, number_of_times_to_upsample=self.number_of_times_to_upsample)
        face_locations = self.detection_model.detect_picture(frame)
        face_encodings = face_recognition.face_encodings(frame, face_locations)

        face_names = []
        scores = []
        for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
            match_name, match_score = face_detector.detect_face(face_encoding)
            face_names.append(match_name)
            scores.append(None if match_score is None else 100 - int(100 * match_score))

        return face_locations, face_names, scores

