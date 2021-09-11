import cv2
from colors_config import green, red, white, black


class BoxConfig:
    def __init__(self,
                 known_face_color=green,
                 unknown_face_color=red,
                 known_face_text=green,
                 unknown_face_text=white,
                 box_thickness=3):
        self.known_face_color = known_face_color
        self.unknown_face_color = unknown_face_color
        self.known_face_text = known_face_text
        self.unknown_face_text = unknown_face_text
        self.box_thickness = box_thickness


class Draw:
    def draw_faces(self, frame, box_config: BoxConfig, face_locations, face_names, scores, print_scores=True):
        # Label the results
        for (top, right, bottom, left), name, score in zip(face_locations, face_names, scores):
            if name:
                text = name
                if print_scores:
                    text += " {}".format(score)
                self._draw(frame, top, right, bottom, left,
                           text=text,
                           frame_color=box_config.known_face_color,
                           text_color=box_config.known_face_text,
                           box_thickness=box_config.box_thickness, )
            else:
                self._draw(frame, top, right, bottom, left,
                           text="",
                           frame_color=box_config.unknown_face_color,
                           text_color=box_config.unknown_face_text,
                           box_thickness=box_config.box_thickness, )

        return face_locations, face_names, scores

    @staticmethod
    def _draw(frame, top, right, bottom, left, text, frame_color, text_color, box_thickness):
        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), frame_color, box_thickness)

        # Draw a label with a name below the face
        # cv2.rectangle(frame, (left, bottom - 25), (right, bottom), frame_color)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, text, (int((left + right) / 2), bottom - 6), font, 0.5, text_color, 1)
