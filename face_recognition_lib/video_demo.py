from our_code import recognition
import cv2
from face_recognition_lib.utils import Detect

# This is a demo of running face recognition on a video file and saving the results to a new video file.
#
# PLEASE NOTE: This example requires OpenCV (the `cv2` library) to be installed only to read from your webcam.
# OpenCV is *not* required to use the face_recognition library. It's only required if you want to run this
# specific demo. If you have trouble installing it, try any of the other demos that don't require it instead.

# Open the input movie file
# input_movie = cv2.VideoCapture("imgs/hamilton_clip.mp4")
input_movie = cv2.VideoCapture("imgs/adi_yaniv.mp4")
length = int(input_movie.get(cv2.CAP_PROP_FRAME_COUNT))
width = (int(input_movie.get(cv2.CAP_PROP_FRAME_WIDTH)))
height = (int(input_movie.get(cv2.CAP_PROP_FRAME_HEIGHT)))
fps = (int(input_movie.get(cv2.CAP_PROP_FPS)))

# Create an output movie file (make sure resolution/frame rate matches input video!)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
output_movie = cv2.VideoWriter('output.avi', fourcc, fps, (width, height))

# Load some sample pictures and learn how to recognize them.
lmm_image = recognition.load_image_file("imgs/lin-manuel-miranda.png")
lmm_face_encoding = recognition.face_encodings(lmm_image)[0]

al_image = recognition.load_image_file("imgs/alex-lacamoire.png")
al_face_encoding = recognition.face_encodings(al_image)[0]

yaniv_image = recognition.load_image_file("imgs/yaniv.png")
yaniv_face_encoding = recognition.face_encodings(yaniv_image)[0]


yaniv_image2 = recognition.load_image_file("imgs/yaniv2.png")
yaniv_face_encoding2 = recognition.face_encodings(yaniv_image2)[0]


adi_image = recognition.load_image_file("imgs/adi.png")
adi_face_encoding = recognition.face_encodings(adi_image)[0]

known_faces = [
    lmm_face_encoding,
    al_face_encoding,
    yaniv_face_encoding,
    adi_face_encoding,
    yaniv_face_encoding2
]


known_names = ["Lin-Manuel Miranda", "Alex Lacamoire", "yaniv", "adi", "yaniv"]

# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
frame_number = 0

detector = Detect(known_faces, known_names)

while True:
    # Grab a single frame of video
    ret, frame = input_movie.read()
    frame_number += 1

    # Quit when the input video file ends
    if not ret or frame_number > 300:
        break

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_frame = frame[:, :, ::-1]

    # Find all the faces and face encodings in the current frame of video
    face_locations = recognition.face_locations(rgb_frame)
    face_encodings = recognition.face_encodings(rgb_frame, face_locations)

    face_names = []
    for face_encoding in face_encodings:
        # See if the face is a match for the known face(s)
        match = recognition.compare_faces(known_faces, face_encoding, tolerance=0.6)

        # If you had more than 2 faces, you could make this logic a lot prettier
        # but I kept it simple for the demo
        # name = None
        # if match[0]:
        #     name = "Lin-Manuel Miranda"
        # elif match[1]:
        #     name = "Alex Lacamoire"
        # elif match[2]:
        #     name = "yaniv"
        # elif match[3]:
        #     name = "adi"
        # elif match[4]:
        #     name = "yaniv"

        name = detector.get_closest(face_encoding)

        face_names.append(name)

    # Label the results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        if not name:
            pass

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 25), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)

    # Write the resulting image to the output video file
    print("Writing frame {} / {}".format(frame_number, length))
    output_movie.write(frame)

# All done!
input_movie.release()
cv2.destroyAllWindows()
