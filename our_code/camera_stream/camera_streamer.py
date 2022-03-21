import cv2
import pickle
import struct

payload_size = struct.calcsize("Q")
RECV_SIZE = 4 * 1024  # 4 KB


def gen_frame():
    # Define a video capture object
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise IOError("Cannot open webcam")

    # Capture one frame
    ret, frame = cap.read()
    if ret is False:
        # TODO think what to do here
        pass

    cap.release()
    return frame


def unpickle_data(data):

    packed_msg_size = data[:payload_size]
    data = data[payload_size:]
    msg_size = struct.unpack("Q", packed_msg_size)[0]

    # while len(data) < msg_size:
    #     data += sock.recv(RECV_SIZE)

    frame_data = data[:msg_size]
    data = data[msg_size:]
    frame = pickle.loads(frame_data)
    return frame


def pickle_data(data):
    pickled_data = pickle.dumps(data)
    message = struct.pack('Q', len(pickled_data)) + pickled_data
    return message

# cap = cv2.VideoCapture(0)
# while(cap.isOpened()):
#     ret, frame = cap.read()
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     cv2.imshow('frame', gray)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
#
# cap.release()
# cv2.destroyAllWindows()