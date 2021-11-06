import socket
import cv2
from datetime import datetime

import camera_stream.camera_streamer as cam_streamer

HOST, PORT = "localhost", 9879


def main():
    # Define a video capture object
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise IOError("Cannot open webcam")

    counter = 1
    # Create a TCP socket (SOCK_STREAM means a TCP socket)
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:

        # Connect to the server
        sock.connect((HOST, PORT))
        while True:
            # Create the data that will be sent to the server

            # Capture one frame
            ret, frame = cap.read()
            if ret is False:
                # TODO: think what to do here
                pass

            # Send the data
            time = datetime.now()
            message = cam_streamer.pickle_data_for_packet(frame)
            sock.sendall(message)
            # logging.info("frame {}  time {} ".format(counter,  (datetime.now() - time).total_seconds()))

            # Receive data back from the server
            updated_data = cam_streamer.get_frame_from_socket(sock)
            (datetime.now() - time).total_seconds()
            # Show the frames in a screen.
            # To exit the screen press 'q'.
            cv2.imshow('updated_data', updated_data)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            counter += 1
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
