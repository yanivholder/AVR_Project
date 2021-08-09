import socketserver
import cv2

import camera_stream.camera_streamer as cam_streamer

HOST, PORT = "localhost", 9879


class MyTCPHandler(socketserver.BaseRequestHandler):

    def handle(self) -> None:
        # Receive the data from TCP socket
        frame = cam_streamer.get_frame_from_socket(self.request)
        print(f"{self.client_address[0]} sent a frame")

        # Manipulate the data
        updated_data = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Send the data back
        message = cam_streamer.pickle_data_for_packet(updated_data)
        self.request.sendall(message)


if __name__ == "__main__":

    # Create the server and bind it to localhost on port 9879
    with socketserver.TCPServer((HOST, PORT), MyTCPHandler) as server:
        # This will run forever until you interrupt the program
        # with Ctrl-C
        server.serve_forever()
