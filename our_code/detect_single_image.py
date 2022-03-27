import argparse

import cv2
import matplotlib.pyplot as plt

from our_code import server_config
from our_code.detection import SSDDetection, Mtcnn
from our_code.draw_faces import BoxConfig, Draw


def main(args):
    detector = Mtcnn(args.tolerance, increase_ratio=0)
    box_config = BoxConfig(box_thickness=server_config.box_thickness)
    box_drawer = Draw()
    img = cv2.imread(args.input_image)
    faces = detector.detect_picture(img)
    box_drawer.draw_faces(img, box_config, faces, [None] * len(faces), [None] * len(faces), print_scores=False)
    cv2.imwrite(args.output_path, img)


if __name__ == '__main__':
    # Parse the command line arguments
    prog = argparse.ArgumentParser()
    prog.add_argument('--input_image', '-i', type=str, help='path to image')
    prog.add_argument('--output_path', '-o', type=str, help='path to image output directory.')
    prog.add_argument('--tolerance', '-t', type=float, default=0.85,  help='path to image output directory.')
    args = prog.parse_args()

    main(args)
