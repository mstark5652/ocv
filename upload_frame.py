#
# Created by Michael Stark on February 06, 2020
#
# Copyright (c) 2020. Michael Stark. All Rights Reserved.
#

import sys
import os
import argparse
import cv2
import requests
import base64

import time

WINDOW_SIZE = 2500

# file download url: https://faces.azurewebsites.net/documents/download/mix_ebc6482b-b335-4e1f-866d-bf718d938f43


def upload_photo(frame):
    headers = {'Content-Type': 'application/json'}
    retval, buffer = cv2.imencode('.jpg', frame)
    image_data = "\"%s\"" % (base64.b64encode(buffer).decode("utf-8"))
    if image_data is not None and len(image_data) > 10:
        with open('./tmp.txt', 'w') as file:
            file.write(image_data)
            file.close()
        url = 'https://faces.azurewebsites.net/documents/mix_ebc6482b-b335-4e1f-866d-bf718d938f43/savedocument'
        response = requests.post(url, headers=headers, data=image_data)
        print('sent photo')
        print(response.status_code)
        print(response.text)


def main(options):

    global cap

    # init video capture
    cap = cv2.VideoCapture(0)
    count = 0

    while True:
        global ret, frame
        ret, frame = cap.read()
        if not ret:
            print("Unable to capture video")
            return
        cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
        frame_resized = cv2.resize(frame, (WINDOW_SIZE, WINDOW_SIZE))
        cv2.imshow('frame', frame_resized)

        count = count + 1
        if count % 100_000_000_000_000:
            upload_photo(frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def parse_args(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument("--min_match", type=int, default=15,
                        help="Minimum amount of feature matches to consider the detection as a match.")

    return parser.parse_args(args=argv)


if __name__ == "__main__":
    argv = sys.argv[1:]
    if len(argv) == 0:
        argv = ["-h"]

    main(parse_args(argv))
