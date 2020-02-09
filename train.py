#
# Created by mstark on January 19, 2019
#
# Copyright (c) 2019. Michael Stark. All Rights Reserved.
#


import sys
import os
import argparse
import math
import cv2
import numpy as np

from obj_loader import OBJ


WINDOW_SIZE = 2000


def extract_features(model_path):
    """ """

    img = cv2.imread(model_path, 0)

    # initiate orb detector
    orb = cv2.ORB_create()

    # find the keypoints with ORB
    kp = orb.detect(img, None)

    # compute the descriptors with ORB
    kp, des = orb.compute(img, kp)

    # draw only keypoints location, not size and orientation
    img2 = cv2.drawKeypoints(img, kp, img, color=(0, 255, 0), flags=0)
    cv2.namedWindow('keypoints', cv2.WINDOW_NORMAL)
    img2s = cv2.resize(img2, (WINDOW_SIZE, WINDOW_SIZE))

    cv2.imshow('keypoints', img2s)
    cv2.waitKey(0)


def feature_matching(model_path, scene_path, min_matches=15):
    """ """

    cap = cv2.imread(scene_path, 0)
    model = cv2.imread(model_path, 0)

    # ORB keypoint detector
    orb = cv2.ORB_create()
    # create brute force matcher object
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    # compute model keypoints and its descriptors
    kp_model, des_model = orb.detectAndCompute(model, None)
    # compute scene keypoints and its descriptors
    kp_frame, des_frame = orb.detectAndCompute(cap, None)
    # match frame descriptors with model descriptors
    matches = bf.match(des_model, des_frame)
    # sort them in the order of their distance
    matches = sorted(matches, key=lambda x: x.distance)

    if len(matches) > min_matches:
        pass
        # draw first matches
        # cap = cv2.drawMatches(model, kp_model, cap, kp_frame, matches[:min_matches], 0, flags=2)

        # cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
        # cap_resized = cv2.resize(cap, (WINDOW_SIZE, WINDOW_SIZE))

        # cv2.imshow('frame', cap_resized)
        # cv2.waitKey(0)
    else:
        print(
            "Not enough matches have been found. - {}/{}".format(len(matches), min_matches))

    return (cap, model, kp_model, kp_frame, matches)


def ransac(kp_frame, matches):
    """ """

    # assuming matches stores the matches found and
    # returned by bf.match(des_model, des_frame)
    # differenciate between source points and destination points
    src_pts = np.float32([kp_model[m.queryIdx].pt for m in matches]).reshape(
        -1, 1, 2)
    dst_pts = np.float32([kp_frame[m.trainIdx].pt for m in matches]).reshape(
        -1, 1, 2)
    # compute Homography
    homography, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    return (src_pts, dst_pts, homography, mask)


def draw_homography_rect(M):
    """ """
    # Draw a rectangle that marks the found model in the frame
    h, w = model.shape
    pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(
        -1, 1, 2)
    # project corners into frame
    dst = cv2.perspectiveTransform(pts, M)
    # connect them with lines
    img2 = cv2.polylines(cap, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)

    # cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
    # cap_resized = cv2.resize(cap, (WINDOW_SIZE, WINDOW_SIZE))

    # cv2.imshow('frame', cap_resized)
    # cv2.waitKey(0)


def projection_matrix(camera_parameters, homography):
    """ 
    From the camera calibration matrix and the estimated homography, 
    compute the 3D projection matrix.

    Parameters
    ----------
    camara_parameters : dict

    homography : object
    """

    # compute rotation along the x and y axis as well as the translation
    homography = homography * (-1)
    rot_and_transl = np.dot(np.linalg.inv(camera_parameters), homography)
    col_1 = rot_and_transl[:, 0]
    col_2 = rot_and_transl[:, 1]
    col_3 = rot_and_transl[:, 2]

    # normalize vectors
    l = math.sqrt(np.linalg.norm(col_1, 2) * np.linalg.norm(col_2, 2))
    rot_1 = col_1 / l
    rot_2 = col_2 / l
    translation = col_3 / l
    # compute the orthonormal basis
    c = rot_1 + rot_2
    p = np.cross(rot_1, rot_2)
    d = np.cross(c, p)
    rot_1 = np.dot(c / np.linalg.norm(c, 2) + d /
                   np.linalg.norm(d, 2), 1 / math.sqrt(2))
    rot_2 = np.dot(c / np.linalg.norm(c, 2) - d /
                   np.linalg.norm(d, 2), 1 / math.sqrt(2))
    rot_3 = np.cross(rot_1, rot_2)
    # finally, compute the 3D projection matrix from the model to the current frame
    projection = np.stack((rot_1, rot_2, rot_3, translation)).T
    return np.dot(camera_parameters, projection)


def render(img, projection, color=False):
    """
    Render a loaded obj model into the current video frame.
    """
    vertices = obj.vertices
    scale_matrix = np.eye(3) * 3
    h, w = model.shape

    for face in obj.faces:
        face_vertices = face[0]
        points = np.array([vertices[vertex - 1] for vertex in face_vertices])
        points = np.dot(points, scale_matrix)
        # render model in the middle of the reference surface. To do so,
        # model points must be displaced
        points = np.array([[p[0] + w / 2, p[1] + h / 2, p[2]] for p in points])
        dst = cv2.perspectiveTransform(points.reshape(-1, 1, 3), projection)
        imgpts = np.int32(dst)
        if color is False:
            cv2.fillConvexPoly(img, imgpts, (137, 27, 211))
        else:
            color = hex_to_rgb(face[-1])
            color = color[::-1]  # reverse
            cv2.fillConvexPoly(img, imgpts, color)

    return img


def hex_to_rgb(hex_color):
    """
    Helper function to convert hex strings to RGB
    """
    hex_color = hex_color.lstrip('#')
    h_len = len(hex_color)
    return tuple(
        int(hex_color[i:i + h_len // 3], 16)
        for i in range(0, h_len, h_len // 3))


def main(options):

    global orb, obj, cap, bf, model, kp_model, des_model

    # create ORB keypiont detector
    orb = cv2.ORB_create()

    # create BFMatcher object based on hamming distance
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # load reference surface to match in live feed
    model = cv2.imread(options.model, 0)

    # compute model keypoints and its descriptors
    kp_model, des_model = orb.detectAndCompute(model, None)

    # camera params
    camera_parameters = np.array([[800, 0, 320], [0, 800, 240], [0, 0, 1]])

    homography = None

    # load 3d model
    if options.obj is not None:
        obj = OBJ(options.obj, swapyz=True)

    # init video capture
    cap = cv2.VideoCapture(0)

    while True:
        global ret, frame
        ret, frame = cap.read()
        if not ret:
            print("Unable to capture video")
            return

        # find and draw keypoints of the frame
        kp_frame, des_frame = orb.detectAndCompute(frame, None)
        # match frame descriptors with model descriptors
        matches = bf.match(des_model, des_frame)
        # sort by distance
        # the lower the distance, the better the match
        matches = sorted(matches, key=lambda x: x.distance)

        if len(matches) > options.min_match:
            src_pts = np.float32(
                [kp_model[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
            dst_pts = np.float32(
                [kp_frame[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

            # src_pts, dst_pts, homography, mask = ransac(kp_frame, matches)
            # compute Homography
            homography, mask = cv2.findHomography(
                src_pts, dst_pts, cv2.RANSAC, 5.0)

            # Draw a rectangle that marks the found model in the frame
            h, w = model.shape
            pts = np.float32(
                [[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
            # project corners into frame
            dst = cv2.perspectiveTransform(pts, homography)
            # connect them with lines
            frame = cv2.polylines(
                frame, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)

            if homography is not None:
                try:
                    # obtain 3D projection matrix from homography matrix and camera parameters
                    projection = projection_matrix(
                        camera_parameters, homography)
                    # project cube or model
                    frame = render(frame, projection, False)
                    #frame = render(frame, model, projection)
                except:
                    pass

            # draw matches
            # frame = cv2.drawMatches(
            #     model, kp_model, frame, kp_frame, matches[:options.min_match], 0, flags=2)

            cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
            frame_resized = cv2.resize(frame, (WINDOW_SIZE, WINDOW_SIZE))
            cv2.imshow('frame', frame_resized)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            print(
                "Not enough matches found. {}/{}".format(len(matches), options.min_match))

    cap.release()
    cv2.destroyAllWindows()

    return 0

    # feature_matching(model_path=options.model, scene_path=options.scene, min_matches=options.min_match)


def parse_args(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", type=str,
                        help="Input file of image to train on.")
    parser.add_argument("--scene", type=str, default=None,
                        help="Input file path of scene.")

    parser.add_argument("--obj", type=str, default=None,
                        help="Input file for 3D object (format: obj) to render.")

    parser.add_argument("--min_match", type=int, default=15,
                        help="Minimum amount of feature matches to consider the detection as a match.")

    return parser.parse_args(args=argv)


if __name__ == "__main__":
    argv = sys.argv[1:]
    if len(argv) == 0:
        argv = ["-h"]

    main(parse_args(argv))
