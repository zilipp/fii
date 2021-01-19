import cv2
import numpy as np
import os
from pathlib import Path

# global variables
_out_root_dir = Path(os.path.dirname(os.path.abspath(__file__)))
_root_dir = _out_root_dir.parent
photo_dir = os.path.join(_root_dir, 'data', 'photos')
show_figure = True


def task1():
    # read the img
    img_file = os.path.join(photo_dir, 'img.jpg')
    img = cv2.imread(img_file)
    if show_figure:
        cv2.imshow("org", img)

    # gray image
    before = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if show_figure:
        cv2.imshow("gray", before)

    # rotated image
    after = rotate_image(before, -15)
    if show_figure:
        cv2.imshow("rotated", after)

    return img, before, after


# rotate it 15 clockwise
def rotate_image(image, angle):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result


def task2(img, img1, img2):
    """
    :param img: org image
    :param img1: gray image
    :param img2: rotated image
    :return: degree of rotation
    """

    height, width = img2.shape

    # Create ORB detector
    orb_detector = cv2.ORB_create(5000)

    # Find keypoints and descriptors.
    keypoints1, d1 = orb_detector.detectAndCompute(img1, None)
    keypoints2, d2 = orb_detector.detectAndCompute(img2, None)

    # Match features between the two images.
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # Match the two sets of descriptors.
    matches = matcher.match(d1, d2)

    # Sort matches on the basis of their Hamming distance.
    matches.sort(key=lambda x: x.distance)

    # Take the top 90 % matches forward.
    matches = matches[:int(len(matches) * 90)]
    no_of_matches = len(matches)

    # Define empty matrices of shape no_of_matches * 2.
    p1 = np.zeros((no_of_matches, 2))
    p2 = np.zeros((no_of_matches, 2))

    for i in range(len(matches)):
        p1[i, :] = keypoints1[matches[i].queryIdx].pt
        p2[i, :] = keypoints2[matches[i].trainIdx].pt

    # Find the homography matrix.
    homography, mask = cv2.findHomography(p1, p2, cv2.RANSAC)

    # Use this matrix to transform the colored image wrt the reference image.
    transformed_img = cv2.warpPerspective(img, homography, (width, height))

    if show_figure:
        cv2.imshow('using transform matrix', transformed_img)
        cv2.waitKey(0)

    return homography


if __name__ == "__main__":
    # task1: rotate 15 degree
    img, before, after = task1()

    # task2: find rotation degree
    homography = task2(img, before, after)
    print("transformation matrix is {}".format(homography))

