import cv2
import numpy as np

path = cv2.data.haarcascades

# get facial classifiers
face_cascade = cv2.CascadeClassifier(path + "haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier(path + "haarcascade_eye.xml")
print("init")


def set_image_in_frame(cap, image_to_insert, escala=0.60):
    # get shape of gap
    gap = image_to_insert
    original_gap_h, original_gap_w, gap_channels = gap.shape

    # convert to gray
    gap_gray = cv2.cvtColor(gap, cv2.COLOR_BGR2GRAY)

    # create mask and inverse mask of gap
    ret, original_mask = cv2.threshold(gap_gray, 10, 255, cv2.THRESH_BINARY_INV)
    original_mask_inv = cv2.bitwise_not(original_mask)

    # read each frame of video and convert to gray
    img = cap
    img_h, img_w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # find faces in image using classifier
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # for every face found:
    for (x, y, w, h) in faces:
        # coordinates of face region
        face_w = w
        face_h = h
        face_x1 = x
        face_x2 = face_x1 + face_w
        face_y1 = y
        face_y2 = face_y1 + face_h

        # gap size in relation to face by scaling
        gap_width = int(1.5 * face_w)
        gap_height = int(gap_width * original_gap_h / original_gap_w)

        # setting location of coordinates of gap
        gap_x1 = face_x2 - int(face_w / 2) - int(gap_width / 2)
        gap_x2 = gap_x1 + gap_width
        gap_y1 = face_y1 - int(face_h * escala)
        gap_y2 = gap_y1 + gap_height

        # check to see if out of frame
        if gap_x1 < 0:
            gap_x1 = 0
        if gap_y1 < 0:
            gap_y1 = 0
        if gap_x2 > img_w:
            gap_x2 = img_w
        if gap_y2 > img_h:
            gap_y2 = img_h

        # Account for any out of frame changes
        gap_width = gap_x2 - gap_x1
        gap_height = gap_y2 - gap_y1

        # resize gap to fit on face
        gap = cv2.resize(gap, (gap_width, gap_height), interpolation=cv2.INTER_AREA)
        mask = cv2.resize(original_mask, (gap_width, gap_height), interpolation=cv2.INTER_AREA)
        mask_inv = cv2.resize(original_mask_inv, (gap_width, gap_height), interpolation=cv2.INTER_AREA)

        # take ROI for gap from background that is equal to size of gap image
        roi = img[gap_y1:gap_y2, gap_x1:gap_x2]

        # original image in background (bg) where gap is not
        roi_bg = cv2.bitwise_and(roi, roi, mask=mask)
        roi_fg = cv2.bitwise_and(gap, gap, mask=mask_inv)
        dst = cv2.add(roi_bg, roi_fg)

        # put back in original image
        img[gap_y1:gap_y2, gap_x1:gap_x2] = dst

        break
    return img
