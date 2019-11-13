import numpy as np
import cv2
import matplotlib.pyplot as plt


def get_document_corners(img):
    b, g, r = cv2.split(img)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(64, 64))
    b = clahe.apply(b)
    g = clahe.apply(g)
    r = clahe.apply(r)
    img = cv2.merge((b, g, r))

    mask = np.zeros(img.shape[:2], np.uint8)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    rect = (10, 10, img.shape[1] - 10, img.shape[0] - 10)
    cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)

    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask2 = cv2.morphologyEx(mask2, cv2.MORPH_CLOSE, kernel)
    mask2 = cv2.morphologyEx(mask2, cv2.MORPH_OPEN, kernel)
    contours, hierarchy = cv2.findContours(mask2, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    contour_img = np.zeros(img.shape[0:2])
    cv2.drawContours(contour_img, contours, -1, 255, 3)
    max_contour_area = 0
    max_contour = contours[0]
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > max_contour_area:
            max_contour = cnt
            max_contour_area = area
    epsilon = 0.1 * cv2.arcLength(max_contour, True)
    approx = cv2.approxPolyDP(max_contour, epsilon, True)
    return approx


def get_ordered_corners(corners):
    sums = []
    for p in corners:
        sums.append(sum(p[0]))
        print(sum(p[0]))

    sums = np.array(sums)
    top_left = corners[np.argmin(sums)]
    bottom_right = corners[np.argmax(sums)]
    mask = np.array([True, True, True, True])
    mask[[np.argmin(sums), np.argmax(sums)]] = False

    top_right = corners[mask][np.argmin(corners[mask][:, :, 1])]
    bottom_left = corners[mask][np.argmax(corners[mask][:, :, 1])]
    print("top left:")
    print(top_left)
    print("top right:")
    print(top_right)
    print("bottom_left:")
    print(bottom_left)
    print("bottom_right")
    print(bottom_right)
    return np.array([top_left, top_right, bottom_right, bottom_left])


def scan_document(img_to_scan, target_width, target_height_to_width_ratio):
    target_height = round(500 * target_height_to_width_ratio)
    target_points = np.array([[0, 0], [500, 0], [500, target_height], [0, target_height]])
    corners = get_document_corners(img_to_scan)
    ordered_corners = get_ordered_corners(corners)

    if len(corners) != 4:
        print("WARNING: could not detect four corners")
        return img_to_scan

    h, status = cv2.findHomography(ordered_corners, target_points)
    im_h = cv2.warpPerspective(img_to_scan, h, (target_width, target_height))
    cv2.drawContours(img_to_scan, corners, -1, (255, 0, 0), 20)

    return im_h


if __name__ == "__main__":
    TARGET_WIDTH = 500
    DIN_A4_HEIGHT_TO_WIDTH_RATIO = 297 / 210
    orig_img = cv2.imread('scanned-form.jpg')
    scanned_doc = scan_document(orig_img, TARGET_WIDTH, DIN_A4_HEIGHT_TO_WIDTH_RATIO)
    plt.imshow(scanned_doc[:, :, ::-1])
    plt.show()
