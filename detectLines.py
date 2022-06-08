import cv2
import numpy as np


rho = 1
theta = np.pi / 180
threshold = 20
blur = (3, 7)
brightness = 75
contrast = 55


def apply_brightness_contrast(input_img, brightness=0, contrast=0):
    if brightness != 0:
        if brightness > 0:
            shadow = brightness
            highlight = 255
        else:
            shadow = 0
            highlight = 255 + brightness
        alpha_b = (highlight - shadow) / 255
        gamma_b = shadow

        buf = cv2.addWeighted(input_img, alpha_b, input_img, 0, gamma_b)
    else:
        buf = input_img.copy()

    if contrast != 0:
        f = 131 * (contrast + 127) / (127 * (131 - contrast))
        alpha_c = f
        gamma_c = 127 * (1 - f)

        buf = cv2.addWeighted(buf, alpha_c, buf, 0, gamma_c)

    return buf


def detect_lines(src, min_line_length, max_line_gap):
    img = np.copy(src)
    blurred_img = cv2.GaussianBlur(src, blur, 2)
    bright_img = apply_brightness_contrast(blurred_img, brightness, contrast)
    grey_img = cv2.cvtColor(bright_img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(grey_img, 20, 50)

    lines = cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]),
                            min_line_length, max_line_gap)
    num_lines = 0
    accepted_lines = [90]
    if np.any(lines):
        for line in lines:
            for x1, y1, x2, y2 in line:
                if (x2 - x1) == 0:
                    accepted_lines.append(90)
                    num_lines += 1
                else:
                    angle = np.arctan2(abs(y2 - y1), abs(x2 - x1)) / np.pi * 180
                    if angle > 55:

                        num_lines += 1
                        accepted_lines.append(angle)
    return num_lines, np.median(accepted_lines)
