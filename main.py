# Imports
import cv2
import numpy as np
from exponentialMovingAverage import exponential_moving_average
from imageCropper import ImageCropper
from detectLines import detect_lines


SLIDER_BACKGROUND_WEIGHT = 0.4
SLIDER_FOREGROUND_WEIGHT = 0.8
font = cv2.FONT_HERSHEY_SIMPLEX
risk_y = 1000
confidence_x = 100
display_offset = 150
RESIZE_FACTOR = 0.5
imageCropper = ImageCropper((350, 700, 1000, 1080), ((255, 10), (520, 10)))
line_medians = []
video = cv2.VideoCapture("3686.mp4")
save = cv2.VideoWriter("output.mp4", cv2.VideoWriter_fourcc(*'MP4V'), 30, (1920, 1080))
a = 0.3
b = 5
while True:

    ret, frame = video.read()
    if ret:
        image = imageCropper.crop_image(frame)

        warpedImage = imageCropper.warp_image(image)
        detected = detect_lines(warpedImage, 30, 15)
        risk = 90 - detected[1]
        line_medians.append(a * risk ** 2 + b * risk)

        adjusted_risk = int(exponential_moving_average(np.array(line_medians), 5))
        slider_background = np.copy(frame)
        slider_foreground = np.copy(frame)

        imageCropper.display_crop_lines(slider_foreground)

        risk_pos = (display_offset + adjusted_risk * 3, risk_y)
        cv2.line(slider_background, (display_offset, risk_y), (frame.shape[1] - display_offset, risk_y),
                 (200, 200, 200), 25)
        cv2.line(slider_background, (display_offset, risk_y), risk_pos, (0, 255, adjusted_risk), 25)
        cv2.circle(slider_foreground, risk_pos, 30, (0, 255, adjusted_risk), -1)
        cv2.line(slider_background, (confidence_x, display_offset),
                 (confidence_x, frame.shape[0] - display_offset),
                 (200, 200, 200), 25)

        confidence_pos = (confidence_x, frame.shape[0] - display_offset - detected[0] * 7)
        cv2.line(slider_foreground, confidence_pos,
                 (confidence_x, frame.shape[0] - display_offset),
                 (255, 0, 0), 25)
        cv2.circle(slider_foreground, confidence_pos, 30, (255, 0, 0), -1)
        cv2.putText(slider_foreground, str(detected[0]), confidence_pos, font, 5, (0, 0, 0), 15)
        cv2.putText(slider_foreground, str(detected[0]), confidence_pos, font, 5, (255, 255, 255), 10)

        out = cv2.addWeighted(cv2.addWeighted(
            frame, 1,
            slider_background, SLIDER_BACKGROUND_WEIGHT, 1), 1,
            slider_foreground, SLIDER_FOREGROUND_WEIGHT, 1)
        cv2.imshow('Result',
                   cv2.resize(out, (int(frame.shape[1] * RESIZE_FACTOR), int(frame.shape[0] * RESIZE_FACTOR))))
        save.write(out)
    if cv2.waitKey(1) == 27:
        break
video.release()
save.release()
cv2.destroyAllWindows()
