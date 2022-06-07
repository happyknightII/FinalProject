import cv2
import numpy as np

DOT_COLOR = (230, 230, 230)
LINE_COLOR = (220, 220, 220)
DOT_THICKNESS = 10
LINE_THICKNESS = 10
    
    
class ImageCropper:
    def __init__(self, dimensions, warp):
        self.dimensions = dimensions
        self.shape = (dimensions[2] - dimensions[0], dimensions[3] - dimensions[1])
        self.warpPoints = (warp[0], warp[1], (0, self.shape[1]), self.shape)

    def offset(self, pts):
        return pts[0] + self.dimensions[0], pts[1] + self.dimensions[1]

    def crop_image(self, src):
        return src[self.dimensions[1]:self.dimensions[3], self.dimensions[0]:self.dimensions[2]]

    def warp_image(self, src):
        pts1 = np.float32(self.warpPoints)

        pts2 = np.float32(((0, 0), (self.shape[0], 0), (0, self.shape[1]), (self.shape[0], self.shape[1])))
        matrix = cv2.getPerspectiveTransform(pts1, pts2)

        return cv2.warpPerspective(src, matrix, (self.shape[0], self.shape[1]))

    def display_crop_lines(self, src):
        for pts in self.warpPoints:
            cv2.circle(src, self.offset(pts), DOT_THICKNESS, DOT_COLOR, -1)

        cv2.line(src, self.offset(self.warpPoints[0]),
                 self.offset(self.warpPoints[1]), LINE_COLOR, LINE_THICKNESS)
        cv2.line(src, self.offset(self.warpPoints[0]),
                 self.offset(self.warpPoints[2]), LINE_COLOR, LINE_THICKNESS)
        cv2.line(src, self.offset(self.warpPoints[1]),
                 self.offset(self.warpPoints[3]), LINE_COLOR, LINE_THICKNESS)
        cv2.line(src, self.offset(self.warpPoints[2]),
                 self.offset(self.warpPoints[3]), LINE_COLOR, LINE_THICKNESS)

