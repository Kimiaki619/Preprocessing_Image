"""
マスク画像を利用して一部の画素のみ処理を行う。
https://pystyle.info/opencv-mask-image/
"""
import cv2
import numpy as np

class mask_ave_image():
    def __init__(self,image,radius):
        self.image = image
        self.radius = radius

    def mask(self):
        # マスク画像を作成する。
        mask = np.full(self.image.shape[:2], 255, dtype=self.image.dtype)

        # 白い画像の中心に黒い円を描画する。
        cx = self.image.shape[1] // 2
        cy = self.image.shape[0] // 2
        mask = cv2.circle(mask, (cx, cy), self.radius, color=0, thickness=-1)
        return mask

    def main(self):
        mask = self.mask()
        #ガウシアンフィルタ
        blur = cv2.GaussianBlur(self.image,(5,5),0)
        #マスクのの値が０の画素は処理を行わないように元の画像の画素値に戻す。
        blur[mask == 0] = self.image[mask == 0]
        return blur
