"""
dataファイルに入っている画像の名前を取得するファイル
配列で帰ってくる
"""
import cv2
import os

class image_path_load():
    def __init__(self,data_path):
        self.data_path = data_path
        

    def data_image_name(self):
        path_image = os.listdir(self.data_path)
        image = []

        for file in path_image:
            _, jpg = os.path.splitext(file)
            if jpg == ".jpg":
                image.append(file)

        return image