import cv2
import os
from matplotlib import pyplot as plt

import Image_path_load
import mask_ave_image

#画像のファイルがあるパスを指定する。
DATA_PATH = "/home/cvmlab/Desktop/前処理/data"
DATA_RESULT_PATH = "/home/cvmlab/Desktop/前処理/result/"

#半径を決める
RADIUS = 80

#画像の名前を所得する関数を実行している。
image_path = Image_path_load.image_path_load(data_path=DATA_PATH).data_image_name()

for path in image_path:
    img = cv2.imread(DATA_PATH+ "/" + path)
    #ここに処理を書く
    img = mask_ave_image.mask_ave_image(image=img,radius=RADIUS).main()
    cv2.imwrite((DATA_RESULT_PATH + path), img)


img = cv2.imread(DATA_PATH+ "/" + path)
cv2.imwrite((DATA_RESULT_PATH + "ss.jpg"),mask_ave_image.mask_ave_image(img,radius=RADIUS).mask())