import cv2
import numpy as np

import Image_path_load

"""
HSV空間の説明
H　色相
S　彩度　色の鮮やかさ。 64~255
V 明度　色の明るさ

サビの形だけで構造を判断したい。
"""
#画像のファイルがあるパスを指定する。
DATA_PATH = "/home/cvmlab/Desktop/前処理/data"
DATA_RESULT_PATH = "/home/cvmlab/Desktop/前処理/result/"

def detect_red_color(img):
    # HSV色空間に変換
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # 赤色のHSVの値域1
    #hsv_min = np.array([0,64,0]) 赤の領域だけ
    hsv_min = np.array([0,0,0])
    hsv_max = np.array([30,255,255])
    mask1 = cv2.inRange(hsv, hsv_min, hsv_max)

    # 赤色のHSVの値域2
    # hsv_min = np.array([150,64,0]) 赤の領域だけ
    hsv_min = np.array([150,0,0])
    hsv_max = np.array([179,255,255])
    mask2 = cv2.inRange(hsv, hsv_min, hsv_max)

    # 赤色領域のマスク（255：赤色、0：赤色以外）    
    mask = mask1 + mask2

    # マスキング処理
    masked_img = cv2.bitwise_and(img, img, mask=mask)

    return mask, masked_img

#画像の名前を所得する関数を実行している。
image_path = Image_path_load.image_path_load(data_path=DATA_PATH).data_image_name()

for path in image_path:
    img = cv2.imread(DATA_PATH+ "/" + path)
    #ここに処理を書く
    img,masked_img = detect_red_color(img)
    cv2.imwrite((DATA_RESULT_PATH + path), masked_img)