import cv2
import numpy as np
from numpy.lib.type_check import imag

import Image_path_load

#画像のファイルがあるパスを指定する。
DATA_PATH = "/home/cvmlab/Desktop/前処理/data"
DATA_RESULT_PATH = "/home/cvmlab/Desktop/前処理/result/"

#クラスター数を指定する
K = 50

#画像の名前を所得する関数を実行している。
image_path = Image_path_load.image_path_load(data_path=DATA_PATH).data_image_name()

def detect_red_color(img):
    # HSV色空間に変換
    #hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # 赤色のHSVの値域1
    #hsv_min = np.array([0,64,0]) 赤の領域だけ
    hsv_min = np.array([0,0,0])
    hsv_max = np.array([30,255,255])
    mask1 = cv2.inRange(img, hsv_min, hsv_max)

    # 赤色のHSVの値域2
    # hsv_min = np.array([150,64,0]) 赤の領域だけ
    hsv_min = np.array([150,0,0])
    hsv_max = np.array([179,255,255])
    mask2 = cv2.inRange(img, hsv_min, hsv_max)

    # 赤色領域のマスク（255：赤色、0：赤色以外）    
    mask = mask1 + mask2

    # マスキング処理
    masked_img = cv2.bitwise_and(img, img, mask=mask)

    return mask, masked_img

def k_means_image(image,K):
    # ndarray(y,x,[B,G,R])を変形(y * x,[B,G,R])
    Z = image.reshape((-1,3))
    # float32に型変換
    Z = np.float32(Z)
    # 計算終了条件の設定。指定された精度(1.0)か指定された回数(10)計算したら終了
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    # K-Meansクラスタリングの実施
    ret,label,center=cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
    # 各クラスタの中心色リストcenterをunit8に型変換
    center = np.uint8(center)
    # 中心色リストcenterから分類結果labelの要素を取り出す
    res = center[label.flatten()]
    # 元画像の形状に変形
    res2 = res.reshape((image.shape))
    return res2,res

def k_means_image_HSV(image,K):
    # ndarray(y,x,[B,G,R])を変形(y * x,[B,G,R])
    ## HSV色空間に変換
    hsv1 = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv = hsv1.reshape((-1,3))
    # float32に型変換
    Z = np.float32(hsv)
    # 計算終了条件の設定。指定された精度(1.0)か指定された回数(10)計算したら終了
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    # K-Meansクラスタリングの実施
    ret,label,center=cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
    # 各クラスタの中心色リストcenterをunit8に型変換
    center = np.uint8(center)
    # 中心色リストcenterから分類結果labelの要素を取り出す
    res = center[label.flatten()]
    # 元画像の形状に変形
    res2 = res.reshape((image.shape))
    #HSVで茶色の領域を復活させる。
    re,res2 = detect_red_color(res2)
    res2[res2 != 0] = hsv1[res2 != 0]
    res2 = cv2.cvtColor(res2, cv2.COLOR_HSV2BGR)
    return res2,res

#mainの処理
for path in image_path:
    img = cv2.imread(DATA_PATH+ "/" + path)
    #ここに処理を書く
    #img,res = k_means_image(image=img,K=K)
    img, res = k_means_image_HSV(image=img,K=K)

    cv2.imwrite((DATA_RESULT_PATH + path), img)