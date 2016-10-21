import cv2
import numpy as np

if __name__ =='__main__':

    filename = '/home/ply/leftout.jpg'
    img = cv2.imread(filename)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    gray = np.float32(gray)
    dst = cv2.cornerHarris(gray,3,3,0.04)

    dst = cv2.dilate(dst,None)

    img[dst>0.01*dst.max()]=[0,0,255]

    cv2.imwrite('/home/ply/dst.jpg',img)

