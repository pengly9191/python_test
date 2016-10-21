import numpy as np

import cv2
import Image

def inver(image):

    size = (image.shape[:2])
    newImg = image.copy()

    for i in range(size[0]):
        for j in range(size[1]):
    #        newImg[i, j] = (255 - image[i, j][0], 255 - image[i, j][1], 255 - image[i, j][2])
    #  or another way
            newImg[i, j] = (255 - image[i, j,0], 255 - image[i, j,1], 255 - image[i, j,2])


    return newImg

def Mirror_image(image):

    UDImg = image.copy()
    LRImg = image.copy()
    AcrossImg = image.copy()
    h,w = image.shape[:2]
    for i in range(h):
        for j in range(w):
            #  one line prossess image
            #UDImg[w-j-1] = image[j]
            UDImg[w - j - 1,i] = image[j,i]
            LRImg[j,h - i - 1] = image[j,i]
            AcrossImg[h - 1 - i, w - 1 - j] = image[i, j]

    return UDImg,LRImg,AcrossImg


def transImg(src):

    h,w = src.shape[:2]
    x = 20
    y= 30
    size = src.shape
    size2 = (w+x,h+y,src.shape[2])
    dst = np.zeros(size, np.uint8)
    dst2 = np.zeros(size2, np.uint8)
    print dst2.shape[:2]

    dst[x:w,y:h]= src [0:w-x,0:h-y]
    dst2[x:w+x,y:y+h] = src

    cv2.imwrite("/home/ply/UDImg.jpg", dst);
    cv2.imwrite("/home/ply/LRImg.jpg", dst2);

    return dst,dst2


if __name__ == '__main__':

    image = cv2.imread("/home/ply/lena.jpg",1)
    emptyImage = np.zeros(image.shape,np.uint8)
    # inver image
    # newImg = inver(image)

    # mirror image
    # UDImg, LRImg,AcrossImg = Mirror_image(image)


    transImg(image)




