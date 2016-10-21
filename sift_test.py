from util import *

import cv2

def FFT(image,flag = 0):
    h, w = image.shape[:2]
    iTmp = np.zeros(image.shape,np.int32)
    cv2.Convert(image,iTmp)
    iMat = cv2.CreateMat(h,w,cv2.CV_32FC2)
    mFFT = cv2.CreateMat(h,w,cv2.CV_32FC2)
    for i in range(h):
        for j in range(w):
            if flag == 0:
                num = -1 if (i+j)%2 == 1 else 1
            else:
                num = 1
            iMat[i,j] = (iTmp[i,j]*num,0)
    cv2.DFT(iMat,mFFT,cv2.CV_DXT_FORWARD)
    return mFFT

def FImage(mat):
    w = mat.cols
    h = mat.rows
    size = (w,h)
    iAdd = cv2.CreateImage(size,cv2.IPL_DEPTH_8U,1)
    for i in range(h):
        for j in range(w):
            iAdd[i,j] = mat[i,j][1]/h + mat[i,j][0]/h
    return iAdd


if __name__ == '__main__':

    image = cv2.imread('/home/ply/lena.jpg', 0)

    for i in range(200):
        temp_x= np.random.randint(0,image.shape[0])
        temp_y = np.random.randint(0,image.shape[1])
        image[temp_x][temp_y] = 255

    image_blur = cv2.GaussianBlur(image, (5,5),0.5)
    image_blur1 = cv2.GaussianBlur(image, (5, 5), 1)
    image_lop = image_blur - image_blur1

    plt.subplot(2,2,1),plt.imshow(image,'gray')
    plt.subplot(2, 2,2), plt.imshow(image_blur,'gray')
    plt.subplot(2, 2, 3), plt.imshow(image_blur1, 'gray')
    plt.subplot(2, 2, 4), plt.imshow(image_lop, 'gray')
    plt.show()


    mAfterFFT = FFT(image)
    mBeginFFT = FFT(image,1)
    iAfter = FImage(mAfterFFT)
    iBegin = FImage(mBeginFFT)

    cv2.ShowImage('image',image)
    cv2.ShowImage('iAfter',iAfter)
    cv2.ShowImage('iBegin',iBegin)



