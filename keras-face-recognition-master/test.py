import utils.utils as utils
import cv2
if __name__=='__main__':
    img=cv2.imread('0.jpg')
    print(img.shape)
    img=utils.reshape_face(img)
    print(img.shape)
    cv2.imshow('result',img)
    cv2.waitKey(0)
