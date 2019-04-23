import cv2
import numpy as np
# #image = cv2.imread("/var/www/html/"+sys.argv[1])
# image = cv2.imread("C:/python3/Indoor-segmentation-master/Indoor-segmentation-master/output/11.png")
# img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# img_edges = cv2.Canny(img_gray, 100, 100, apertureSize=3)
# cv2.imshow('img_edges', img_edges)
# cv2.waitKey(0)
# lower = np.array([0,65,70])
# high = np.array([8,255,255])
# blurred = cv2.pyrMeanShiftFiltering(image, 11,21)
# hsv = cv2.cvtColor(blurred,cv2.COLOR_BGR2HSV)
# mask = cv2.inRange(hsv,lower,high)
# res = cv2.bitwise_and(blurred,blurred,mask=mask)
#
#
#
# import cv2
#
# img1 = cv2.imread('C:/python3/Indoor-segmentation-master/Indoor-segmentation-master/output/11.png')
# img2 = cv2.imread('C:/python3/Indoor-segmentation-master/Indoor-segmentation-master/11.png')
#
# # img1[img1[:, :, 1:].all(axis=-1)] = 0
# # img2[img2[:, :, 1:].all(axis=-1)] = 0
# #
# # dst = cv2.addWeighted(img1, 1, img2, 1, 0)
# dst = cv2.addWeighted(img2,0.5,img1,0.5,1)
# cv2.imshow('dst',dst)
# cv2.imwrite('main.png',dst)
# img_gray = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)
# cv2.imshow('img_gray',img_gray)
# lower = np.array([0,65,70])
# high = np.array([80,255,255])

import numpy as np
import cv2
image = cv2.imread('image.png')
image1 = cv2.imread('main.jpg')
img_gray = cv2.cvtColor(image,cv2.COLOR_BGR2RGBA)
print("img_gray",img_gray.shape)
#cv2.imwrite('image.png',image)
# data = np.array(image)
# print (data)
lower = np.array([48,72,78])
high = np.array([70,255,255])
#blurred = cv2.pyrMeanShiftFiltering(img_gray, 9,19)
hsv = cv2.cvtColor(img_gray,cv2.COLOR_BGR2HSV)
mask = cv2.inRange(hsv,lower,high)
res = cv2.bitwise_and(image,image,mask=mask)
a= res.shape
print(a)
img_gray = cv2.cvtColor(res,cv2.COLOR_BGR2RGBA)
_,alpha = cv2.threshold(img_gray,0,255,cv2.THRESH_BINARY)
b, g, r,alpha = cv2.split(img_gray)
rgba = [b,g,r, alpha]

dst = cv2.merge(rgba,4)
cv2.imwrite("test.png", dst)
image2=cv2.imread("test.png")
# /image[image[:, :, 1:].all(axis=-1)] = 0
# dst[dst[:, :, 1:].all(axis=-1)] = 0
resultant = cv2.addWeighted(image1, 0.8, image2, 0.2, 1)
cv2.imshow("resultant",resultant)
cv2.waitKey(0)
cv2.destroyAllWindows()














# _,alpha = cv2.threshold(img_res,0,255,cv2.THRESH_BINARY)
#cv2.imshow('img_gray', img_gray)
# b, g, r = cv2.split(img_gray)
# rgba = [b,g,r, alpha]

#
# dst = cv2.merge(rgba,4)
# cv2.imshow('dst1', dst)
# cv2.imshow('dst1',res)
cv2.waitKey(0)
cv2.destroyAllWindows()
# # converted = np.where(data ==4, 200, 3)
# img = Image.fromarray(converted.astype(np.uint8))
# img.save('new_pic.jpg')
# import cv2
# img1 = cv2.imread("new_pic.jpg")
#
# img2 = cv2.imread('C:/python3/Indoor-segmentation-master/Indoor-segmentation-master/11.png')
# dst = cv2.addWeighted(img2,0.7,img1,0.3,1)
# cv2.imshow('dst',dst)