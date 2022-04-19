import cv2
import numpy as np
from PIL import Image
from math import atan2,degrees
from scipy import ndimage


face_cascade = cv2.CascadeClassifier('D:/Anaconda/Library/etc/haarcascades/haarcascade_frontalface_alt.xml')
eye_cascade = cv2.CascadeClassifier('D:/Anaconda/Library/etc/haarcascades/haarcascade_eye.xml')

if face_cascade.empty():
    raise IOError('Unable to load the face cascade classifier xml file')

if eye_cascade.empty():
    raise IOError('Unable to load the eye cascade classifier xml file')

img = cv2.imread('C:/Users/Mag/Documents/Image/me.jpg')
sunglasses_img = cv2.imread('C:/Users/Mag/Documents/Image/glasses.png',cv2.IMREAD_UNCHANGED)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

centers = []
faces = face_cascade.detectMultiScale(gray, 1.3, 5)

for (x,y,w,h) in faces:
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = img[y:y+h, x:x+w]
    eyes = eye_cascade.detectMultiScale(roi_gray)
    for (x_eye,y_eye,w_eye,h_eye) in eyes:
        #cv2.rectangle(roi_color, (x_eye,y_eye), (x_eye+w_eye,y_eye+h_eye), (0,255,0), 3)
        #center of eyes in original image
        centers.append((x + int(x_eye + 0.5*w_eye), y + int(y_eye + 0.5*h_eye)))
print(centers)
# Overlay sunglasses
# دو برابر فاصله بین مرکز دو چشم
sunglasses_width = 2.12 * abs(centers[1][0] - centers[0][0])
#مقداردهی اولیه برای تصویر عینک روی چشم اندازه تصویر اصلی عکس اول
n_channels = 4
#overlay_img = np.zeros((img.shape[0],img.shape[1], n_channels), dtype=np.uint8)
overlay_img = np.ones(img.shape, np.uint8) * 255
# ارتفاع و عرض تصویر عینک
h, w = sunglasses_img.shape[:2]
# تقریبا نسبت فاصله دوبرابر دو چشم به عرض تصویر عینک
scaling_factor = sunglasses_width / w
print(scaling_factor)
# سایز تصویر عینک رو متناسب با عکس تصویر شخص کم یا زیاد میکند
width = int(sunglasses_img.shape[1] * scaling_factor)
height = int(sunglasses_img.shape[0] * scaling_factor)
dsize = (width, height)
# resize image
overlay_sunglasses = cv2.resize(sunglasses_img, dsize)
# overlay_sunglasses = cv2.resize(sunglasses_img_RGBA, None, fx=scaling_factor,
#         fy=scaling_factor, interpolation=cv2.INTER_AREA)

#   مقداردهی ایکس به مرکز چشم سمت چپ تصویر اصلی
y = centers[0][0] if centers[0][0] < centers[1][0] else centers[1][0]
yprim = centers[1][0] if centers[0][0] < centers[1][0] else centers[0][0]
if(centers[0][0] < centers[1][0]):
    x = centers[0][1]
    xprim = centers[1][1]
else:
    x = centers[1][1]
    xprim = centers[0][1]

# x = centers[0][1] if centers[0][1] < centers[1][1] else centers[1][1]
# xprim = centers[1][1] if centers[0][1] < centers[1][1] else centers[0][1]
anglebetween2point = degrees(atan2(xprim-x,yprim-y))

x -= int(0.25*overlay_sunglasses.shape[1])
# y bayad dorost meghdar dehi shavad, y feli male akharin cheshm ast
y -= int(0.46*overlay_sunglasses.shape[0] )
h, w = overlay_sunglasses.shape[:2]
#overlay_img[y:y+h, x:x+w] = overlay_sunglasses

# cv2.imshow('window_name', overlay_sunglasses)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

#rotation angle in degree
overlay_sunglasses = ndimage.rotate(overlay_sunglasses, -1*anglebetween2point)

cv2.imwrite('C:/Users/Mag/Documents/Image/NewGlasses.png', overlay_sunglasses)

# overlay_sunglasses[np.where(np.all(overlay_sunglasses[:,:,:3] == 255, -1))] = 0
# cv2.imwrite("C:/Users/Mag/Documents/Image/transparent.png", overlay_sunglasses)

# Create mask
# gray_sunglasses = cv2.cvtColor(overlay_img, cv2.COLOR_BGR2GRAY)
# ret, mask = cv2.threshold(gray_sunglasses, 110, 255, cv2.THRESH_BINARY)
# mask_inv = cv2.bitwise_not(mask)
# temp = cv2.bitwise_and(img, img, mask=mask)
# temp2 = cv2.bitwise_and(overlay_img, overlay_img, mask=mask_inv)
# final_img = cv2.add(temp, temp2)
#
# cv2.imshow('Eye Detector', img)
# cv2.imshow('Sunglasses', final_img)
# cv2.waitKey()
# cv2.destroyAllWindows()


img = Image.open("C:/Users/Mag/Documents/Image/NewGlasses.png")

background = Image.open("C:/Users/Mag/Documents/Image/me.jpg")

background.paste(img, (y, x), img)
background.save('C:/Users/Mag/Documents/Image/two_images_03.png',"PNG")