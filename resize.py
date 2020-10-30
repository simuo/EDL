import os
from cfg import *
from PIL import Image
import cv2
import numpy

# imgPath=r'D:\VIDI\XWVision-VP9.5 -20200228-Point\ImageFile\12.bmp'
# img=cv2.imread(imgPath,1)
# cv2.namedWindow("BMP", cv2.WINDOW_GUI_NORMAL)
# cv2.imshow("BMP", img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

path = r'D:\VIDI\ExploreDLInIndustry\Betel_nut'
imgs = [os.path.join(path, name) for name in os.listdir(path)]
for i, imgstr in enumerate(imgs):
    print(imgstr)
    img = Image.open(imgstr)
    img_resize = img.resize((IMG_WIDTH, IMG_HEIGHT),Image.BILINEAR)
    img_resize.save('D:\VIDI\ExploreDLInIndustry\Betel_nut\imgresize\{}.jpg'.format(imgstr.split('\\')[-1].split('.')[0]))
