import cv2
import random
import numpy as np
from matplotlib import pyplot as plt

# 图片切割
def image_crop(img, h, w):
    img_crop = img[0:h, 0:w]
    return img_crop

# change color - random
# 随机更改BGR三个通道的数值，从而改变颜色
def random_light_color(img):
    # brightness
    B, G, R = cv2.split(img) # 通道分离

    b_rand = random.randint(-50, 50) # -50到50随机选择数字
    if b_rand == 0:
        pass
    elif b_rand > 0:
        lim = 255 - b_rand
        B[B > lim] = 255 # 防止越界， 将大于255的数值统一更新为255
        B[B <= lim] = (b_rand + B[B <= lim]).astype(img.dtype)
    elif b_rand < 0:
        lim = 0 - b_rand
        B[B < lim] = 0 # 同255， 将小于0的所有数值统一更新为0
        B[B >= lim] = (b_rand + B[B >= lim]).astype(img.dtype)

    g_rand = random.randint(-50, 50)
    if g_rand == 0:
        pass
    elif g_rand > 0:
        lim = 255 - g_rand
        G[G > lim] = 255
        G[G <= lim] = (g_rand + G[G <= lim]).astype(img.dtype)
    elif g_rand < 0:
        lim = 0 - g_rand
        G[G < lim] = 0
        G[G >= lim] = (g_rand + G[G >= lim]).astype(img.dtype)

    r_rand = random.randint(-50, 50)
    if r_rand == 0:
        pass
    elif r_rand > 0:
        lim = 255 - r_rand
        R[R > lim] = 255
        R[R <= lim] = (r_rand + R[R <= lim]).astype(img.dtype) # 将(r_rand + R[R <= lim])的格式转换成和img.dtype一样
    elif r_rand < 0:
        lim = 0 - r_rand
        R[R < lim] = 0
        R[R >= lim] = (r_rand + R[R >= lim]).astype(img.dtype)

    img_merge = cv2.merge((B, G, R)) # 将三个通道融合一下 merge
    #img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img_merge

# gamma correction 伽马映射（矫正）
# 将图变亮
def adjust_gamma(image, gamma=1.0):
    invGamma = 1.0/gamma
    table = []
    for i in range(256):
        table.append(((i / 255.0) ** invGamma) * 255) # 为了防止数值溢出
    table = np.array(table).astype("uint8") # 将table中的所有元素强制转换成uint8类型
    return cv2.LUT(img_dark, table) # look-up table

# histogram 
# 直方图, 统计每个色度出现的占比，形成一个类似与直方图的统计图
def histogram(img):
    plt.hist(img.flatten(), 256, [0, 256], color = 'r')
    plt.show()
    return 0
# img_small_brighter = cv2.resize(img_brighter, (int(img_brighter.shape[0]*0.5), int(img_brighter.shape[1]*0.5)))
# plt.hist(img_brighter.flatten(), 256, [0, 256], color = 'r') # flatten 
# img_yuv = cv2.cvtColor(img_small_brighter, cv2.COLOR_BGR2YUV)

# equalize the histogram of the Y channel
# 直方图均衡
def equalize_histogram(img):
    img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])   # only for 1 channel 直方图均衡
    # convert the YUV image back to RGB format
    img_output = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)   # y: luminance(明亮度), u&v: 色度饱和度
    # cv2.imshow('Color input image', img)
    # cv2.imshow('Histogram equalized', img_output)
    return img_output

# rotation 旋转(getRotationMatrix2D)
# scale 比例尺，如果是1就是原图比例，如果是0.5就是缩小一半
# angle 角度
def rotation(img, angle, scale):
    M = cv2.getRotationMatrix2D((img.shape[1] / 2, img.shape[0] / 2), angle, scale) # center, angle, scale 
    img_rotate = cv2.warpAffine(img, M, (img.shape[1], img.shape[0])) # 仿射变换
    return img_rotate

# Affine Transform
def rotation2(img)
    rows, cols, ch = img.shape
    pts1 = np.float32([[0, 0], [cols - 1, 0], [0, rows - 1]])
    pts2 = np.float32([[cols * 0.2, rows * 0.1], [cols * 0.9, rows * 0.2], [cols * 0.1, rows * 0.9]])
    
    M = cv2.getAffineTransform(pts1, pts2)
    return cv2.warpAffine(img, M, (cols, rows))

# perspective transform
def random_warp(img, row, col):
    height, width, channels = img.shape

    # warp:
    random_margin = 60
    x1 = random.randint(-random_margin, random_margin)
    y1 = random.randint(-random_margin, random_margin)
    x2 = random.randint(width - random_margin - 1, width - 1)
    y2 = random.randint(-random_margin, random_margin)
    x3 = random.randint(width - random_margin - 1, width - 1)
    y3 = random.randint(height - random_margin - 1, height - 1)
    x4 = random.randint(-random_margin, random_margin)
    y4 = random.randint(height - random_margin - 1, height - 1)

    dx1 = random.randint(-random_margin, random_margin)
    dy1 = random.randint(-random_margin, random_margin)
    dx2 = random.randint(width - random_margin - 1, width - 1)
    dy2 = random.randint(-random_margin, random_margin)
    dx3 = random.randint(width - random_margin - 1, width - 1)
    dy3 = random.randint(height - random_margin - 1, height - 1)
    dx4 = random.randint(-random_margin, random_margin)
    dy4 = random.randint(height - random_margin - 1, height - 1)

    pts1 = np.float32([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])
    pts2 = np.float32([[dx1, dy1], [dx2, dy2], [dx3, dy3], [dx4, dy4]])
    M_warp = cv2.getPerspectiveTransform(pts1, pts2)
    img_warp = cv2.warpPerspective(img, M_warp, (width, height))
    return M_warp, img_warp

# img_gray = cv2.imread('C:\Users\rlu058\Desktop\CV Course\lena.jpg', 0)
url = r'C:\Users\rlu058\Desktop\CV Course\lena.jpg'
img = cv2.imread(url)
img_random_color = random_light_color(img)
cv2.imshow('img_random_color', img_random_color)
key = cv2.waitKey()
if key == 27:
    cv2.destroyAllWindows()