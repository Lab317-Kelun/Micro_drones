# coding=utf-8
# 导入python包
import numpy as np
import argparse
import cv2

# 构建并解析参数
#ap = argparse.ArgumentParser()
#ap.add_argument("-i", "--image", required = True, help = "/home/nvidia/robot_ws/src/robot_cv/script/333.png")
#args = vars(ap.parse_args())

img='/home/nvidia/robot_ws/src/robot_cv/script/333.png'
# 读取彩色图片
image = cv2.imread(img,1)
output = image.copy()
# 将其转换为灰度图片
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 应用hough变换进行圆检测
circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1.2, 100)

# 确保至少发现一个圆
if circles is not None:
	# 进行取整操作
	circles = np.round(circles[0, :]).astype("int")

	# 循环遍历所有的坐标和半径
	for (x, y, r) in circles:
		# 绘制结果
		cv2.circle(output, (x, y), r, (0, 255, 0), 4)
		cv2.rectangle(output, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)

	# 显示结果
	cv2.imshow("output", np.hstack([image, output]))
	cv2.waitKey(0)


