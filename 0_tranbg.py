import cv2
import numpy as np
import os

# 输入和输出文件夹路径
input_folder = '/Users/alandeng/Documents/VScode/root quantify'
output_folder = '/Users/alandeng/Documents/VScode/root quantify/tranbg'

# 确保输出文件夹存在
os.makedirs(output_folder, exist_ok=True)

# 遍历文件夹中的所有jpg图像
for filename in os.listdir(input_folder):
    if filename.lower().endswith('.jpg'):
        img_path = os.path.join(input_folder, filename)
        img = cv2.imread(img_path)

        if img is None:
            print(f"无法读取图像: {img_path}")
            continue

        # 转换为灰度图
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 阈值分割
        _, mask = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)

        # 去除小黑点（噪声）
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)

        # 创建白色背景图
        white_bg = np.ones_like(img, dtype=np.uint8) * 255

        # 使用掩膜复制图像内容到白色背景
        result = cv2.bitwise_and(img, img, mask=mask)
        inv_mask = cv2.bitwise_not(mask)
        white_part = cv2.bitwise_and(white_bg, white_bg, mask=inv_mask)
        final = cv2.add(result, white_part)

        # 保存处理后的图像
        output_path = os.path.join(output_folder, filename)
        cv2.imwrite(output_path, final)
        print(f"已保存: {output_path}")