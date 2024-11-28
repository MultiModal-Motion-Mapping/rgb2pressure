'''
本代码用于拼接2D关节点图和图像
'''
from PIL import Image

for index in range(2, 104):
    # 假设你已经保存了多个 plt 生成的图像
    img1 = Image.open('./keypoints/{:03d}.jpg'.format(index))
    img2 = Image.open('/nasdata/jiayi/MMVP/images/images/20230422/S10/MoCap_20230422_171710/color/{:03d}.jpg'.format(index))

    # 获取图像尺寸
    width1, height1 = img1.size
    width2, height2 = img2.size

    # 创建一个新图像，宽度是两张图片宽度的总和，高度是两张图片的最大高度（水平拼接）
    new_img = Image.new('RGB', (width1 + width2, max(height1, height2)))

    # 将两张图片粘贴到新图像中
    new_img.paste(img1, (0, 0))  # 第一张图放左侧
    new_img.paste(img2, (width1, 0))  # 第二张图放右侧

    # 保存拼接后的图像
    new_img.save('./kps_images/{:03d}.jpg'.format(index))
