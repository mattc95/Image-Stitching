import cv2
import numpy as np
import sys
import os


class Image_Stitching:
    def __init__(self):
        # 初始化参数
        self.ratio = 0.65
        self.min_match = 5
        self.sift = cv2.xfeatures2d.SIFT_create()
        self.smoothing_window_size = 100

    def registration(self, img1, img2, num):
        # 使用SIFT获得特征点
        key_point1, des1 = self.sift.detectAndCompute(img1, None)
        key_point2, des2 = self.sift.detectAndCompute(img2, None)
        raw_matches = cv2.BFMatcher().knnMatch(des1, des2, k=2)
        best_features = []
        best_matches = []
        # 通过KNN（K=2）获得最佳特征点
        for m1, m2 in raw_matches:
            if m1.distance < self.ratio * m2.distance:
                best_features.append((m1.trainIdx, m1.queryIdx))
                best_matches.append([m1])
        img3 = cv2.drawMatchesKnn(img1, key_point1, img2, key_point2, best_matches, None, flags=2)
        # 输出特征点匹配图
        cv2.imwrite('./output/' + str(num) + '_matching.jpg', img3)
        if len(best_features) > self.min_match:
            image1_kp = np.float32(
                [key_point1[i].pt for (_, i) in best_features])
            image2_kp = np.float32(
                [key_point2[i].pt for (i, _) in best_features])
            # 计算单应性矩阵，加入了Ransac
            H, status = cv2.findHomography(image2_kp, image1_kp, cv2.RANSAC, 5.0)
        return H

    def create_mask(self, img1, img2, side):
        height_blended = img1.shape[0]
        width_blended = img1.shape[1] + img2.shape[1]
        offset = int(self.smoothing_window_size / 2)
        barrier = img1.shape[1] - int(self.smoothing_window_size / 2)
        mask = np.zeros((height_blended, width_blended))

        # 左右图使用不同的MASK
        if side == 'left':
            mask[:, barrier - offset:barrier + offset] = np.tile(np.linspace(1, 0, 2 * offset).T, (height_blended, 1))
            mask[:, :barrier - offset] = 1
        else:
            mask[:, barrier - offset:barrier + offset] = np.tile(np.linspace(0, 1, 2 * offset).T, (height_blended, 1))
            mask[:, barrier + offset:] = 1
        return cv2.merge([mask, mask, mask])

    def blending(self, img1, img2, num, order):
        # 反序时，将交换图片位置
        if order != '1':
            img1, img2 = img2, img1
        # 获得单应性矩阵
        H = self.registration(img1, img2, num)
        height_blended = img1.shape[0]
        width_blended = img1.shape[1] + img2.shape[1]

        # 进行图片仿射变换与融合
        blend1 = np.zeros((height_blended, width_blended, 3))
        blend1[0:img1.shape[0], 0:img1.shape[1], :] = img1
        blend1 *= self.create_mask(img1, img2, 'left')
        blend2 = cv2.warpPerspective(img2, H, (width_blended, height_blended)) * self.create_mask(img1, img2, 'right')
        result = blend1 + blend2

        rows, cols = np.where(result[:, :, 0] != 0)
        result_image = result[min(rows):max(rows) + 1, min(cols):max(cols) + 1, :]
        return result_image

    def read_all_images(self, files, path):
        imgs = []
        # 遍历文件夹，获得所有图片
        for i in range(len(files)):
            imgs.append(cv2.imread(path + '/' + files[i]))
        return imgs


def main(argv1, argv2='1'):
    folder_name = argv1
    order = argv2
    print('Get all images in folder: ', folder_name)
    files = os.listdir(folder_name)
    print('All images in folder: ', files)
    print('Number of images is: ', len(files))
    stitch = Image_Stitching()
    # 读取folder内的所有图片
    imgs = stitch.read_all_images(files, argv1)
    num_of_imgs = len(imgs)

    # 将图片列表反转
    if order != '1' and num_of_imgs > 2:
        print('Blending the images in reverse order')
        imgs = list(reversed(imgs))

    # 拼接前两张图片
    result = stitch.blending(imgs[0], imgs[1], 1, order)
    cv2.imwrite('./output/1_result.jpg', result)
    if num_of_imgs != 2:
        # 图片数量大于2时，将一张张拼接
        for i in range(num_of_imgs - 2):
            lastImg = cv2.imread('./output/' + str(i + 1) + '_result.jpg')
            print('read complete ' + str(i + 2))
            result = stitch.blending(lastImg, imgs[i + 2], i + 2, order)
            print('result complete ' + str(i + 2))
            # 输出结果
            cv2.imwrite('./output/' + str(i + 2) + '_result.jpg', result)
            print('blend complete ' + str(i + 2))
    print('All complete')


if __name__ == '__main__':
    try:
        if len(sys.argv) == 3:
            main(sys.argv[1], sys.argv[2])
        else:
            main(sys.argv[1])
    except IndexError:
        print("Please input: python ./")
