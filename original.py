import cv2
import numpy as np
import sys
import os


class Image_Stitching:
    def __init__(self):
        self.ratio = 0.65
        self.min_match = 5
        self.sift = cv2.xfeatures2d.SIFT_create()
        self.smoothing_window_size = 50

    def registration(self, img1, img2, num):
        kp1, des1 = self.sift.detectAndCompute(img1, None)
        kp2, des2 = self.sift.detectAndCompute(img2, None)
        matcher = cv2.BFMatcher()
        raw_matches = matcher.knnMatch(des1, des2, k=2)
        good_points = []
        good_matches = []
        for m1, m2 in raw_matches:
            if m1.distance < self.ratio * m2.distance:
                good_points.append((m1.trainIdx, m1.queryIdx))
                good_matches.append([m1])
        img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good_matches, None, flags=2)
        cv2.imwrite('./output/' + str(num) + '_matching.jpg', img3)
        if len(good_points) > self.min_match:
            image1_kp = np.float32(
                [kp1[i].pt for (_, i) in good_points])
            image2_kp = np.float32(
                [kp2[i].pt for (i, _) in good_points])
            H, status = cv2.findHomography(image2_kp, image1_kp, cv2.RANSAC, 5.0)
        return H

    def create_mask(self, img1, img2, version):
        height_1 = img1.shape[0]
        width_1 = img1.shape[1]
        width_2 = img2.shape[1]
        height_blended = height_1
        width_blended = width_1 + width_2
        offset = int(self.smoothing_window_size / 2)
        barrier = img1.shape[1] - int(self.smoothing_window_size / 2)
        mask = np.zeros((height_blended, width_blended))
        if version == 'left_image':
            mask[:, barrier - offset:barrier + offset] = np.tile(np.linspace(1, 0, 2 * offset).T, (height_blended, 1))
            mask[:, :barrier - offset] = 1
        else:
            mask[:, barrier - offset:barrier + offset] = np.tile(np.linspace(0, 1, 2 * offset).T, (height_blended, 1))
            mask[:, barrier + offset:] = 1
        return cv2.merge([mask, mask, mask])

    def blending(self, img1, img2, num, order):
        if order != '1':
            img1, img2 = img2, img1
        H = self.registration(img1, img2, num)
        height_1 = img1.shape[0]
        width_1 = img1.shape[1]
        width_2 = img2.shape[1]
        height_blended = height_1
        width_blended = width_1 + width_2

        blend1 = np.zeros((height_blended, width_blended, 3))
        mask1 = self.create_mask(img1, img2, version='left_image')
        blend1[0:img1.shape[0], 0:img1.shape[1], :] = img1
        blend1 *= mask1
        mask2 = self.create_mask(img1, img2, version='right_image')
        blend2 = cv2.warpPerspective(img2, H, (width_blended, height_blended)) * mask2
        result = blend1 + blend2

        rows, cols = np.where(result[:, :, 0] != 0)
        min_row, max_row = min(rows), max(rows) + 1
        min_col, max_col = min(cols), max(cols) + 1
        result_image = result[min_row:max_row, min_col:max_col, :]
        return result_image

    def read_all_images(self, files, path):
        imgs = []
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
    imgs = stitch.read_all_images(files, argv1)
    num_of_imgs = len(imgs)
    #    for i in range(num_of_imgs):
    #        cv2.imwrite('./output/'+str(i)+'_result.jpg',imgs[i])
    #    for i in range(num_of_imgs):
    #        imgs[i] = imutils.resize(imgs[i], width=600, height=400)

    if (order != '1' and num_of_imgs > 2):
        print('Blending the images in reverse order')
        imgs = list(reversed(imgs))

    result = stitch.blending(imgs[0], imgs[1], 1, order)
    cv2.imwrite('./output/1_result.jpg', result)
    if num_of_imgs != 2:
        for i in range(num_of_imgs - 2):
            lastImg = cv2.imread('./output/' + str(i + 1) + '_result.jpg')
            print('read complete ' + str(i + 2))
            result = stitch.blending(lastImg, imgs[i + 2], i + 2, order)
            print('result complete ' + str(i + 2))
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
