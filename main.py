# -*- coding: utf-8 -*-
import copy
import pprint
import random
import imageio
import numpy as np
from skimage.measure import compare_ssim as ssim
from skimage.color import rgb2grey
from skimage.filters import gaussian


def getImageListFromVideo():
    filename = './example.mp4'
    video = imageio.get_reader(filename,  'ffmpeg')
    all_images = list()
    try:
        counter = 1
        while True:
            image = video.get_data(counter)
            counter += 1
            all_images.append(image)
    except Exception:
        print("Opened {} images".format(counter))
        return all_images


def mse(x, y):
    return np.linalg.norm(x - y)


def pick_random_image(all_images):
    return all_images[random.randint(0,len(all_images))]


def get_list_of_mse(main_image, all_images):
    return [mse(main_image, all_images[i]) for i in range(len(all_images))]


def get_list_of_ssim(main_image, all_images):
    return [ssim(main_image, all_images[i], multichannel=True) for i in range(len(all_images))]

def get_grey_scale_copies(all_images):
    images_to_transform = copy.deepcopy(all_images)
    return [rgb2grey(images_to_transform[i]) for i in range(len(images_to_transform))]

def get_gaussian_detection_copies(all_images):
    images_to_transform = copy.deepcopy(all_images)
    return [gaussian(images_to_transform[i], sigma=1, multichannel=True) for i in range(len(images_to_transform))]


if __name__ == '__main__' :
    all_images = getImageListFromVideo()
    main_image = pick_random_image(all_images)
    list_of_mse = get_list_of_mse(main_image, all_images)
    list_of_ssim = get_list_of_ssim(main_image, all_images)
    grey_images = get_grey_scale_copies(all_images)
    grey_list_of_ssim = get_list_of_ssim(rgb2grey(main_image), grey_images)
    grey_list_of_mse = get_list_of_mse(rgb2grey(main_image), grey_images)
    gaussian_images = get_gaussian_detection_copies(all_images)
    gaussian_list_of_ssim = get_list_of_ssim(gaussian(main_image, sigma=1, multichannel=True), gaussian_images)
    gaussian_list_of_mse = get_list_of_mse(gaussian(main_image, sigma=1, multichannel=True), gaussian_images)
    pprint.pprint(list_of_mse)
    pprint.pprint(list_of_ssim)
    pprint.pprint(grey_list_of_ssim)
    pprint.pprint(grey_list_of_mse)
    pprint.pprint(gaussian_list_of_ssim)
    pprint.pprint(gaussian_list_of_mse)
