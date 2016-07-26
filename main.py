# -*- coding: utf-8 -*-
import pprint
import random
import imageio
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pylab
from skimage import data, img_as_float
from skimage.measure import compare_ssim as ssim


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


if __name__ == '__main__' :
    all_images = getImageListFromVideo()
    main_image = pick_random_image(all_images)
    list_of_mse = get_list_of_mse(main_image, all_images)
    list_of_ssim = get_list_of_ssim(main_image, all_images)
    pprint.pprint(list_of_mse)
    pprint.pprint(list_of_ssim)
