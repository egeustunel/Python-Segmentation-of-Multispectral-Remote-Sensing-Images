from builtins import print
from scipy import ndimage
import numpy as np
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float
from math import pi, sqrt, exp
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import cv2
import os
from PIL import  Image


def part1(blue, green, red, name):
    image = cv2.merge((blue, green, red))  # merge channels into one BGR image
    #show image on the screen and save to disk
    cv2.imshow('im', image)
    cv2.imwrite("part1/merged" + name + ".jpg", image)
    cv2.waitKey(0)
    seg = 25
    cluster = 8
    #Seperating into the superpixels
    segments = slic(img_as_float(image), n_segments=seg, compactness=28, sigma=2.0, enforce_connectivity=True)
    #creating segmented image according to the color means of the segments
    out = np.zeros_like(image)
    for segment in np.unique(segments):
        indices = np.nonzero(segments == segment)
        out[indices] = np.mean(image[indices], axis=0)

    plt.imshow(out)
    plt.savefig('part1/' + name + 'colored' + str(cluster) + '_' + str(seg) + '.png')
    plt.show()
    #k means clustering algorithm
    out = out.reshape(-1, 3)
    kmeans = KMeans(n_clusters=cluster).fit(out)
    #revisualiziton of the clustered image
    segmented_img = kmeans.cluster_centers_[kmeans.labels_]
    segmented_img = segmented_img.reshape(image.shape)
    #showing the result and saving to the disk
    plt.figure()
    plt.imshow(segmented_img.astype(np.uint8))
    plt.savefig('part1/' + name + str(cluster) + '_' + str(seg) + '.png')
    plt.show()
    #showing the superpixel representation and saving it to the disk
    fig = plt.figure("Superpixels")
    ax = fig.add_subplot(1, 1, 1)
    ax.imshow(mark_boundaries(img_as_float(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)), segments))
    plt.savefig('part1/superpixel' + name + '100.png')
    plt.axis("off")
    plt.show()

    return


def part2():


    return


def bonus():

    return


def main():
    try:
        if not os.path.exists("part1/"):
            os.makedirs("part1/")
    except OSError:
        print('Error: Creating directory. ')

    # Reading the images
    Blue = cv2.imread("data/iowa/iowa-band1.tif", 0)
    Green = cv2.imread("data/iowa/iowa-band2.tif", 0)
    Red = cv2.imread("data/iowa/iowa-band3.tif", 0)

    name = "iowa";

    part1(Blue, Green, Red, name)

    Blue = cv2.imread("data/owens_valley/owens_valley-band1.tif", 0)
    Green = cv2.imread("data/owens_valley/owens_valley-band2.tif", 0)
    Red = cv2.imread("data/owens_valley/owens_valley-band3.tif", 0)

    name = "owens_valley";

    part1(Blue, Green, Red, name)

    Blue = cv2.imread("data/salt_lake/salt_lake-band1.tif", 0)
    Green = cv2.imread("data/salt_lake/salt_lake-band2.tif", 0)
    Red = cv2.imread("data/salt_lake/salt_lake-band3.tif", 0)

    name = "salt_lake";

    part1(Blue, Green, Red, name)


    return


main()
