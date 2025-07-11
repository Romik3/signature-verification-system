import numpy as np
from skimage import io, color
from skimage.filters import threshold_otsu
from skimage.measure import regionprops
from scipy import ndimage

def rgb2gray(img):
    if len(img.shape) == 2:
        return img
    return color.rgb2gray(img)

def greybin(img):
    blur_radius = 0.8
    img = ndimage.gaussian_filter(img, blur_radius)
    thres = threshold_otsu(img)
    binimg = img > thres
    return np.logical_not(binimg)

def preproc(img):
    gray = rgb2gray(img)
    binimg = greybin(gray)
    r, c = np.where(binimg)
    return binimg[r.min():r.max(), c.min():c.max()]

def Ratio(img):
    return np.sum(img) / img.size

def Centroid(img):
    rows, cols = np.nonzero(img)
    centroid = np.mean(np.stack([rows, cols], axis=1), axis=0)
    return centroid[0] / img.shape[0], centroid[1] / img.shape[1]

def EccentricitySolidity(img):
    props = regionprops(img.astype(int))
    return props[0].eccentricity, props[0].solidity

def SkewKurtosis(img):
    h, w = img.shape
    x = np.arange(w)
    y = np.arange(h)
    xp = np.sum(img, axis=0)
    yp = np.sum(img, axis=1)
    cx = np.sum(x * xp) / np.sum(xp)
    cy = np.sum(y * yp) / np.sum(yp)
    sx = np.sqrt(np.sum(((x - cx) ** 2) * xp) / np.sum(img))
    sy = np.sqrt(np.sum(((y - cy) ** 2) * yp) / np.sum(img))
    skewx = np.sum(xp * ((x - cx) ** 3)) / (np.sum(img) * sx ** 3)
    skewy = np.sum(yp * ((y - cy) ** 3)) / (np.sum(img) * sy ** 3)
    kurtx = np.sum(xp * ((x - cx) ** 4)) / (np.sum(img) * sx ** 4) - 3
    kurty = np.sum(yp * ((y - cy) ** 4)) / (np.sum(img) * sy ** 4) - 3
    return (skewx, skewy), (kurtx, kurty)

def extract_features(image_path):
    img = io.imread(image_path)
    img = preproc(img)
    ratio = Ratio(img)
    cent_y, cent_x = Centroid(img)
    eccentricity, solidity = EccentricitySolidity(img)
    skewx, skewy = SkewKurtosis(img)[0]
    kurtx, kurty = SkewKurtosis(img)[1]
    return [ratio, cent_y, cent_x, eccentricity, solidity, skewx, skewy, kurtx, kurty]
