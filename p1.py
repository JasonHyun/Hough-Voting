import numpy as np
from PIL import Image

############### ---------- Basic Image Processing ------ ##############

# TODO 1: Read an Image and convert it into a floating point array with values between 0 and 1. You can assume a color image


def imread(filename):
    img = Image.open(filename)
    img = img.convert('RGB')
    arr = np.array(img)
    res = arr.astype(float)
    res = res / 255.0
    return res


# TODO 2: Convolve an image (m x n x 3 or m x n) with a filter(l x k). Perform "same" filtering. Apply the filter to each channel if there are more than 1 channels


def convolve(img, filt):
    filter = np.flipud(np.fliplr(filt))
    filth = filter.shape[0]
    filtw = filter.shape[1]

    imgh = img.shape[0]
    imgw = img.shape[1]

    locsum = 0.0

    res = np.zeros_like(img, dtype=np.float32)

    for n in range(imgh):
        for m in range(imgw):
            for k in range(filth):
                for l in range(filtw):
                    x = m - (filtw//2) + l
                    y = n - (filth//2) + k
                    if ((x >= 0.0 and x < imgw) and (y >= 0.0 and y < imgh)):
                        locsum = locsum + (img[y, x] * filter[k, l])

            res[n, m] = locsum
            locsum = 0.0

    return res


# TODO 3: Create a gaussian filter of size k x k and with standard deviation sigma


def gaussian_filter(k, sigma):

    res = np.zeros((k, k))
    for i in range(k):
        for j in range(k):
            i2 = (i-(k//2)) ** 2.0
            j2 = (j-(k//2)) ** 2.0
            s2 = sigma ** 2.0
            p = np.pi
            e = np.e
            res[i, j] = np.exp(-(i2 + j2)/(2.0*s2))

    sum = np.sum(res)
    res = res / sum

    return res


# TODO 4: Compute the image gradient.
# First convert the image to grayscale by using the formula:
# Intensity = Y = 0.2125 R + 0.7154 G + 0.0721 B
# Then convolve with a 5x5 Gaussian with standard deviation 1 to smooth out noise.
# Convolve with [0.5, 0, -0.5] to get the X derivative on each channel
# convolve with [[0.5],[0],[-0.5]] to get the Y derivative on each channel
# Return the gradient magnitude and the gradient orientation (use arctan2)


def gradient(img):
    gray = np.dot(img[..., :3], [0.2125, 0.7154, 0.0721])
    gauss = gaussian_filter(5, 1)
    temp = convolve(gray, gauss)
    xderiv = convolve(temp, np.reshape(np.array([0.5, 0, -0.5]), (1, 3)))
    yderiv = convolve(temp, [[0.5], [0], [-0.5]])
    mag = np.sqrt((xderiv**2)+(yderiv**2))
    ori = np.arctan2(yderiv, xderiv)

    return mag, ori


# ----------------Line detection----------------

# TODO 5: Write a function to check the distance of a set of pixels from a line parametrized by theta and c. The equation of the line is:
# x cos(theta) + y sin(theta) + c = 0
# The input x and y are arrays representing the x and y coordinates of each pixel
# Return a boolean array that indicates True for pixels whose distance is less than the threshold


def check_distance_from_line(x, y, theta, c, thresh):

    d = np.abs((x*np.cos(theta))+(y*np.sin(theta))+c)
    return d < thresh


# TODO 6: Write a function to draw a set of lines on the image. The `lines` input is a list of (theta, c) pairs. Each line must appear as red on the final image
# where every pixel which is less than thresh units away from the line should be colored red


def draw_lines(img, lines, thresh):
    pass
    # make copy of image
    # check distance from line (function call)
    # find all pixels that are within threshold distance from line
    # make pixels red
    copy = np.copy(img)
    ind = np.indices(copy.shape[:2])
    for i in lines:
        d = check_distance_from_line(ind[1], ind[0], i[0], i[1], thresh)
        copy[(ind[0][d]), (ind[1][d])] = np.array([1, 0, 0])
    return copy


# TODO 7: Do Hough voting. You get as input the gradient magnitude and the gradient orientation, as well as a set of possible theta values and a set of possible c
# values. If there are T entries in thetas and C entries in cs, the output should be a T x C array. Each pixel in the image should vote for (theta, c) if:
# (a) Its gradient magnitude is greater than thresh1
# (b) Its distance from the (theta, c) line is less than thresh2, and
# (c) The difference between theta and the pixel's gradient orientation is less than thresh3


def hough_voting(gradmag, gradori, thetas, cs, thresh1, thresh2, thresh3):
    res = np.zeros((len(thetas), len(cs)))
    x, y = np.where(gradmag > thresh1)
    count1 = 0
    count2 = 0

    for i in range(len(thetas)):
        for j in range(len(cs)):
            cond_a = check_distance_from_line(y, x, thetas[i], cs[j], thresh2)
            cond_b = np.abs(gradori[x, y] - thetas[i])
            res[i, j] += np.sum(cond_a & (cond_b < thresh3))

            # if (np.all(check_distance_from_line(y, x, theta, c, thresh2))):
            #     if (gradori[count1, count2] - theta < thresh3):
            #         res[count1, count2] += 1

    return res


# TODO 8: Find local maxima in the array of votes. A (theta, c) pair counts as a local maxima if (a) its votes are greater than thresh, and
# (b) its value is the maximum in a nbhd x nbhd beighborhood in the votes array.
# Return a list of (theta, c) pairs


def localmax(votes, thetas, cs, thresh, nbhd):
    res = []
    for n in range(len(cs)):
        for m in range(len(thetas)):
            if (votes[m, n] > thresh):
                temp = 0
                for k in range(nbhd):
                    for l in range(nbhd):
                        x = m - (nbhd//2) + l
                        y = n - (nbhd//2) + k
                        if ((x >= 0.0 and x < len(thetas)) and (y >= 0 and y < len(cs))):
                            temp = max(temp, votes[x, y])
                if (temp <= votes[m, n]):
                    res.append((thetas[m], cs[n]))

    return res

# Final product: Identify lines using the Hough transform


def do_hough_lines(filename):

    # Read image in
    img = imread(filename)

    # Compute gradient
    gradmag, gradori = gradient(img)

    # Possible theta and c values
    thetas = np.arange(-np.pi-np.pi/40, np.pi+np.pi/40, np.pi/40)
    imgdiagonal = np.sqrt(img.shape[0]**2 + img.shape[1]**2)
    cs = np.arange(-imgdiagonal, imgdiagonal, 0.5)

    # Perform Hough voting
    votes = hough_voting(gradmag, gradori, thetas, cs, 0.1, 0.5, np.pi/40)

    # Identify local maxima to get lines
    lines = localmax(votes, thetas, cs, 20, 11)

    # Visualize: draw lines on image
    result_img = draw_lines(img, lines, 0.5)

    # Return visualization and lines
    return result_img, lines
