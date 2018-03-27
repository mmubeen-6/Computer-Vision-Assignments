# Importing the necessary packages
import cv2
import numpy, scipy.sparse
from numpy.linalg import inv
import time
from matplotlib import pyplot as plt
from scipy import optimize

# Defining the Precision
numpy.set_printoptions(suppress=True)

# Creating array to store the corresponding points between the images
refPt = numpy.zeros(shape=[4, 2, 3], dtype=numpy.uint32)
counter_left = 0
counter_right = 0


# Function to find the size of the panorama after applying the homography
def calculate_size(size_image1, size_image2, homography):
    # Getting size of each input image
    (h1, w1) = size_image1[:2]
    (h2, w2) = size_image2[:2]

    # Remapping the coordinates of the projected image onto the original image space
    top_left = numpy.dot(homography, numpy.asarray([0, 0, 1]))
    top_right = numpy.dot(homography, numpy.asarray([w2, 0, 1]))
    bottom_left = numpy.dot(homography, numpy.asarray([0, h2, 1]))
    bottom_right = numpy.dot(homography, numpy.asarray([w2, h2, 1]))

    # Normalize the third value to 1
    top_left = top_left / top_left[2]
    top_right = top_right / top_right[2]
    bottom_left = bottom_left / bottom_left[2]
    bottom_right = bottom_right / bottom_right[2]

    # Finding the width
    pano_left = int(min(top_left[0], bottom_left[0], 0))
    # print(pano_left)
    pano_right = int(max(top_right[0], bottom_right[0], w1))
    # print(pano_right)
    W = pano_right - pano_left

    # Finding the Height
    pano_top = int(min(top_left[1], top_right[1], 0))
    # print(pano_top)
    pano_bottom = int(max(bottom_left[1], bottom_right[1], h1))
    # print(pano_bottom)
    H = pano_bottom - pano_top

    size = (W, H)

    # Getting the offset of projected image relative to panorama
    X = int(min(top_left[0], bottom_left[0], 0))
    Y = int(min(top_left[1], top_right[1], 0))
    offset = (-X, -Y)
    # print(offset)

    return (size, offset)


# Merging the Images Function
def merge_images(image1, image2, homography, size, offset):
    (h1, w1) = image1.shape[:2]
    (h2, w2) = image2.shape[:2]

    # Creating variable to store images
    panorama = numpy.zeros((size[1], size[0], 3), numpy.uint8)

    # Getting the offset for shifting the image
    (ox, oy) = offset

    # Converting the offset into a tranformation matrix
    translation = numpy.matrix([[1.0, 0.0, ox],
                                [0, 1.0, oy],
                                [0.0, 0.0, 1.0]
                                ])

    # Combining the homography and translation matrix
    homography = translation * homography
    # print homography

    # Applying the homography on the right image and storing it in the panorama variable
    cv2.warpPerspective(image2, homography, size, panorama)

    # Pasting the left image at the specific offset
    panorama[oy:h1 + oy, ox:ox + w1] = image1
    # panorama[0:h1, 0:w1] = image1

    # print(oy,h1 + oy, ox,ox + w1)

    return panorama


# Function to get points from the images
def click_and_getPoint(event, x, y, flags, param):
    # grab references to the global variables
    global refPt, counter_left, counter_right

    # if the left mouse button was clicked, record that point
    # It is assumed that left mouse clicked point lies in the left image
    if event == cv2.EVENT_LBUTTONDOWN or flags == cv2.EVENT_FLAG_CTRLKEY:
        # Saving the clicked point's coordinates
        refPt[counter_left % 4, 0, 0] = x
        refPt[counter_left % 4, 0, 1] = y
        refPt[counter_left % 4, 0, 2] = 1
        counter_left = counter_left + 1

    # if the right mouse button was clicked, record that point
    # It is assumed that right mouse clicked point lies in the right image
    elif event == cv2.EVENT_RBUTTONDOWN or flags == cv2.EVENT_FLAG_SHIFTKEY:
        # Saving the clicked point's coordinates
        refPt[counter_right % 4, 1, 0] = x
        refPt[counter_right % 4, 1, 1] = y
        refPt[counter_right % 4, 1, 2] = 1
        counter_right = counter_right + 1


def applyTransformation(H, points):
    output = numpy.zeros(shape=[points.shape[0], 3], dtype=numpy.int32)

    for i in range(0, points.shape[0]):
        temp = numpy.dot(H, points[i, :])
        temp = temp / temp[2]

        temp[0] = numpy.round(temp[0])
        temp[1] = numpy.round(temp[1])

        output[i, :] = temp

    return output

def costFunc(H, p1, p2):
    cost = 0.
    H = H.reshape((3, 3))
    H_inv = inv(H)

    for i in range(0, p1.shape[0]):
        # Forward Transformation
        x = p1[i, :]
        x_dash = p2[i, :]

        x = numpy.reshape(x, [1, 3])
        x_dash_estimated = applyTransformation(H, x)

        diff = numpy.sum(numpy.square(numpy.subtract(x_dash_estimated, x_dash)))
        cost = cost + diff
        # print(x_dash_estimated, x_dash, diff)

        # # Inverse Transformation
        # x = p2[i, :]
        # x_dash = p1[i, :]
        #
        # x = numpy.reshape(x, [1, 3])
        # x_dash_estimated = applyTransformation(H_inv, x)
        #
        # diff = numpy.sum(numpy.square(numpy.subtract(x_dash_estimated, x_dash)))
        # cost = cost + diff
        # # print(x_dash_estimated, x_dash, diff)

    return cost/p1.shape[0]


# Has gradient of forward transformation only
def gradient(H, p1, p2):
    H = H.reshape((3, 3))
    gradients = numpy.zeros(shape=[9,], dtype=numpy.float32)
    temp = 0.

    h = H.flatten()

    for i in range(0, p1.shape[0]):
        x_dash = p2[i, 0]
        y_dash = p2[i, 1]

        # This is repeated value in partial derivative of C wrt h1, h2, h3
        repeated_val = 2. * ((numpy.dot(h[0:3], p1[i, :].T) / numpy.dot(h[6:9], p1[i, :].T)) - x_dash) \
                                            * (1. / numpy.dot(h[6:9], p1[i, :].T))
        # Partial C by partial h1
        temp = p1[i, 0] * repeated_val
        gradients[0] = gradients[0] + temp
        # print(temp)

        # Partial C by partial h2
        temp = p1[i, 1] * repeated_val
        gradients[1] = gradients[1] + temp
        # print(temp)

        # Partial C by partial h3
        temp = p1[i, 2] * repeated_val
        gradients[2] = gradients[2] + temp
        # print(temp)

        # This is repeated value in partial derivative of C wrt h4, h5, h6
        repeated_val = 2. * ((numpy.dot(h[3:6], p1[i, :].T) / numpy.dot(h[6:9], p1[i, :].T)) - y_dash) \
                                            * (1. / numpy.dot(h[6:9], p1[i, :].T))
        # Partial C by partial h4
        temp = p1[i, 0] * repeated_val
        gradients[3] = gradients[3] + temp
        # print(temp)

        # Partial C by partial h5
        temp = p1[i, 1] * repeated_val
        gradients[4] = gradients[4] + temp
        # print(temp)

        # Partial C by partial h6
        temp = p1[i, 2] * repeated_val
        gradients[5] = gradients[5] + temp
        # print(temp)

        repeated_val = (((x_dash - numpy.dot(h[0:3], p1[i, :].T / numpy.dot(h[6:9], p1[i, :].T))) * \
                                             numpy.dot(h[0:3], p1[i, :].T)) + \
                                             ((y_dash - numpy.dot(h[3:6], p1[i, :].T / numpy.dot(h[6:9], p1[i, :].T))) * \
                                              numpy.dot(h[3:6], p1[i, :].T)))
        # Partial C by partial h7
        temp = ((2. * p1[i, 0]) / numpy.square(numpy.dot(h[6:9], p1[i, :].T))) * repeated_val
        gradients[6] = gradients[6] + temp
        # print(temp)

        # Partial C by partial h8
        temp = ((2. * p1[i, 1]) / numpy.square(numpy.dot(h[6:9], p1[i, :].T))) * repeated_val
        gradients[7] = gradients[7] + temp
        # print(temp)

        # Partial C by partial h9
        temp = ((2. * p1[i, 2]) / numpy.square(numpy.dot(h[6:9], p1[i, :].T))) * repeated_val
        gradients[8] = gradients[8] + temp
        # print(temp)

    # print(gradients)
    return  gradients/p1.shape[0]

if __name__ == "__main__":
    # load the imageS
    # using these images because the given images were not been able to be read by opencv python
    image_left = cv2.imread("img44.jpg")
    image_right = cv2.imread("img45.jpg")

    # Resizing the images
    image_left = cv2.resize(image_left, (673, 374))
    image_right = cv2.resize(image_right, (673, 374))

    # Getting a copy of each image
    clone_left = image_left.copy()
    clone_right = image_right.copy()

    # Making the Windows to show the images
    cv2.namedWindow("left-view", cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow("right-view", cv2.WINDOW_AUTOSIZE)

    # Position the windows next to eachother
    cv2.moveWindow("left-view", 5, 0)
    cv2.moveWindow("right-view", 688, 0)

    # Setting up the mouse call back functions
    cv2.setMouseCallback("left-view", click_and_getPoint)
    cv2.setMouseCallback("right-view", click_and_getPoint)

    # Showing the images in a continuously on the screen
    while False:
        # Display the image on the screen
        cv2.imshow("left-view", image_left)
        cv2.imshow("right-view", image_right)

        # Wait for a keypress
        key = cv2.waitKey(1) & 0xFF

        # if the 'r' key is pressed, reset the images
        if key == ord("r"):
            clone_left = image_left.copy()
            clone_right = image_right.copy()

        # if the 'c' key is pressed, break from the loop
        # In this case, it is assumed your would have selected the four image points
        elif key == ord("c"):
            break

    # Saved some points for usage of the two images
    # Image 44 and 45
    refPt = numpy.array([[[182, 267, 1], [119, 270, 1]],
                         [[264, 111, 1], [202, 110, 1]],
                         [[544, 92, 1], [479, 95, 1]],
                         [[329, 356, 1], [269, 357, 1]]])


    print("The input points of the two images after converting them to required format for homography:")
    print(refPt)

    p1 = refPt[:, 0, :]
    p2 = refPt[:, 1, :]

    ost = 1e10

    while cost > 5:
        # initial_H = numpy.random.randint(1, 10, (3, 3)) * 1.
        initial_H = numpy.random.rand(3, 3)
        initial_h = initial_H.flatten()
        
        result =  optimize.fmin_tnc(func=costFunc, x0=initial_h, args=(p1, p2), fprime=gradient, disp=True)
        # result = optimize.fmin_bfgs(f=costFunc, x0=initial_h, args=(p1,p2), full_output=False, disp=True)
        optimal_H = result
        optimal_H = optimal_H.reshape((3,3))
        optimal_H = optimal_H/optimal_H[-1,-1]
        cost = costFunc(optimal_H.flatten(), p1, p2)
        # break


    print(optimal_H)
    print(initial_H/initial_H[-1,-1])
    print("Cost: {}".format(costFunc(optimal_H, p1, p2)))

    H_inv = inv(optimal_H)
    # Finding the size of the panorama,
    # Along with the offset required for shifting the image
    (size, offset) = calculate_size(image_left.shape, image_right.shape, H_inv)
    # print("\nThe output size of Panorama is: %ix%i" % size)

    # Merging the images
    panorama = merge_images(image_left, image_right, H_inv, size, offset)

    # Merging the Size
    cv2.imshow("Panorama", panorama)

    cv2.waitKey()
    cv2.destroyAllWindows()
