# Importing the necessary packages
import cv2
import numpy
from numpy.linalg import inv
from numpy.linalg import svd
import time
from matplotlib import pyplot as plt

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

if __name__== "__main__":
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
    while True:
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
    # For 44 and 45
    refPt = numpy.array([[[182, 267, 1], [119, 270, 1]],
                         [[264, 111, 1], [202, 110, 1]],
                         [[544, 92, 1], [479, 95, 1]],
                         [[329, 356, 1], [269, 357, 1]]])
    print("The input points of the two images after converting them to required format for homography:")
    print(refPt)

    # Declaring matrix to store the A matrix
    A = numpy.zeros(shape=[8, 9], dtype=numpy.int32)

    # Getting values from point 1 for A matrix
    A_1 = numpy.zeros(shape=[2, 9], dtype=numpy.int32)
    A_1[0, 0:3] = refPt[0, 0, :]
    A_1[0, 6:9] = -1 * numpy.multiply(refPt[0, 1, 0], refPt[0, 0, :])
    A_1[1, 3:6] = refPt[0, 0, :]
    A_1[1, 6:9] = -1 * numpy.multiply(refPt[0, 1, 1], refPt[0, 0, :])

    # Getting values from point 2 for A matrix
    A_2 = numpy.zeros(shape=[2, 9], dtype=numpy.int32)
    A_2[0, 0:3] = refPt[1, 0, :]
    A_2[0, 6:9] = -1 * numpy.multiply(refPt[1, 1, 0], refPt[1, 0, :])
    A_2[1, 3:6] = refPt[1, 0, :]
    A_2[1, 6:9] = -1 * numpy.multiply(refPt[1, 1, 1], refPt[1, 0, :])

    # Getting values from point 3 for A matrix
    A_3 = numpy.zeros(shape=[2, 9], dtype=numpy.int32)
    A_3[0, 0:3] = refPt[2, 0, :]
    A_3[0, 6:9] = -1 * numpy.multiply(refPt[2, 1, 0], refPt[2, 0, :])
    A_3[1, 3:6] = refPt[2, 0, :]
    A_3[1, 6:9] = -1 * numpy.multiply(refPt[2, 1, 1], refPt[2, 0, :])

    # Getting values from point 4 for A matrix
    A_4 = numpy.zeros(shape=[2, 9], dtype=numpy.int32)
    A_4[0, 0:3] = refPt[3, 0, :]
    A_4[0, 6:9] = -1 * numpy.multiply(refPt[3, 1, 0], refPt[3, 0, :])
    A_4[1, 3:6] = refPt[3, 0, :]
    A_4[1, 6:9] = -1 * numpy.multiply(refPt[3, 1, 1], refPt[3, 0, :])

    # Storing values in the A matrix
    A[0:2, :] = A_1
    A[2:4, :] = A_2
    A[4:6, :] = A_3
    A[6:8, :] = A_4

    print("\nPrinting the A matrix")
    print(A)

    # Getting the SVD of A matrix
    U, s, Vh = svd(A, full_matrices=True)

    # Extracting the last column of the V matrix
    # Normalizing L matrix wrt to last element i.e. H33
    L = Vh[-1, :] / Vh[-1, -1]

    # Reshaping the L matrix to 3x3 to make it H matrix
    H = L.reshape(3, 3)

    print("\nThe H matrix is: ")
    print(H)

    # Getting the inverse homography matrix
    H_inv = inv(H)

    # Finding the size of the panorama,
    # Along with the offset required for shifting the image
    (size, offset) = calculate_size(image_left.shape, image_right.shape, H_inv)
    print("\nThe output size of Panorama is: %ix%i" % size)

    # Merging the images
    panorama = merge_images(image_left, image_right, H_inv, size, offset)

    # Merging the Size
    cv2.imshow("Panorama", panorama)
    # plt.imshow(cv2.cvtColor(panorama, cv2.COLOR_BGR2RGB), interpolation = 'bilinear')
    # plt.show()

    cv2.waitKey()
    cv2.destroyAllWindows()
