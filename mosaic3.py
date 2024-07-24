import cv2
import numpy as np
import glob

# Load Calibration Parameters:
#     Function load_calibration(calibration_file):
def load_calibration(calibration_file):
    #    Load calibration data from the file
    data = np.load(calibration_file)

    #    Extract camera matrix and distortion coefficients
    mtx = data["camera_matrix"]
    dist = data["dist_coeffs"]

    #    Return camera matrix and distortion coefficients
    return mtx, dist

# Undistort Image:
#     Function undistort_image(image, camera_matrix, dist_coeffs):
def undistort_image(image, camera_matrix, dist_coeffs):
    # NOTE: Resizing images to 25% to improve processing speed.
    image = cv2.resize(image, (image.shape[1] // 4, image.shape[0] // 4))

    # NOTE: Input images are likely already undistorted (confirmed with Immanuel)
    # Undistorting them will just distort them more.
    #    Get image dimensions (height, width)
    # height, width = image.shape[:2]

    # #    Compute new camera matrix for undistortion
    # new_mtx, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (width, height), 1, (width, height))

    # #    Undistort the image (use cv2 undistort)
    # dst = cv2.undistort(image, camera_matrix, dist_coeffs, None, new_mtx)

    #    Crop the undistorted image using ROI
    # x, y, w, h = roi
    # dst = dst[y:y + h, x:x + w]

    #    Return undistorted image
    return image

# Harris Corner Detection:
#     Function harris_corner_detection(image):
def harris_corner_detection(image):
    #    Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)

    #    Apply Harris corner detection
    dst = cv2.cornerHarris(gray, 2, 3, 0.04)

    #    Dilate corners
    dst = cv2.dilate(dst, None)

    #    Mark corners on the image
    gray[dst > 0.01 * dst.max()] = 255

    #    Return image with marked corners and detected corners
    return np.uint8(gray)

# Match Features Between Images:
#     Function match_features(image1, image2):
def match_features(image1, image2):
    sift = cv2.SIFT_create()

    #    Detect keypoints and descriptors in image1 using SIFT
    kp1, des1 = sift.detectAndCompute(image1, None)

    #    Detect keypoints and descriptors in image2 using SIFT
    kp2, des2 = sift.detectAndCompute(image2, None)

    #    Match descriptors using brute-force matcher
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    #    Extract matched points from both images
    # Apply ratio test to select good candidates
    good = [m for m, n in matches if m.distance < 0.7 * n.distance]

    # Build src and dst arrays
    src = []
    dst = []
    distances = []
    for match in good:
        srcpt = kp1[match.queryIdx].pt
        dstpt = kp2[match.trainIdx].pt

        src.append(srcpt)
        dst.append(dstpt)

        # Get vertical distance between matched features
        distances.append(abs(dstpt[1] - srcpt[1]))

    # Filter points based on vertical distance
    # Because the images were taken at about the same height, the best matched features will be
    # closest to matching perfectly horizontally (y difference == 0).
    # If a feature in the source image is matched to a feature in the destination image with a
    # completely different height, it is likely noise or an outlier and should be ignored.
    least = np.array(distances) < 30

    #    Return matched points from image1 and image2
    least = np.tile(least, (2, 1)).T
    format_pts = lambda x: np.float32(x)[least].reshape(-1, 1, 2)

    return format_pts(src), format_pts(dst)

# Create Mosaic:
#     Function create_mosaic(images, camera_matrix, dist_coeffs):
def create_mosaic(images, camera_matrix, dist_coeffs):
    #    Undistort all images using undistort_image function
    undistorted = [undistort_image(cv2.imread(i), camera_matrix, dist_coeffs) for i in images]

    #    Initialize mosaic with the first undistorted image
    mosaic = undistorted[0]

    #    For each subsequent undistorted image:
    for img in undistorted[1:]:
        #    Detect Harris corners in both mosaic and current image using harris_corner_detection
        corners_img = harris_corner_detection(img)
        corners_mosaic = harris_corner_detection(mosaic)

        #    Match features between mosaic and current image using match_features
        pts1, pts2 = match_features(corners_img, corners_mosaic)

        #    Estimate homography using matched points
        h, _ = cv2.findHomography(pts1, pts2, cv2.RANSAC, 5.0)

        #    Warp mosaic image using the estimated homography
        # NOTE: Very inefficient in terms of memory since the image width is expanded very quickly.
        dst = cv2.warpPerspective(img, h, (img.shape[1] + mosaic.shape[1], img.shape[0]))

        #    Blend current image into mosaic
        # NOTE: Mosaic is padded right with zeros so dimensions match the warped image
        z = np.zeros((dst.shape[0], dst.shape[1] - mosaic.shape[1], 3))
        mosaic2 = np.concatenate((mosaic, z), axis=1)

        # Add non-black pixels to final image
        tmp = np.where(mosaic2 > 0, mosaic2, dst)
        mosaic = np.uint8(tmp)

    # Remove empty pixels caused by overallocation of image width
    gray = cv2.cvtColor(mosaic, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    x, y, w, h = cv2.boundingRect(contours[0])
    mosaic = mosaic[y:y + h, x:x + w]

    # Return final mosaic image
    return mosaic

# Main:
#     Load camera matrix and distortion coefficients from calibration file
mtx, dist = load_calibration("/home/trevis/bwsi-uav/laboratory_20242.0/week_1_Hw/camera_calibration.npz")

#     Load images from specified directory
# NOTE: Change the directory of the image dataset if desired
images = sorted(glob.glob("/home/trevis/bwsi-uav/laboratory_2024/week_1_Hw/camera_calibration_photo_mosaic/International Village - 50 Percent Overlap/*.jpg"))

#     Create mosaic using create_mosaic function
mosaic_image = create_mosaic(images, mtx, dist)

#     Save the mosaic image to a file
cv2.imwrite("mosaic3.png", mosaic_image)

# Display the mosaic image
cv2.imshow("Mosaic", mosaic_image)
cv2.waitKey(0)
cv2.destroyAllWindows()