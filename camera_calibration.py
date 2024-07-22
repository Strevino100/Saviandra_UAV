import numpy as np
import cv2
import glob

# Define checkerboard_dims as (9, 6)
checkerboard_dims = (8,6)

# Create objp as a zero array of shape (number of corners, 3), float32
# Set the first two columns of objp to the coordinate grid of corners
objp = np.zeros((checkerboard_dims[0]*checkerboard_dims[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:checkerboard_dims[0], 0:checkerboard_dims[1]].T.reshape(-1, 2)
print("Object points (objp):")
print(objp)

# Initialize objpoints as an empty list
objpoints = []

# Initialize imgpoints as an empty list
imgpoints = []

# Load all checkerboard images using glob
images = glob.glob('/home/trevis/bwsi-uav/laboratory_2024/week_1_Hw/camera_calibration_photo_mosaic/calibration photos/calib*.jpg')
print(f"Found {len(images)} images")

# For each image in images:
for idx, image in enumerate(images):
    # Read the image
    img = cv2.imread(image)
    if img is None:
        print(f"Image at {image} could not be loaded.")
        continue
    print(f"Processing image {idx+1}/{len(images)}: {image}")
    
    # Convert the image to grayscale
    gs_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Find the chessboard corners in the grayscale image
    ret, corners = cv2.findChessboardCorners(gs_img, checkerboard_dims, None)
    
    # If corners are found:
    if ret:
        # Append objp to objpoints
        objpoints.append(objp)
        
        # Refine corner positions using cornerSubPix
        corners2 = cv2.cornerSubPix(gs_img, corners, (11,11), (-1,-1), criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
        
        # Append refined corners to imgpoints
        imgpoints.append(corners2)
        
        # Optionally, draw chessboard corners on the image
        cv2.drawChessboardCorners(img, checkerboard_dims, corners2, ret)
        
        # Optionally, display the image with drawn corners
        cv2.imshow('img', img)
        cv2.waitKey(500)

# Destroy all OpenCV windows
cv2.destroyAllWindows()

if len(objpoints) == 0 or len(imgpoints) == 0:
    print("No object points or image points found, cannot calibrate.")
else:
    # Calibrate the camera using calibrateCamera with objpoints, imgpoints, and image size
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gs_img.shape[::-1], None, None)
    #print("Camera calibration results:")
    #print(f"ret: {ret}")
    #print(f"Camera matrix: {mtx}")
    #print(f"Distortion coefficients: {dist}")

    # Save the calibration results (camera matrix, distortion coefficients) to a file.


    # Verify the calibration:
    # Initialize mean_error to 0
mean_error = 0

    # For each pair of object points and image points:
for i in range(len(objpoints)):
    # Project the object points to image points using projectPoints
    imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        
# Compute the error between the projected and actual image points
    error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        
        # Accumulate the error
    mean_error += error

    # Compute the average error
total_error = mean_error / len(objpoints)

    # Print the total average error
print(f"Total error: {total_error}")


np.savez('/home/trevis/camera_calibration.npz', camera_matrix=mtx, dist_coeffs=dist)