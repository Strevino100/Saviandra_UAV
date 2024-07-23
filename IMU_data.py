import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation as rot
import matplotlib.pyplot as plt

accel_data = pd.read_csv("/home/trevis/Nonchalant_Noise_Smoothing/Accelerometer.csv")
accel_times = accel_data["seconds_elapsed"].tolist()
orient_data = pd.read_csv("/home/trevis/Nonchalant_Noise_Smoothing/Orientation.csv")
orient_times = orient_data["seconds_elapsed"].tolist() # Because the linear and rotational acceleration data have different sample rates

# Uses a Savitzky-Golay filter to denoise data based on https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.savgol_filter.html
def smooth(data):
    window_size = 200
    polynomial_order = 2
    return savgol_filter(data, window_size, polynomial_order)

# Trapezoidal Reimann Sum
def integrate(data, time):
    result = np.zeros(len(data))
    result[0] = 0
    for i in range(1, len(data)):
        average = 0.5 * (data[i-1] + data[i])
        result[i] = average * (time[i] - time[i-1]) + result[i-1]
    return result

# Used to estimate rotational acceleration at timestamp of linear acceleration, since they have different sample rates
def estimate_value(axis, time):
    data = orient_data[axis].tolist()
    for i in range(len(data)):
        if time < orient_times[0]:
            return data[0]
        if time == orient_times[i]:
            return data[i]
        if time < orient_times[i]:
            return ((time - orient_times[i-1])/(orient_times[i] - orient_times[i-1]))*(data[i] - data[i-1]) + data[i-1] # Weighted average of the two rotational acceleration data points the timestamp falls between

# Denoise all data before doing anything with it
smooth_x_phone = smooth(accel_data["x"].tolist())
smooth_y_phone = smooth(accel_data["y"].tolist())
smooth_z_phone = smooth(accel_data["z"].tolist())
smooth_roll = smooth(orient_data["roll"].tolist())
smooth_pitch = smooth(orient_data["pitch"].tolist())
smooth_yaw = smooth(orient_data["yaw"].tolist())

# To find position as a function of time via integrating acceleration, the phone's recorded accelerations need to be tranformed into the ground's reference frame so the axes don't change direction midway
# This can be done using rotational matrices

# Construct a list of state vectors for translational and rotational acceleration at each timestamp in accel_data to feed into rotational matrix transformation algorithm
accel_matrices, attitude_matrices = [], []
for i in range(len(smooth_x_phone)):
    accel_matrices.append([smooth_z_phone[i], smooth_x_phone[i], smooth_y_phone[i]])
    attitude_matrices.append([estimate_value("roll", accel_times[i]), estimate_value("pitch", accel_times[i]), estimate_value("yaw", accel_times[i])])

accel_matrices, attitude_matrices = np.array(accel_matrices), np.array(attitude_matrices)

# Rotational matrix transformation turns the phone-axis accelerations to ground-axis accelerations for each state vector
transformation_matrices = rot.from_euler("yzx", attitude_matrices, degrees = True)
earth_accel_matrices = transformation_matrices.apply(accel_matrices)

# Extract the ground-axis accelerations from the list of transformed state vectors
a_x, a_y, a_z = [], [], []
for i in range(len(earth_accel_matrices)):
    a_x.append(earth_accel_matrices[i][0])
    a_y.append(earth_accel_matrices[i][1])
    a_z.append(earth_accel_matrices[i][2])

# Integrate acceleration relative to ground twice to get position relative to ground
v_x = integrate(a_x, accel_times)
v_y = integrate(a_y, accel_times)
v_z = integrate(a_z, accel_times)
x = integrate(v_x, accel_times)
y = integrate(v_y, accel_times)
z = integrate(v_z, accel_times)

# Plot using matplotlib 3D scatterplot
fig = plt.figure()
plt.plot(x, y)
##ax = fig.add_subplot(projection='3d')
##ax.scatter(x, y, z)
plt.show()