import numpy as np
from cfg import init
from scipy.interpolate import CubicSpline


def moving_average(data, weights = [0.2, 0.3, 0.5]):
    """
    Apply a moving average filter to the input data.

    Parameters:
    - data: Input data (1D array-like)
    - window_size: Size of the moving average window (integer)

    Returns:
    - smoothed_data: Data after applying the moving average filter (1D array)
    """
    # Pad the data at both ends to handle edge cases
    padded_data = np.pad(data, (len(weights) // 2, len(weights) // 2), mode='edge')

    # Apply the moving average filter
    smoothed_data = np.convolve(padded_data, weights[::-1], mode='valid')

    return smoothed_data

# Define Interpolate Function
def cubic_interp(predictions, desired_length):
    interpolated = []

    for filter in predictions:
        original_time = np.arange(0, len(filter)) * init.winDisp

        # Create a cubic spline interpolation object
        cubic_spline = CubicSpline(original_time, filter)

        new_time = np.linspace(0, original_time[-1], desired_length)

        # Interpolate the filter coefficients at the new time points
        interpolated.append(cubic_spline(new_time))

    return interpolated