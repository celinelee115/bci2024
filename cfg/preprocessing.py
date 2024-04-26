import numpy as np
from scipy import signal as sig

# filter raw signal
def filter_data(data, pass_band=(0.15, 200), fs=1000, order=4):
    """
    Input:
      data (samples x channels): the raw (noisy) signal
      fs: the sampling rate (1000 for this dataset)
    Output:
      clean_data (samples x channels): the filtered signal
    """
    sos = sig.butter(order, Wn=pass_band, btype="bandpass", fs=fs, output="sos")
    if len(data.shape) > 1:
        clean_data = np.array([sig.sosfiltfilt(sos, eeg) for eeg in data.T]).T
    else:
        clean_data = sig.sosfiltfilt(sos, data)

    # assert clean_data.shape == data.shape
    return clean_data


# Calculate features for a given filtered window from the ECoG signal 
# Define zero crossing function used in get_features function
def fn_zero_crossings(x):
        x_cent = x - np.mean(x, axis=0)
        x_shift = np.hstack([x_cent[:, 1:], x_cent[:, -1].reshape((-1, 1))])
        x_cross = (x_cent * x_shift) < 0
        return np.sum(x_cross, axis=0)

def get_features(filtered_window, fs=1000):
    """
    Input:
        filtered_window (window_samples x channels): the window of the filtered ecog signal
        fs: sampling rate
    Output:
        features (channels x num_features): the features calculated on each channel for the window
    """
    features = np.zeros((len(filtered_window[0]), 9))

    # Feature 1: Average time-domain voltage
    features[:,0] = np.mean(filtered_window, axis=0)

    # Feature 2-4: Hjorth parameters
    # Hjorth Activity
    features[:,1] = np.var(filtered_window, axis=0)
    # Hjorth Mobility
    first_derivative  = np.diff(filtered_window, axis=0)
    features[:,2] = np.sqrt(np.var(first_derivative, axis=0) / np.var(filtered_window, axis=0))
    # Hjorth Complexity
    features[:,3] = np.sqrt(np.var(np.diff(first_derivative, axis=0)) / np.var(first_derivative, axis=0))

    # Feature 5-7: Power spectrum with different bandwidth range
    freqs, psd = sig.welch(filtered_window, fs=fs, axis=0, nperseg = 100)
    # Alpha (12-30 Hz)
    features[:,4] = np.mean(psd[(freqs >= 12) & (freqs <= 30)], axis=0)
    # Beta (30-70 Hz)
    features[:,5] = np.mean(psd[(freqs >= 30) & (freqs <= 70)], axis=0)
    # Gamma (70-150 Hz)
    features[:,6] = np.mean(psd[(freqs >= 70) & (freqs <= 150)], axis=0)

    # Feature 8: Zero Crossings
    features[:,7] = fn_zero_crossings(filtered_window)

    # Feature 9: Absolute Sum of Differences
    features[:,8] = np.sum(np.absolute(np.diff(filtered_window, axis=0)), axis=0)

    return features


# Define the function for calculating number of window
def NumWins(x, length, displacement, fs=1000):
  return (len(x) - length)/displacement  + 1


def get_windowed_feats(raw_ecog, fs, window_length, window_overlap):
    """
    Inputs:
        raw_eeg (samples x channels): the raw signal
        fs: the sampling rate (1000 for this dataset)
        window_length: the window's length
        window_overlap: the window's overlap
    Output:
        all_feats (num_windows x (channels x features)): the features for each channel for each time window
        note that this is a 2D array.
    """
    # Filter the raw ecog signal
    filtered_ecog = filter_data(raw_ecog, fs=fs)

    # Calculate number of windows
    M = NumWins(filtered_ecog, window_length, window_overlap, fs)

    # Initialize output array
    num_channels = len(filtered_ecog[0])
    num_feature = 9
    all_feats = np.zeros((int(M), num_channels * num_feature))

    for n in range(int(M)):
        start = int(n * window_overlap)
        end = start + int(window_length)
        filtered_window = filtered_ecog[start:end]

        # Calculate and store features of each window
        all_feats[n, :] = get_features(filtered_window).flatten()

    return all_feats

# Calculate the R matrix
def create_R_matrix(features, N_wind):
  """
  Input:
    features (samples (number of windows in the signal) x channels x features):
      the features you calculated using get_windowed_feats
    N_wind: number of windows to use in the R matrix

  Output:
    R (samples x (N_wind*channels*features))
  """
  # Append first N-1 to the beginning of feature array
  features_adjusted = np.concatenate([features[:2,:], features], axis=0)

  # Initialization of response matrix
  num_rows = len(features)
  num_cols = len(features[0]) * N_wind + 1
  num_features = 9
  num_channels = int(len(features[0])/num_features)
  R = np.zeros((num_rows, num_cols))

  for i in range(num_rows):
    # Append one to the beginning of each row
    R[i,0] = 1

    for j in range(num_channels):
      start_col = j * num_features * N_wind + 1
      end_col = start_col + num_features * N_wind

      start_index = j * num_features
      end_index = start_index + num_features

      R[i, start_col:end_col] = features_adjusted[i:i+N_wind, start_index:end_index].flatten()

  return R



# from scipy.io import loadmat 
# fs = 1000 # (Hz)
# winLen = 100e-3 # (s)
# winDisp = 50e-3 # (s)
# length = winLen * fs
# displacement = winDisp * fs

# truetest_data = loadmat('leaderboard_data.mat') # truetest_data
# test_ecog = truetest_data['leaderboard_ecog'] # truetest_data
# test1_features = get_windowed_feats(test_ecog[0,0], fs, length, displacement)
# print(test1_features.shape)