import os
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import tifffile
from careamics import CAREamist


def save_grayscale_stack_to_png(images: np.ndarray, output_folder: str, prefix: str = "image"):
    """
    Saves a stack of grayscale images to individual PNG files.

    Args:
        images (np.ndarray): 3D NumPy array of shape (N, W, H)
        output_folder (str): Path to the folder where PNGs will be saved
        prefix (str): Optional prefix for filenames
    """

    assert images.ndim == 3, "Input must be a 3D NumPy array (N, W, H)"
    N = images.shape[0]

    os.makedirs(output_folder, exist_ok=True)

    for i in range(N):
        img_array = images[i]

        filename = os.path.join(output_folder, f"{prefix}__{i:03d}.png")
        plt.imsave(filename, img_array, cmap="gray")

    print(f"Saved {N} images to '{output_folder}'")


def calculate_traces(data, n=7):
    """
    Generate sliding window traces of length `n` along the time axis.

    Parameters:
        data (np.ndarray): Input image stack with time as the first dimension.
        n (int): Length of each trace window.

    Returns:
        np.ndarray: Sliding window view of the data.
    """
    return np.lib.stride_tricks.sliding_window_view(data, window_shape=n, axis=0)


def calculate_tSNR_raw(data_folder="./data/noisy_data/", trace_len=7):
    """
    Calculate temporal SNR (tSNR) for raw noisy data.

    Parameters:
        data_folder (str): Path to folder containing noisy .tif image stacks.
        trace_len (int): Length of temporal trace for tSNR calculation.

    Returns:
        float: Average tSNR across all files.
    """
    noisy_folder = Path(data_folder)
    noisy_files = sorted(noisy_folder.glob("*.tif"))
    tsnr = 0
    n_files = len(noisy_files)

    for file in noisy_files:
        data = tifffile.imread(str(file))
        traces = calculate_traces(data, trace_len)
        s = 0
        n = traces.shape[0]
        for trace in traces:
            s += np.mean(trace) / np.std(trace)
        s /= n
        tsnr += s

    return tsnr / n_files


def calculate_tSNR_pred_n2v(data_folder="./data/noisy_data/", model_folder="./checkpoints/last-v4.ckpt", trace_len=7):
    """
    Calculate temporal SNR (tSNR) for denoised data using a CAREamics Noise2Void model.

    Parameters:
        data_folder (str): Path to folder containing noisy .tif image stacks.
        model_folder (str): Path to the trained CAREamics N2V model checkpoint.
        trace_len (int): Length of temporal trace for tSNR calculation.

    Returns:
        float: Average tSNR across all denoised files.
    """
    noisy_folder = Path(data_folder)
    noisy_files = sorted(noisy_folder.glob("*.tif"))
    careamist = CAREamist(model_folder)

    tsnr = 0
    n_files = len(noisy_files)

    for file in noisy_files:
        data = tifffile.imread(str(file))
        pred = careamist.predict(data)
        traces = calculate_traces(pred, trace_len)
        s = 0
        n = traces.shape[0]
        for trace in traces:
            s += np.mean(trace) / np.std(trace)
        s /= n
        tsnr += s

    return tsnr / n_files
