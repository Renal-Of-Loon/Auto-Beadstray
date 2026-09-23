import pydicom as dcm
import numpy as np
from scipy.signal import find_peaks
import matplotlib.pyplot as plt

from numpy.typing import NDArray

from pathlib import Path
from typing import List

def crop_and_normalize(beads_array: NDArray, crop_size: int):
    # To not have to deal with the low-dose field edge
    cropped_beads = beads_array[crop_size:-crop_size, crop_size:-crop_size].astype(float)

    # Lets also deal with possible dead pixels
    cropped_beads[cropped_beads < 100] = 0.

    # To try and "isolate" the beads, remove anything high-dose
    median = np.median(cropped_beads)
    threshold = 0.9
    cropped_beads[cropped_beads > threshold * median] = 0.

    # To make future math easier, normalize from 0-1
    cropped_beads[cropped_beads != 0] = cropped_beads[cropped_beads != 0] / np.amax(cropped_beads)
    cropped_beads[cropped_beads != 0.] = 1. / cropped_beads[cropped_beads != 0.]

    return cropped_beads


def find_midlines(beads_only: NDArray):

    # We can determine where the beads are in the array by summing up along an axis
    # The index with the max value will be the centerline containing the most BBs
    max_y = np.argmax(np.sum(beads_only, axis=0))
    max_x = np.argmax(np.sum(beads_only, axis=1))
    return max_x, max_y


def merge_band(data, midpoint, axis, band_width=20):
    start = int(midpoint - (band_width // 2))
    stop = int(midpoint + (band_width // 2))

    # Create a slice object for our band
    sl = [slice(None)] * data.ndim
    sl[axis] = slice(start, stop)

    band = data[tuple(sl)]

    return np.nanmax(band, axis=axis)


def get_peaks(beads_line: NDArray, pixel_spacing: float) -> NDArray:

    # We expect large beads (i.e big peaks) to be ~5cm apart
    pix_peak_dist_x = int(10 / pixel_spacing)


    peaks, _ = find_peaks(beads_line, height=1, distance=pix_peak_dist_x)
    return peaks


def plot_2D_w_marks(beads_array, x_coords, y_coords):

    # Since we cropped the initial image, need to correct the index

    plt.imshow(beads_array)
    plt.scatter(*zip(*x_coords), marker="x", color="red")
    plt.scatter(*zip(*y_coords), marker="x", color="red")
    plt.show()


    print()


def main():
    dcm_filepath = Path(r"C:\Users\rlefol\Documents\QAT\MV_Beads\beads.dcm")
    beads_dcm = dcm.dcmread(dcm_filepath)
    beads_array = beads_dcm.pixel_array
    pixel_spacing = beads_dcm.ImagePlanePixelSpacing
    print(f"shape: {beads_array.shape}\nmax: {np.amax(beads_array)}\nmin: {np.amin(beads_array)}")
    print(f"Spacing: {pixel_spacing}")

    # Crop the image to ensure we aren't working in the low-dose region around the edges
    crop_size = 100
    cropped_beads = crop_and_normalize(beads_array, crop_size)

    # Get the midpoints for x & y (in array coordinates)
    mid_x, mid_y = find_midlines(cropped_beads)

    # Isolate 1D array traversing the hoz and vert BB lines
    cropped_beads[cropped_beads == 0] = np.nan
    #x_beads = merge_band(cropped_beads, mid_x, axis=0, band_width=50)
    #y_beads = merge_band(cropped_beads, mid_y, axis=1, band_width=50)
    x_beads = cropped_beads[mid_x, :]
    y_beads = cropped_beads[:, mid_y]

    # Isolate the most prominent peaks
    x_peaks = get_peaks(x_beads, pixel_spacing[0])
    y_peaks = get_peaks(y_beads, pixel_spacing[1])

    # Generate coordinates for the full beads display
    x_coords = [(x+crop_size, mid_x+crop_size) for x in x_peaks]
    y_coords = [(mid_y+crop_size, y + crop_size) for y in y_peaks]

    # Reference plot with markers on the BBs
    plot_2D_w_marks(beads_array, x_coords, y_coords)

    # Convert to mm
    x_mm_pos = (x_peaks + crop_size) * pixel_spacing[0]
    y_mm_pos = (y_peaks + crop_size) * pixel_spacing[1]

    # TODO: FIGURE THIS OUT
    # There's a weird issue with scaling which we correct with...
    x_mm_pos /= 1.5
    y_mm_pos /= 1.5
    print("YOU ARE NOT USING PROPER SCIENCE")

    # Get distance in mm between each peak (bb)
    distances_x = np.diff(x_mm_pos)
    distances_y = np.diff(y_mm_pos)

    results = {
        "mean_x": np.mean(distances_x),
        "std_x": np.std(distances_x),
        "mean_y": np.mean(distances_y),
        "std_y": np.std(distances_y),
    }

    print(results)

    plt.scatter(np.arange(0, len(x_beads)), x_beads, s=2)
    plt.plot(x_peaks, x_beads[x_peaks], "x", c="red")
    plt.show()
    print()


if __name__ == "__main__":
    main()