import pydicom as dcm
import numpy as np
from scipy.signal import find_peaks
import matplotlib.pyplot as plt

from numpy.typing import NDArray

from pathlib import Path
from typing import Dict, List, Union

class BeadsTray:
    def __init__(self, dicom_filepath: Union[str, Path], crop_size: int = 100,
                 expected_peaks_distance_mm: float = 10):
        self.dcm_filepath: Union[Path, str] = dicom_filepath
        self.beads_dcm: dcm.FileDataset = dcm.dcmread(dicom_filepath)
        self.beads_array: NDArray = self.beads_dcm.pixel_array
        self.pixel_spacing: List = self.beads_dcm.ImagePlanePixelSpacing
        self.crop_size: int = crop_size
        self.peaks_distance: float = expected_peaks_distance_mm

        self.cropped_beads: Union[None, NDArray] = None
        self.bands: Union[None, NDArray] = None
        self.y_band: Union[None, NDArray] = None
        self.x_peaks: Union[None, NDArray] = None
        self.y_peaks: Union[None, NDArray] = None
        self.results: Dict = {}


        print(f"shape: {self.beads_array.shape}\nmax: {np.amax(self.beads_array)}\nmin: {np.amin(self.beads_array)}")
        print(f"Spacing: {self.pixel_spacing}")


    def crop_and_normalize(self):
        # To not have to deal with the low-dose field edge
        cropped_beads = self.beads_array[self.crop_size:-self.crop_size, self.crop_size:-self.crop_size].astype(float)

        # Deal with possible dead pixels
        cropped_beads[cropped_beads < 100] = 0.

        # To try and "isolate" the beads, remove anything high-dose
        median = np.median(cropped_beads)
        threshold = 0.9
        cropped_beads[cropped_beads > threshold * median] = 0.

        # To make future math easier, normalize from 0-1
        cropped_beads[cropped_beads != 0] = cropped_beads[cropped_beads != 0] / np.amax(cropped_beads)
        cropped_beads[cropped_beads != 0.] = 1. / cropped_beads[cropped_beads != 0.]

        self.cropped_beads = cropped_beads
        return cropped_beads


    def find_midlines(self):

        # We can determine where the beads are in the array by summing up along an axis
        # The index with the max value will be the centerline containing the most BBs
        max_y = np.argmax(np.sum(self.cropped_beads, axis=0))
        max_x = np.argmax(np.sum(self.cropped_beads, axis=1))
        return max_x, max_y


    @staticmethod
    def merge_band(data, midpoint, axis, band_width=20):
        # Currently not used
        start = int(midpoint - (band_width // 2))
        stop = int(midpoint + (band_width // 2))

        # Create a slice object for our band
        sl = [slice(None)] * data.ndim
        sl[axis] = slice(start, stop)

        band = data[tuple(sl)]

        return np.nanmax(band, axis=axis)


    def get_peaks(self, beads_line: NDArray, pixel_spacing: float) -> NDArray:

        # We expect large beads (i.e big peaks) to be ~5cm apart
        pix_peak_dist_x = int(self.peaks_distance / pixel_spacing)


        peaks, _ = find_peaks(beads_line, height=1, distance=pix_peak_dist_x)
        return peaks

    def plot_peaks(self, beads_band: NDArray, beads_peaks: NDArray):
        plt.scatter(np.arange(0, len(beads_band)), beads_band, s=2)
        plt.plot(beads_peaks, beads_band[beads_peaks], "x", c="red")
        plt.show()


    def plot_2D_w_marks(self, x_coords, y_coords):

        plt.imshow(self.beads_array)
        plt.scatter(*zip(*x_coords), marker="x", color="red")
        plt.scatter(*zip(*y_coords), marker="x", color="red")
        plt.show()


    def analyze(self):
        cropped_beads = self.crop_and_normalize()

        # Get the midpoints for x & y (in array coordinates)
        mid_x, mid_y = self.find_midlines()

        # Isolate 1D array traversing the hoz and vert BB lines
        cropped_beads[cropped_beads == 0] = np.nan
        # x_beads = merge_band(cropped_beads, mid_x, axis=0, band_width=50)
        # y_beads = merge_band(cropped_beads, mid_y, axis=1, band_width=50)
        self.bands = [
            cropped_beads[mid_x, :],
            cropped_beads[:, mid_y]
            ]

        # Isolate the most prominent peaks
        self.peaks = [
            self.get_peaks(self.bands[0], self.pixel_spacing[0]),
            self.get_peaks(self.bands[1], self.pixel_spacing[1])
        ]

        self.plot_peaks(self.bands[0], self.peaks[0])
        self.plot_peaks(self.bands[1], self.peaks[1])

        # Generate coordinates for the full beads display
        x_coords = [(x + self.crop_size, mid_x + self.crop_size) for x in self.x_peaks]
        y_coords = [(mid_y + self.crop_size, y + self.crop_size) for y in self.y_peaks]

        # Reference plot with markers on the BBs
        self.plot_2D_w_marks(x_coords, y_coords)

        # Convert to mm
        x_mm_pos = (self.x_peaks + self.crop_size) * self.pixel_spacing[0]
        y_mm_pos = (self.y_peaks + self.crop_size) * self.pixel_spacing[1]

        # TODO: FIGURE THIS OUT
        # There's a weird issue with scaling which we correct with...
        x_mm_pos /= 1.5
        y_mm_pos /= 1.5
        print("YOU ARE NOT USING PROPER SCIENCE")

        # Get distance in mm between each peak (bb)
        distances_x = np.diff(x_mm_pos)
        distances_y = np.diff(y_mm_pos)

        self.results = {
            "mean_x": np.mean(distances_x),
            "std_x": np.std(distances_x),
            "mean_y": np.mean(distances_y),
            "std_y": np.std(distances_y),
        }

        print(self.results)


def main():
    dcm_filepath = Path(r"C:\Users\rlefol\Documents\QAT\MV_Beads\beads.dcm")

    # Crop the image to ensure we aren't working in the low-dose region around the edges
    crop_size = 100
    beads = BeadsTray(dcm_filepath, crop_size)
    beads.analyze()


if __name__ == "__main__":
    main()