import pydicom as dcm
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage as ndi

try:
    from numpy.typing import NDArray
except ImportError:
    NDArray = np.ndarray

from dataclasses import dataclass, field
from pathlib import Path
import zipfile
from typing import List, Union


@dataclass
class Profile:
    direction: str
    collimator: float
    gantry: float
    data: NDArray[float]

    center_px: int = field(init=False)
    profile_max: int = field(init=False)
    fwhm_left: int = field(init=False)
    fwhm_right: int = field(init=False)
    bb_centroid: int = field(init=False)

    def __post_init__(self):
        # Small tweak
        self.collimator = int(round(self.collimator, 2))
        self.gantry = int(round(self.gantry, 2))
        if int(self.gantry) == 360:
            self.gantry = 0.

        self.data = self.data.astype(float)
        self.center_px = len(self.data) // 2

        self.locate_fwhm()
        self.locate_bb(search_radius=20)


    def locate_fwhm(self):

        # To avoid noisy pixels giving a false max
        median_filter = ndi.median_filter(self.data, size=20, mode="reflect")
        self.profile_max = np.amax(median_filter)

        # Subtract data by half the maximum. The smallest value will be that closest to FWHM
        left = np.abs(self.data[0: self.center_px] - self.profile_max // 2)
        self.fwhm_left = np.argmin(left)
        right = np.abs(self.data[self.center_px:] - self.profile_max // 2)
        self.fwhm_right = np.argmin(right) + self.center_px

    def locate_bb(self, search_radius: int = 20):
        # We assume BB is easily with +/- 25 pixels of middle
        min_vis = self.center_px - search_radius
        max_vis = self.center_px + search_radius
        self.bb_centroid = int(np.argmin(self.data[min_vis:max_vis])) + min_vis

    def plot(self):
        plt.scatter(np.arange(0, len(self.data)), self.data, s=2)
        plt.scatter(self.fwhm_left, self.data[self.fwhm_left], color="red", marker="x")
        plt.scatter(self.fwhm_right, self.data[self.fwhm_right], color="red", marker="x")
        plt.scatter(self.bb_centroid, self.data[self.bb_centroid], color="red", marker="x")

        plt.title(f"G{self.gantry}C{self.collimator} - {self.direction}")
        plt.ylabel("Intensity [A.U]")
        plt.xlabel("Pixel position [px]")

        plt.show()


class MVkV:
    def __init__(self, zip_filepath: str, unzip_path: str):
        self.zip_filepath: str = zip_filepath
        self.dcm_folder: str = unzip_path

        self.dcm_filenames: Union[List[str], None] = None
        self.dcm_files: Union[List[dcm.FileDataset], None] = None

        self.profiles: Union[List[Profile], None] = None


        self.unzip()
        self.load_dcms()

    def unzip(self):
        with zipfile.ZipFile(self.zip_filepath, 'r') as zip_ref:
            self.dcm_filenames = zip_ref.namelist()
            zip_ref.extractall(self.dcm_folder)


    def load_dcms(self):
        dcm_paths = [Path(self.dcm_folder) / f for f in self.dcm_filenames]
        self.dcm_files = [dcm.dcmread(f) for f in dcm_paths]


    def extract_profile(self, dcm_file: dcm.FileDataset):
        data = dcm_file.pixel_array

        y_profile = np.sum(data, axis=1)
        x_profile = np.sum(data, axis=0)

        profile = Profile("X", collimator=dcm_file.BeamLimitingDeviceAngle,
                          gantry=dcm_file.GantryAngle, data=x_profile)
        profile.plot()

        profile = Profile("Y", collimator=dcm_file.BeamLimitingDeviceAngle,
                          gantry=dcm_file.GantryAngle, data=y_profile)
        profile.plot()

        # print(dcm_file.filename)
        # plt.scatter(np.arange(0, len(x_profile)), x_profile, s=2)
        # plt.show()




def main():
    zip_filepath = r"C:\Users\rlefol\Documents\QAT\Ethos_MV_kV\data.zip"
    tmp_directory = r"C:\Users\rlefol\Documents\QAT\Ethos_MV_kV\unzip_dir"

    mvkv = MVkV(zip_filepath, tmp_directory)
    for f in mvkv.dcm_files:
        mvkv.extract_profile(f)


if __name__ == "__main__":
    main()