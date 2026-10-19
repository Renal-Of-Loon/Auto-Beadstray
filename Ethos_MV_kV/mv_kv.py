import pydicom as dcm
import numpy as np
from numpy.polynomial import Polynomial
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.patches import Circle, Rectangle
from scipy import ndimage as ndi
from scipy import optimize as opt
from scipy.special import erf

try:
    from numpy.typing import NDArray
except ImportError:
    NDArray = np.ndarray

from dataclasses import dataclass, field
from pathlib import Path
import zipfile
from typing import Dict, List, Union, Tuple
import warnings

def gaussian_2d(coords, amplitude, x0, y0, sigma_x, sigma_y, offset):
    x, y = coords

    g = offset - amplitude * np.exp(
        -(((x - x0)**2) / (2 * sigma_x**2)
        + ((y - y0)**2) / (2 * sigma_y**2))
    )

    return g.ravel()

@dataclass
class FieldData:
    x1: int
    x2: int
    y1: int
    y2: int
    bb_x: float
    bb_y: float
    bb_fwhm: float
    px2mm: float

    @property
    def bb_x_int(self):
        return int(round(self.bb_x, 0))

    @property
    def bb_y_int(self):
        return int(round(self.bb_y, 0))

    @property
    def x_field_center(self):
        return np.average([self.x1, self.x2])

    @property
    def y_field_center(self):
        return np.average([self.y1, self.y2])


@dataclass
class ProfileData:
    projection: str
    projection_data: NDArray
    profile_max: int
    profile_bgd: int
    fwhm_left: int
    fwhm_right: int
    edge_left: int
    edge_right: int

@dataclass
class BBCenter:
    ANT: float
    LEFT: float
    INF: float


class FieldAnalysis:
    def __init__(self, dicom: dcm.FileDataset):
        self.data = dicom.pixel_array
        self.gantry = dicom.GantryAngle
        self.collimator = dicom.BeamLimitingDeviceAngle
        self.px_to_mm: float = float(dicom.ImagePlanePixelSpacing[0])
        self.magnification: float = float(dicom.RadiationMachineSAD) / float(dicom.RTImageSID)

        self.profile_x: Union[ProfileData, None] = None
        self.profile_y: Union[ProfileData, None] = None
        self.cropped_data: Union[NDArray, None] = None
        self.gauss_opt: Dict = {}
        self.gauss_offsets: Tuple = ()

        # Minor data cleanup
        self.collimator = int(round(self.collimator, 2))
        self.gantry = int(round(self.gantry, 2))
        if int(self.gantry) == 360:
            self.gantry = 0.

        self.name = f"G{self.gantry}C{self.collimator}"


    def analyze_bb(self, max_trials: int=6, max_bb_size: int = 100):

        # We crop the field in a square shape as we use ravel() in the fit which creates odd position
        #   with non-square shape arrays
        min_x, max_x = self.profile_x.fwhm_left, self.profile_x.fwhm_right

        px_pad = 5
        for trial in range(0, max_trials):
            try:
                pad = px_pad * trial
                self.cropped_data = self.data[min_x + pad:max_x - pad, min_x + pad:max_x - pad]
                self.fit_gauss_2d()
                self.gauss_offsets = (min_x + pad, min_x + pad)

                # The fit sometimes converts to an absurd size, a BB should never be more than 100px
                if self.gauss_opt["bb_fwhm"] > max_bb_size:
                    self.gauss_opt = {}
                    continue

                break
            except RuntimeError:
                pass

        if len(self.gauss_opt) == 0:
            raise RuntimeError(f"Could not perform gaussian fit on the BB for {self.name}")


    def analyze_field_edges(self):
        self.profile_x = self.find_field_edge("X")
        self.profile_y = self.find_field_edge("Y")


    def find_field_edge(self, projection: str):
        if projection == "X":
            profile_data = np.sum(self.data, axis=0)
        elif projection == "Y":
            profile_data = np.sum(self.data, axis=1)
        else:
            raise ValueError(f"Projection value {projection} is not X or Y")

        center_px = len(profile_data) // 2

        # To avoid noisy pixels giving a false max
        median_filter = ndi.median_filter(profile_data, size=20, mode="reflect")
        profile_max = np.percentile(median_filter, 99)
        profile_bgd = np.percentile(median_filter, 5)

        # Subtract data by half the maximum. The smallest value will be that closest to FWHM
        left = np.abs(profile_data[0: center_px] - profile_max // 2)
        fwhm_left = np.argmin(left)
        right = np.abs(profile_data[center_px:] - profile_max // 2)
        fwhm_right = np.argmin(right) + center_px

        # To fit the gaussian on the BB we want virtually only the field
        left = np.abs(profile_data[0: center_px] - profile_max // 1.15)
        edge_left = np.argmin(left)
        right = np.abs(profile_data[center_px:] - profile_max // 1.15)
        edge_right = np.argmin(right) + center_px

        return ProfileData(
            projection=projection,
            projection_data=profile_data,
            profile_max=profile_max,
            profile_bgd=profile_bgd,
            fwhm_right=fwhm_right,
            fwhm_left=fwhm_left,
            edge_left=edge_left,
            edge_right=edge_right
        )

    def plot_1D_field_edges(self, projection: str):
        if projection == "X":
            profile_data = self.profile_x
        elif projection == "Y":
            profile_data = self.profile_y
        else:
            raise ValueError(f"Projection value {projection} is not X or Y")

        data = profile_data.projection_data
        x = np.arange(0, len(data))
        plt.scatter(x, data, s=2)
        plt.scatter(profile_data.fwhm_left, data[profile_data.fwhm_left], color="red", marker="x")
        plt.scatter(profile_data.fwhm_right, data[profile_data.fwhm_right], color="red", marker="x")

        plt.title(f"G{self.gantry}C{self.collimator} - {profile_data.projection}")
        plt.ylabel("Intensity [A.U]")
        plt.xlabel("Pixel position [px]")

        plt.show()

    def fit_gauss_2d(self):
        # Assume darkest pixel is BB center
        y0, x0 = np.unravel_index(np.argmin(self.cropped_data), self.cropped_data.shape)
        amplitude0 = self.cropped_data.max() - self.cropped_data.min()
        sigma0 = 3
        offset0 = self.cropped_data.max()

        y, x = np.indices(self.cropped_data.shape)
        p0 = (amplitude0, x0, y0, sigma0, sigma0, offset0)

        # Use scipy curve_fit to fit our 2D gaussian to the BB "dip" in dose
        popt, pcov = opt.curve_fit(gaussian_2d, (x, y),
            self.cropped_data.ravel(), p0=p0)
        amplitude, x0, y0, sigma_x, sigma_y, offset = popt

        # Get fwhm from sigma to draw circle
        sigma = (sigma_x + sigma_y) / 2
        fwhm = 2.3548 * sigma

        self.gauss_opt = {
            "bb_x": x0,
            "bb_y": y0,
            "bb_sigma_x": sigma_x,
            "bb_sigma_y": sigma_y,
            "bb_fwhm": fwhm,
            "bb_params": popt,
            "bb_cov": pcov,
        }


    def plot_gauss(self, ax: Union[Axes, None] = None, display: bool = True) -> Union[Axes, None]:

        if "bb_sigma_x" not in self.gauss_opt:
            warnings.warn("Cannot plot without first performing a fit")
            return None

        if ax is None:
            fig, ax = plt.subplots()

        # Cropped dose image
        ax.imshow(self.cropped_data)
        # X marks BB center
        ax.plot(self.gauss_opt["bb_x"], self.gauss_opt["bb_y"], marker="X", color="red")

        # Draw a circle at the FWHM (we kind of force the circle shape assumption above)
        circ = Circle(
            (self.gauss_opt["bb_x"], self.gauss_opt["bb_y"]),
            radius=self.gauss_opt["bb_fwhm"],
            fill=False,
            color="red",
            linewidth=2
        )
        ax.add_patch(circ)

        if display:
            plt.show()

        return ax


    def plot_combined(self, px_padding: int = 20, in_mm: bool = False):
        if "bb_fwhm" not in self.gauss_opt:
            warnings.warn("Cannot plot without first performing a fit")
            return None

        scale = self.px_to_mm if in_mm is True else 1
        padding = px_padding # px

        fig, ax = plt.subplots()
        x_min = (self.profile_x.fwhm_left - padding) * scale
        x_max = (self.profile_x.fwhm_right + padding) * scale
        y_min = (self.profile_y.fwhm_left - padding) * scale
        y_max = (self.profile_y.fwhm_right + padding) * scale

        # Our BB fit was done with a semi-arbitrary crop size, account for that here
        # bb_x = (self.gauss_opt["bb_x"] + self.gauss_offsets[0] - x_min) * scale
        # bb_y = (self.gauss_opt["bb_y"] + self.gauss_offsets[1] - y_min) * scale
        # bb_fwhm = self.gauss_opt["bb_fwhm"] * scale
        bb_x = (self.gauss_opt["bb_x"] + self.gauss_offsets[0]) * scale
        bb_y = (self.gauss_opt["bb_y"] + self.gauss_offsets[1]) * scale
        bb_fwhm = self.gauss_opt["bb_fwhm"] * scale

        # Display the "background" field
        #display_data = self.data[x_min:x_max, y_min:y_max]
        #ax.imshow(display_data, extent=[y_min-0.5, y_max-0.5, x_min-0.5, x_max-0.5], origin="upper")
        ax.imshow(self.data)

        # Display our BB center & FWHM radius
        # X marks BB center
        ax.plot(bb_x, bb_y, marker="X", color="red")
        # Draw a circle at the FWHM (we kind of force the circle shape assumption above)
        circ = Circle(
            (bb_x, bb_y),
            radius=bb_fwhm,
            fill=False,
            color="red",
            linewidth=2
        )
        ax.add_patch(circ)

        # Display FWHM as a rectangle
        # x_line_coords = (self.profile_x.fwhm_left - x_min) * scale, (self.profile_x.fwhm_right - x_min) * scale
        # y_line_coords = (self.profile_y.fwhm_left - y_min) * scale, (self.profile_y.fwhm_right - y_min) * scale
        x_line_coords = self.profile_x.fwhm_left * scale, self.profile_x.fwhm_right * scale
        y_line_coords = self.profile_y.fwhm_left * scale, self.profile_y.fwhm_right * scale

        width = x_line_coords[1] - x_line_coords[0]
        height = y_line_coords[1] - y_line_coords[0]
        rect = Rectangle(
            (x_line_coords[0], y_line_coords[0]), width, height,
            linewidth=2, edgecolor='r', facecolor='none'
        )
        ax.add_patch(rect)

        ax.set_xlim(y_min, y_max)
        ax.set_ylim(x_min, x_max)
        plt.title(self.name)
        plt.show()


    @property
    def metrics(self):
        return FieldData(
            x1 = int(self.profile_x.fwhm_left),
            x2 = int(self.profile_x.fwhm_right),
            y1 = int(self.profile_y.fwhm_left),
            y2 = int(self.profile_y.fwhm_right),
            bb_x = float(self.gauss_opt["bb_x"] + self.gauss_offsets[0]),
            bb_y = float(self.gauss_opt["bb_y"] + self.gauss_offsets[1]),
            bb_fwhm = float(self.gauss_opt["bb_fwhm"]),
            px2mm = float(self.px_to_mm * self.magnification)
        )


class MVkV:
    def __init__(self, zip_filepath: str, unzip_path: str):
        self.zip_filepath: str = zip_filepath
        self.dcm_folder: str = unzip_path

        self.dcm_filenames: Union[List[str], None] = None
        self.dcm_files: Union[List[dcm.FileDataset], None] = None
        self.field: Union[FieldAnalysis, None] = None
        self.fields: Dict[str, FieldAnalysis] = {}
        self.px2mm: Union[float, None] = None

        self.crosshairs: Tuple = ()
        self.kv_bb_shifts: BBCenter
        self.mv_bb_shifts: BBCenter
        self.mv_qa_values: Dict = {}
        self.qa_values: Dict = {}


        self.unzip()
        self.load_dcms()

    def unzip(self):
        with zipfile.ZipFile(self.zip_filepath, 'r') as zip_ref:
            self.dcm_filenames = zip_ref.namelist()
            zip_ref.extractall(self.dcm_folder)


    def load_dcms(self):
        dcm_paths = [Path(self.dcm_folder) / f for f in self.dcm_filenames]
        self.dcm_files = [dcm.dcmread(f) for f in dcm_paths]


    def field_analysis(self, dcm_file: dcm.FileDataset, plot: bool = False):
        f = FieldAnalysis(dcm_file)
        f.analyze_field_edges()
        f.analyze_bb(max_trials=10)

        self.fields[f.name] = f

        if plot:
            f.plot_combined()

    def set_mv_qa_values(self):

        px2mm = self.fields["G0C90"].metrics.px2mm
        self.px2mm = px2mm

        L_R_field_center_90 = self.fields["G0C90"].metrics.x_field_center
        L_R_field_center_270 = self.fields["G0C270"].metrics.x_field_center
        L_R_field_center = float(np.average([L_R_field_center_90, L_R_field_center_270]))

        A_P_field_center_90 = self.fields["G90C90"].metrics.x_field_center
        A_P_field_center_270 = self.fields["G90C270"].metrics.x_field_center
        A_P_field_center = float(np.average([A_P_field_center_90, A_P_field_center_270]))

        S_I_field_center_90 = self.fields["G90C90"].metrics.y_field_center
        S_I_field_center_270 = self.fields["G90C270"].metrics.y_field_center
        S_I_field_center = float(np.average([S_I_field_center_90, S_I_field_center_270]))

        L_R_bb = self.fields["G0C90"].metrics.bb_x_int
        A_P_bb = self.fields["G90C90"].metrics.bb_x_int
        S_I_bb = float(np.average([self.fields["G0C90"].metrics.bb_y_int, self.fields["G90C90"].metrics.bb_y_int]))

        bb_ANT_shift = (A_P_field_center - A_P_bb) * px2mm
        bb_LEFT_shift = -1 * (L_R_field_center - L_R_bb) * px2mm
        bb_INF_shift = -1 * (S_I_field_center - S_I_bb) * px2mm

        field_asym_leaf_edges = max(L_R_field_center_90, A_P_field_center_90) * px2mm
        field_asym_leaf_ends = S_I_field_center_90 * px2mm
        mv_panel_shift_y = (self.fields["G0C90"].metrics.bb_y_int - self.fields["G90C90"].metrics.bb_y_int) * px2mm
        mv_panel_shift_x = abs(
            (self.fields["G0C90"].metrics.bb_x_int - L_R_field_center) - (A_P_field_center - self.fields["G90C90"].metrics.bb_x_int)
        ) * px2mm

        self.mv_bb_shifts = BBCenter(
            ANT=bb_ANT_shift,
            LEFT=bb_LEFT_shift,
            INF=bb_INF_shift
        )

        self.mv_qa_values = {
            "L/R_field_center": L_R_field_center,
            "A/P_field_center": A_P_field_center,
            "S/I_field_center": S_I_field_center,
            "BB_ANT_shift": bb_ANT_shift,
            "BB_LEFT_shift": bb_LEFT_shift,
            "BB_INF_shift": bb_INF_shift,
            "field_asym_leaf_edges": field_asym_leaf_edges,
            "field_asym_leaf_ends": field_asym_leaf_ends,
            "mv_panel_shift_y": mv_panel_shift_y,
            "mv_panel_shift_x": mv_panel_shift_x
        }

    def set_kv_mv_qa_values(self):

        if len(self.crosshairs) != 2:
            raise ValueError("Cannot perform final QA compute without crosshair positions")

        max_laser_mviso_delta = float(np.max(np.abs(
            [self.mv_qa_values["BB_ANT_shift"], self.mv_qa_values["BB_LEFT_shift"], self.mv_qa_values["BB_INF_shift"]]
        )))

        kvmv_AP_delta = abs(self.mv_bb_shifts.ANT - self.kv_bb_shifts.ANT)
        kvmv_LR_delta = abs(self.mv_bb_shifts.LEFT - self.kv_bb_shifts.LEFT)
        kvmv_SI_delta = abs(self.mv_bb_shifts.INF - self.kv_bb_shifts.INF)

        mv_crosshairs_AP_delta = abs(self.crosshairs[0] - self.mv_qa_values["A/P_field_center"]) * self.px2mm
        mv_crosshairs_LR_delta = abs(self.crosshairs[0] - self.mv_qa_values["L/R_field_center"]) * self.px2mm
        mv_crosshairs_SI_delta = abs(self.crosshairs[1] - self.mv_qa_values["S/I_field_center"]) * self.px2mm

        self.mvkv_qa_values = {
            "max_laser_mviso_delta": max_laser_mviso_delta,
            "kvmv_AP_delta": kvmv_AP_delta,
            "kvmv_LR_delta": kvmv_LR_delta,
            "kvmv_SI_delta": kvmv_SI_delta,
            "mv_crosshairs_AP_delta": mv_crosshairs_AP_delta,
            "mv_crosshairs_LR_delta": mv_crosshairs_LR_delta,
            "mv_crosshairs_SI_delta": mv_crosshairs_SI_delta
        }

        self.qa_values = {**self.mv_qa_values, **self.mvkv_qa_values}



def main():
    zip_filepath = r"C:\Users\rlefol\Documents\QAT\Ethos_MV_kV\data.zip"
    tmp_directory = r"C:\Users\rlefol\Documents\QAT\Ethos_MV_kV\unzip_dir"

    mvkv = MVkV(zip_filepath, tmp_directory)
    for f in mvkv.dcm_files:
        mvkv.field_analysis(f, plot=True)

    mvkv.crosshairs = (641, 640)
    # Expects: ANT, LEFT, INF
    mvkv.kv_bb_shifts = BBCenter(ANT=-0.5, LEFT=-0.4, INF=1)
    mvkv.set_mv_qa_values()



    print()


if __name__ == "__main__":
    main()