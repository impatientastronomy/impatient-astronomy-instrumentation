"""
Tests for astrocore/camera/catalog.py.

Uses tmp_path to create temporary directories of fake filenames — no actual
image data needed.
Run with: pytest astrocore/tests/test_catalog.py -v
"""

import pytest

from astrocore.camera.catalog import scan_folder


def _touch(folder, *names):
    """Create empty .tif files in folder."""
    for name in names:
        (folder / name).touch()


# ---------------------------------------------------------------------------
# Basic scanning
# ---------------------------------------------------------------------------

class TestScanFolder:
    def test_empty_folder_returns_empty_dataframe(self, tmp_path):
        df = scan_folder(tmp_path)
        assert len(df) == 0
        assert list(df.columns) == [
            "filename", "path", "camera_id", "bin", "flip", "filter_id",
            "exposure_us", "exposure_s", "temperature_c", "frame_index",
        ]

    def test_non_matching_files_are_skipped(self, tmp_path):
        _touch(tmp_path, "random.tif", "notes.txt", "IMG_001.jpg")
        df = scan_folder(tmp_path)
        assert len(df) == 0

    def test_single_matching_file(self, tmp_path):
        _touch(tmp_path, "Cam3Bin1Flip0Filt3_1250e3us_22C_001.tif")
        df = scan_folder(tmp_path)
        assert len(df) == 1
        row = df.iloc[0]
        assert row.camera_id == 3
        assert row["bin"] == 1
        assert row["flip"] == 0
        assert row.filter_id == 3
        assert row.exposure_us == 1_250_000
        assert row.exposure_s == pytest.approx(1.25)
        assert row.temperature_c == 22.0
        assert row.frame_index == 1

    def test_multiple_files_all_parsed(self, tmp_path):
        _touch(tmp_path,
               "Cam1Bin1Flip0Filt0_1e6us_20C_001.tif",
               "Cam2Bin1Flip0Filt5_2e6us_m10C_001.tif",
               "Cam3Bin1Flip0Filt3_500e3us_unkC_001.tif")
        df = scan_folder(tmp_path)
        assert len(df) == 3

    def test_mixed_matching_and_non_matching(self, tmp_path):
        _touch(tmp_path,
               "Cam3Bin1Flip0Filt3_1250e3us_22C_001.tif",
               "dark_master.tif",
               "Cam1Bin1Flip0Filt0_2e6us_5C_002.tif")
        df = scan_folder(tmp_path)
        assert len(df) == 2


# ---------------------------------------------------------------------------
# Column dtypes
# ---------------------------------------------------------------------------

class TestDataTypes:
    def test_camera_id_is_int(self, tmp_path):
        _touch(tmp_path, "Cam3Bin1Flip0Filt3_1250e3us_22C_001.tif")
        df = scan_folder(tmp_path)
        assert df.camera_id.dtype == int

    def test_bin_is_int(self, tmp_path):
        _touch(tmp_path, "Cam3Bin2Flip0Filt3_1250e3us_22C_001.tif")
        df = scan_folder(tmp_path)
        assert df["bin"].dtype == int

    def test_flip_is_int(self, tmp_path):
        _touch(tmp_path, "Cam3Bin1Flip1Filt3_1250e3us_22C_001.tif")
        df = scan_folder(tmp_path)
        assert df["flip"].dtype == int

    def test_filter_id_is_int(self, tmp_path):
        _touch(tmp_path, "Cam3Bin1Flip0Filt3_1250e3us_22C_001.tif")
        df = scan_folder(tmp_path)
        assert df.filter_id.dtype == int

    def test_exposure_us_is_int(self, tmp_path):
        _touch(tmp_path, "Cam3Bin1Flip0Filt3_1250e3us_22C_001.tif")
        df = scan_folder(tmp_path)
        assert df.exposure_us.dtype == int

    def test_frame_index_is_int(self, tmp_path):
        _touch(tmp_path, "Cam3Bin1Flip0Filt3_1250e3us_22C_001.tif")
        df = scan_folder(tmp_path)
        assert df["frame_index"].dtype == int

    def test_unknown_temperature_is_nan(self, tmp_path):
        import math
        _touch(tmp_path, "Cam1Bin1Flip0Filt3_1e6us_unkC_001.tif")
        df = scan_folder(tmp_path)
        assert math.isnan(df.temperature_c.iloc[0])

    def test_path_is_absolute(self, tmp_path):
        _touch(tmp_path, "Cam3Bin1Flip0Filt3_1250e3us_22C_001.tif")
        df = scan_folder(tmp_path)
        assert df.path.iloc[0].is_absolute()


# ---------------------------------------------------------------------------
# Filtering and sorting (the primary use cases)
# ---------------------------------------------------------------------------

class TestFilteringAndSorting:
    def test_filter_by_camera_id(self, tmp_path):
        _touch(tmp_path,
               "Cam1Bin1Flip0Filt3_1e6us_20C_001.tif",
               "Cam3Bin1Flip0Filt3_2e6us_20C_001.tif",
               "Cam3Bin1Flip0Filt3_500e3us_20C_002.tif")
        df = scan_folder(tmp_path)
        cam3 = df[df.camera_id == 3]
        assert len(cam3) == 2

    def test_longest_exposure_for_camera(self, tmp_path):
        _touch(tmp_path,
               "Cam3Bin1Flip0Filt3_500e3us_20C_001.tif",
               "Cam3Bin1Flip0Filt3_2e6us_20C_002.tif",
               "Cam3Bin1Flip0Filt3_1e6us_20C_003.tif")
        df = scan_folder(tmp_path)
        best = df[df.camera_id == 3].sort_values("exposure_us", ascending=False).iloc[0]
        assert best.exposure_us == 2_000_000

    def test_filter_by_filter_id(self, tmp_path):
        _touch(tmp_path,
               "Cam1Bin1Flip0Filt3_1e6us_20C_001.tif",   # Ha
               "Cam1Bin1Flip0Filt5_1e6us_20C_001.tif",   # OIII
               "Cam1Bin1Flip0Filt3_2e6us_20C_002.tif")   # Ha
        df = scan_folder(tmp_path)
        ha = df[df.filter_id == 3]
        assert len(ha) == 2

    def test_sort_by_frame_index(self, tmp_path):
        _touch(tmp_path,
               "Cam1Bin1Flip0Filt3_1e6us_20C_003.tif",
               "Cam1Bin1Flip0Filt3_1e6us_20C_001.tif",
               "Cam1Bin1Flip0Filt3_1e6us_20C_002.tif")
        df = scan_folder(tmp_path)
        sorted_df = df.sort_values("frame_index")
        assert list(sorted_df["frame_index"]) == [1, 2, 3]

    def test_negative_temperature_parsed_correctly(self, tmp_path):
        _touch(tmp_path, "Cam2Bin1Flip0Filt4_3e6us_m15C_001.tif")
        df = scan_folder(tmp_path)
        assert df.temperature_c.iloc[0] == -15.0

    def test_binning_2_parsed(self, tmp_path):
        _touch(tmp_path, "Cam3Bin2Flip0Filt3_2e6us_20C_001.tif")
        df = scan_folder(tmp_path)
        assert df["bin"].iloc[0] == 2

    def test_flip_horiz_parsed(self, tmp_path):
        _touch(tmp_path, "Cam3Bin1Flip1Filt3_2e6us_20C_001.tif")
        df = scan_folder(tmp_path)
        assert df["flip"].iloc[0] == 1
