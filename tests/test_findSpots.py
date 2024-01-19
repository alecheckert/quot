import numpy as np
import tifffile
from pathlib import Path
from unittest import TestCase
from quot.findSpots import METHODS as DETECT_METHODS, detect, llr


TEST_DIR = Path(__file__).absolute().parent


class TestFindSpots(TestCase):
    """Unit tests for the quot.findSpots interface."""

    @classmethod
    def setUpClass(cls):
        movie_path = TEST_DIR / "fixtures" / "sample_movie.tif"
        assert movie_path.is_file(), movie_path
        cls.im = tifffile.TiffFile(movie_path).pages[0].asarray().astype(np.float64)

    def test_detect(self):
        """Test consistency of the quot.findSpots.detect interface
        with respect to all current detection methods.

        We test that:
            - All detection methods return a 2D array of YX coordinates
            for each detected spot.
            - This array is integer-valued.

        Some detection methods - such as hess_det_var - will return
        zero spots on this test image. Such methods should return an
        ndarray of shape (0, 2)."""
        for method in DETECT_METHODS:
            detections = detect(self.im.copy(), method=method)
            assert isinstance(detections, np.ndarray), type(detections)
            assert len(detections.shape) == 2, detections.shape
            assert detections.shape[1] == 2, detections.shape
            assert np.issubdtype(detections.dtype, np.integer), detections.dtype


class TestLLRRegression(TestCase):
    """Test that LLR continues to return numerically identical
    results at the detection map level, regardless of changes
    in the LLR function."""

    def test_llr_regression(self):
        image_path = TEST_DIR / "fixtures" / "sample_movie.tif"
        ref_map_path = TEST_DIR / "fixtures" / "ref_llr_map.tif"
        im = tifffile.TiffFile(image_path).pages[0].asarray().astype(np.float64)
        llr_map, _, detections = llr(im, w=11, k=1.5, t=14.0, return_filt=True)
        ref_llr_map = tifffile.imread(ref_map_path)
        np.testing.assert_allclose(llr_map, ref_llr_map, atol=1e-6, rtol=1e-6)
        assert detections.shape == (6, 2)
        detections = {(y, x) for (y, x) in detections}
        expected = {(5, 122), (22, 10), (42, 104), (81, 89), (97, 79), (36, 95)}
        assert expected == detections, f"expected {expected}, got {detections}"


class TestLLRHoles(TestCase):
    """Convex and concave spots look identical to the LLR algorithm
    without selecting for the curvature sign. This test requires that
    we only return convex spots by synthesizing a small test with a
    concave and convex spot."""

    def test_llr_holes(self):
        np.random.seed(666)
        image_size = (128, 128)
        spot_center = (64, 64)
        camera_offset = 100.0
        k = 1.5
        w = 11
        kwargs = dict(method="llr", k=k, w=w, t=14.0, return_filt=False)

        # Prerequisite for this test: there are no "background" detections
        im = np.random.normal(loc=camera_offset, size=image_size)
        out = detect(im, **kwargs)
        assert out.shape == (0, 2), out.shape

        # Place a convex Gaussian with intensity 100.0 at (32, 32)
        # and a concave Gaussian with the same intensity at (70, 70)
        gauss = np.exp(-((np.indices((w, w)) - w // 2) ** 2).sum(axis=0) / (2 * k**2))
        gauss /= gauss.sum()
        convex = (32, 32)
        concave = (70, 70)
        im[
            convex[0] - w // 2 : convex[0] + w // 2 + 1,
            convex[1] - w // 2 : convex[1] + w // 2 + 1,
        ] += (
            100.0 * gauss
        )
        im[
            concave[0] - w // 2 : concave[0] + w // 2 + 1,
            concave[1] - w // 2 : concave[1] + w // 2 + 1,
        ] -= (
            100.0 * gauss
        )

        # Destructure output
        out = [(y, x) for (y, x) in detect(im, **kwargs)]
        # Convex spot is present
        assert convex in out, out
        # Concave spot is not present
        assert concave not in out, out
        # No other spots are present
        assert len(out) == 1, out
