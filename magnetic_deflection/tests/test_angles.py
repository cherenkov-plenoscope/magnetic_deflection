import magnetic_deflection as mdfl
import numpy as np


def test_azimuth_range():
    assert 0 == mdfl.discovery._azimuth_range(0)
    assert 90 == mdfl.discovery._azimuth_range(90)
    assert 180 == mdfl.discovery._azimuth_range(180)
    assert -179 == mdfl.discovery._azimuth_range(181)
    assert -90 == mdfl.discovery._azimuth_range(-90)
    assert 180 == mdfl.discovery._azimuth_range(180)


def test_azimuth_range_array():
    arr = mdfl.discovery._azimuth_range(np.array([0., 90., 180., 181.]))
    assert arr[0] == 0.
    assert arr[1] == 90.
    assert arr[2] == 180.
    assert arr[3] == -179.


def test_angles_scalars():
    for az_deg in np.linspace(-380, 380, 25):
        for zd_deg in np.linspace(0, 89, 25):

            cx, cy = mdfl.discovery._az_zd_to_cx_cy(
                azimuth_deg=az_deg,
                zenith_deg=zd_deg)

            az_deg_back, zd_deg_back = mdfl.discovery._cx_cy_to_az_zd_deg(
                cx=cx,
                cy=cy)

            dist_deg = mdfl.discovery._angle_between_az_zd_deg(
                az1_deg=az_deg,
                zd1_deg=zd_deg,
                az2_deg=az_deg_back,
                zd2_deg=zd_deg_back)

            print(az_deg, zd_deg, az_deg_back, zd_deg_back, cx, cy)
            assert dist_deg < 1


def test_angles_arrays():
    num = 25
    az_deg = np.linspace(-380, 380, num)
    zd_deg = np.linspace(0, 89, num)

    cx, cy = mdfl.discovery._az_zd_to_cx_cy(
        azimuth_deg=az_deg,
        zenith_deg=zd_deg)

    az_deg_back, zd_deg_back = mdfl.discovery._cx_cy_to_az_zd_deg(
        cx=cx,
        cy=cy)

    dist_deg = mdfl.discovery._angle_between_az_zd_deg(
        az1_deg=az_deg,
        zd1_deg=zd_deg,
        az2_deg=az_deg_back,
        zd2_deg=zd_deg_back)

    print(az_deg, zd_deg, az_deg_back, zd_deg_back, cx, cy)
    assert np.sum(dist_deg < 1) == num
