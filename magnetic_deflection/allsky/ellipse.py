import numpy as np


def init_ellipse(x, y):
    median_x = np.median(x)
    median_y = np.median(y)

    cov_matrix = np.cov(np.c_[x, y].T)
    eigen_values, eigen_vectors = np.linalg.eig(cov_matrix)
    major_idx = np.argmax(eigen_values)
    if major_idx == 0:
        minor_idx = 1
    else:
        minor_idx = 0

    major_axis = eigen_vectors[:, major_idx]
    major_std = np.sqrt(np.abs(eigen_values[major_idx]))
    minor_std = np.sqrt(np.abs(eigen_values[minor_idx]))

    azimuth = np.arctan2(major_axis[0], major_axis[1])
    return {
        "median_cx": median_cx,
        "median_cy": median_cy,
        "azimuth": azimuth,
        "major_std": major_std,
        "minor_std": minor_std,
    }
