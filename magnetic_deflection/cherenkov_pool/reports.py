import numpy as np
import tarfile
import os
import gzip
import io

import spherical_coordinates
import dynamicsizerecarray

from . import production


def append(in_paths, out_path, remove_in_paths=True):
    opj = os.path.join

    in_paths = sorted(in_paths)
    out_path_tmp = out_path + ".part"

    with tarfile.open(out_path_tmp, "w|") as otf:
        if os.path.exists(out_path):
            with tarfile.open(out_path, "r|") as itf:
                for tarinfo in itf:
                    payload = itf.extractfile(tarinfo).read()
                    _append_tar(otf, tarinfo.name, payload)

        for report_path in in_paths:
            with open(report_path, "rb") as f:
                payload = f.read()
            report_basename = os.path.basename(report_path)
            _append_tar(otf, report_basename + ".gz", gzip.compress(payload))

    os.rename(out_path_tmp, out_path)

    if remove_in_paths:
        for report_path in in_paths:
            os.remove(report_path)


def _append_tar(tarfout, name, payload_bytes):
    tarinfo = tarfile.TarInfo()
    tarinfo.name = name
    tarinfo.size = len(payload_bytes)
    with io.BytesIO() as fileobj:
        fileobj.write(payload_bytes)
        fileobj.seek(0)
        tarfout.addfile(tarinfo=tarinfo, fileobj=fileobj)


def read(path, dtype=None, mask_function=None):
    if dtype is None:
        dtype = production.histogram_cherenkov_pool_report_dtype()

    full_dtype = production.histogram_cherenkov_pool_report_dtype()

    out = dynamicsizerecarray.DynamicSizeRecarray(dtype=dtype)

    with tarfile.open(path, "r|") as itf:
        for tarinfo in itf:
            payload_gz = itf.extractfile(tarinfo).read()
            payload = gzip.decompress(payload_gz)
            reports_block = np.frombuffer(buffer=payload, dtype=full_dtype)

            if mask_function is not None:
                mask = mask_function(reports_block)
                reports_block = reports_block[mask]

            reports_block_out = recarray_init(
                dtype=dtype,
                size=reports_block.size,
            )
            for key in dtype:
                name = key[0]
                reports_block_out[name] = reports_block[name]

            out.append(reports_block_out)

    return out.to_recarray()


class MaskCherenkovInCone:
    def __init__(self, azimuth_rad, zenith_rad, half_angle_rad):
        self.azimuth_rad = azimuth_rad
        self.zenith_rad = zenith_rad
        self.half_angle_rad = half_angle_rad
        self.cx, self.cy, self.cz = spherical_coordinates.az_zd_to_cx_cy_cz(
            azimuth_rad=self.azimuth_rad, zenith_rad=self.zenith_rad
        )

    def __call__(self, reports):
        theta = spherical_coordinates.angle_between_cx_cy(
            cx1=self.cx,
            cy1=self.cy,
            cx2=reports["cherenkov_cx_modus"],
            cy2=reports["cherenkov_cy_modus"],
        )
        return theta <= self.half_angle_rad


class MaskPrimaryInCone:
    def __init__(self, azimuth_rad, zenith_rad, half_angle_rad):
        self.azimuth_rad = azimuth_rad
        self.zenith_rad = zenith_rad
        self.half_angle_rad = half_angle_rad
        self.cx, self.cy, self.cz = spherical_coordinates.az_zd_to_cx_cy_cz(
            azimuth_rad=self.azimuth_rad, zenith_rad=self.zenith_rad
        )

    def __call__(self, reports):
        theta = spherical_coordinates.angle_between_cx_cy(
            cx1=self.cx,
            cy1=self.cy,
            cx2=reports["particle_cx"],
            cy2=reports["particle_cy"],
        )
        return theta <= self.half_angle_rad


def recarray_init(dtype, size):
    return np.recarray(
        shape=size,
        dtype=dtype,
    )
