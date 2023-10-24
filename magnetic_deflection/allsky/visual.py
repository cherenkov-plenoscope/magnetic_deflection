import sebastians_matplotlib_addons as sebplt
import svg_cartesian_plot as splt
import numpy as np
import os
import shutil


def make_deflection_video(
    allsky,
    work_dir,
    num_frames=30,
    num_traces=1000,
    min_num_cherenkov_photons=1e3,
):
    energies_GeV = np.geomspace(
        allsky.binning.energy["stop"],
        allsky.binning.energy["start"],
        num_frames,
    )

    os.makedirs(work_dir, exist_ok=True)

    for ii in np.arange(0, num_frames, 1):
        f1 = ii
        f2 = (-1) * (ii - num_frames) + (num_frames - 1)

        energy_GeV = energies_GeV[ii]

        allsky.plot_deflection(
            path=os.path.join(work_dir, "{:06d}.svg".format(f1)),
            energy_GeV=energy_GeV,
            energy_factor=0.1,
            num_traces=num_traces,
            min_num_cherenkov_photons=min_num_cherenkov_photons,
        )

        splt.bitmap.render(
            os.path.join(work_dir, "{:06d}.svg".format(f1)),
            os.path.join(work_dir, "{:06d}.png".format(f1)),
            background_opacity=1.0,
        )
        os.remove(os.path.join(work_dir, "{:06d}.svg".format(f1)))

        shutil.copy(
            os.path.join(work_dir, "{:06d}.png".format(f1)),
            os.path.join(work_dir, "{:06d}.png".format(f2)),
        )

    sebplt.video.write_video_from_image_slices(
        image_sequence_wildcard_path=os.path.join(work_dir, "%06d.png"),
        output_path=os.path.join(work_dir, "distortion.mov"),
        frames_per_second=30,
        threads=1,
    )
