import spherical_coordinates
import corsika_primary
import numpy as np


def Query(
    azimuth_rad,
    zenith_rad,
    half_angle_rad,
    energy_start_GeV,
    energy_stop_GeV,
):
    return {
        "azimuth_rad": azimuth_rad,
        "zenith_rad": zenith_rad,
        "half_angle_rad": half_angle_rad,
        "energy_start_GeV": energy_start_GeV,
        "energy_stop_GeV": energy_stop_GeV,
    }


def example_deg(min_energy_GeV=0.25, max_energy_GeV=64.0, num=1):
    minE = min_energy_GeV
    maxE = max_energy_GeV
    qs = []
    tr = []
    # zenith, go to low energies
    qs.append([0.0, 0.0, 3.25, 32, 64])
    tr.append((24 * num, "direct"))
    qs.append([0.0, 0.0, 3.25, 0.5, 1.0])
    tr.append((2 * num, "direct"))
    qs.append([0.0, 0.0, 3.25, 0.25, 0.5])
    # reverse
    tr.append((2 * num, "direct"))
    qs.append([0.0, 0.0, 3.25, 0.5, 1.0])
    # narrow energy
    tr.append((2 * num, "direct"))
    qs.append([0.0, 0.0, 3.25, 0.5, 0.625])

    # move to zenith 30 deg
    tr.append((3 * num, "polar"))
    qs.append([0.0, 30.0, 3.25, 0.5, 0.625])
    tr.append((36 * num, "polar"))
    qs.append([360.0, 30.0, 3.25, 0.5, 0.625])
    tr.append((0, "polar"))
    qs.append([0.0, 30.0, 3.25, 0.5, 0.625])

    # higher energies
    tr.append((3 * num, "direct"))
    qs.append([0.0, 30.0, 3.25, 5, 7.5])

    tr.append((3 * num, "polar"))
    qs.append([0.0, 60.0, 3.25, 5, 7.5])
    tr.append((36 * num, "polar"))
    qs.append([360.0, 60.0, 3.25, 5, 7.5])
    tr.append((0, "polar"))
    qs.append([0.0, 60.0, 3.25, 5, 7.5])

    tr.append((3 * num, "polar"))
    qs.append([0.0, 0.0, 3.25, 32, 64])

    qs = np.asarray(qs)
    Elow = qs[:, 3]
    Ehig = qs[:, 4]

    _Elow = (Elow - 0.25) / (64 - 0.25)
    _Ehig = (Ehig - 0.25) / (64 - 0.25)

    _Elow = minE + (_Elow * (maxE - minE))
    _Ehig = minE + (_Ehig * (maxE - minE))
    qs[:, 3] = _Elow
    qs[:, 4] = _Ehig

    return qs, tr


def random(
    prng,
    max_zenith_distance_rad,
    half_angle_rad,
    energy_start_GeV,
    energy_stop_GeV,
    size=1,
):
    out = []
    for i in range(size):
        az, zd = spherical_coordinates.random.uniform_az_zd_in_cone(
            prng=prng,
            azimuth_rad=0.0,
            zenith_rad=0.0,
            min_half_angle_rad=0.0,
            max_half_angle_rad=max_zenith_distance_rad,
        )
        energy = corsika_primary.random.distributions.draw_power_law(
            prng=prng,
            lower_limit=energy_start_GeV,
            upper_limit=energy_stop_GeV,
            power_slope=-1.5,
            num_samples=None,
        )
        q = Query(
            azimuth_rad=az,
            zenith_rad=zd,
            half_angle_rad=half_angle_rad,
            energy_start_GeV=energy * 0.95,
            energy_stop_GeV=energy * 1.05,
        )
        out.append(q)
    return out


def compile_deg(queries_deg=example_deg()):
    qs, tr = queries_deg
    queries = []
    transissions = []
    for i in range(len(qs)):
        queries.append(
            Query(
                azimuth_rad=np.deg2rad(qs[i][0]),
                zenith_rad=np.deg2rad(qs[i][1]),
                half_angle_rad=np.deg2rad(qs[i][2]),
                energy_start_GeV=qs[i][3],
                energy_stop_GeV=qs[i][4],
            )
        )
    for i in range(len(tr)):
        transissions.append({"num": tr[i][0], "skymode": tr[i][1]})
    return queries, transissions


def example(**kwargs):
    qs, tr = compile_deg(queries_deg=example_deg(**kwargs))
    return interpolate(queries=qs, transissions=tr)


def interpolate(queries, transissions):
    all_steps = []
    for i in range(len(queries) - 1):
        steps = linspace(
            start=queries[i],
            stop=queries[i + 1],
            num=transissions[i]["num"],
            endpoint=False,
            skymode=transissions[i]["skymode"],
        )
        all_steps += steps
    return all_steps


def linspace(start, stop, num, endpoint=True, skymode="direct"):
    if skymode == "direct":
        acx, acy, acz = spherical_coordinates.az_zd_to_cx_cy_cz(
            azimuth_rad=start["azimuth_rad"], zenith_rad=start["zenith_rad"]
        )
        bcx, bcy, bcz = spherical_coordinates.az_zd_to_cx_cy_cz(
            azimuth_rad=stop["azimuth_rad"], zenith_rad=stop["zenith_rad"]
        )

        # good enough
        for i in range(num):
            scx = np.linspace(acx, bcx, num=num, endpoint=endpoint)
            scy = np.linspace(acy, bcy, num=num, endpoint=endpoint)
            scz = np.linspace(acz, bcz, num=num, endpoint=endpoint)

        norms = np.sqrt(scx**2 + scy**2 + scz**2)
        scx /= norms
        scy /= norms
        scz /= norms

        azimuth_rad, zenith_rad = spherical_coordinates.cx_cy_cz_to_az_zd(
            cx=scx, cy=scy, cz=scz
        )
    elif skymode == "polar":
        azimuth_rad = np.linspace(
            start=start["azimuth_rad"],
            stop=stop["azimuth_rad"],
            num=num,
            endpoint=endpoint,
        )
        zenith_rad = np.linspace(
            start=start["zenith_rad"],
            stop=stop["zenith_rad"],
            num=num,
            endpoint=endpoint,
        )
    else:
        raise ValueError("No such sky mode '{:s}'.".format(sky))

    half_angle_rad = np.linspace(
        start=start["half_angle_rad"],
        stop=stop["half_angle_rad"],
        num=num,
        endpoint=endpoint,
    )

    energy_start_GeV = np.geomspace(
        start=start["energy_start_GeV"],
        stop=stop["energy_start_GeV"],
        num=num,
        endpoint=endpoint,
    )

    energy_stop_GeV = np.geomspace(
        start=start["energy_stop_GeV"],
        stop=stop["energy_stop_GeV"],
        num=num,
        endpoint=endpoint,
    )

    out = []
    for i in range(len(half_angle_rad)):
        q = Query(
            azimuth_rad=azimuth_rad[i],
            zenith_rad=zenith_rad[i],
            half_angle_rad=half_angle_rad[i],
            energy_start_GeV=energy_start_GeV[i],
            energy_stop_GeV=energy_stop_GeV[i],
        )
        out.append(q)
    return out
