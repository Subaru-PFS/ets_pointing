import argparse
import os
import tempfile
import time
from re import I

import ets_fiber_assigner.netflow as nf
import matplotlib.path as mppath
import numpy as np
import pandas as pd
import pfs.datamodel
import psycopg2
import psycopg2.extras
import toml
from astroplan import FixedTarget
from astroplan import Observer
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.table import Table
from astropy.time import Time
from ets_shuffle import query_utils
from ets_shuffle.convenience import flag_close_pairs
from ets_shuffle.convenience import guidecam_geometry
from ets_shuffle.convenience import update_coords_for_proper_motion
from ics.cobraOps.Bench import Bench
from ics.cobraOps.BlackDotsCalibrationProduct import BlackDotsCalibrationProduct
from ics.cobraOps.cobraConstants import NULL_TARGET_ID
from ics.cobraOps.cobraConstants import NULL_TARGET_POSITION
from ics.cobraOps.CollisionSimulator2 import CollisionSimulator2
from ics.cobraOps.TargetGroup import TargetGroup
from pfs.utils.coordinates.CoordTransp import CoordinateTransform as ctrans
from pfs.utils.coordinates.CoordTransp import ag_pfimm_to_pixel
from pfs.utils.fiberids import FiberIds
from pfs.utils.pfsDesignUtils import makePfsDesign
from procedures.moduleTest.cobraCoach import CobraCoach
from targetdb import targetdb

from pointing_utils.dbutils import connect_subaru_gaiadb


def generate_pfs_design(
    df_targets,
    df_fluxstds,
    df_sky,
    vis,
    tp,
    tel,
    tgt,
    tgt_class_dict,
    arms="br",
    n_fiber=2394,
    df_raster=None,
    is_no_target=False,
    design_name=None,
):
    is_raster = df_raster is not None

    # n_fiber = len(FiberIds().scienceFiberId)
    # NOTE: fiberID starts with 1 (apparently; TBC).
    # fiber_id = np.arange(n_fiber, dtype=int) + 1

    idx_array = np.arange(n_fiber)

    ra = np.full(n_fiber, np.nan)
    dec = np.full(n_fiber, np.nan)
    pfi_nominal = np.full((n_fiber, 2), [np.nan, np.nan])
    cat_id = np.full(n_fiber, -1, dtype=int)
    obj_id = np.full(n_fiber, -1, dtype=np.int64)
    target_type = np.full(n_fiber, 4, dtype=int)  # filled as unassigned number

    filter_band_names = ["g", "r", "i"]
    flux_default_values = np.full(len(filter_band_names), np.nan)
    filter_default_values = ["none" for _ in filter_band_names]

    # print(flux_default_values)
    # print(filter_default_values)

    dict_of_flux_lists = {
        "fiber_flux": [flux_default_values for _ in range(n_fiber)],
        "total_flux": [flux_default_values for _ in range(n_fiber)],
        "psf_flux": [flux_default_values for _ in range(n_fiber)],
        "filter_names": [filter_default_values for _ in range(n_fiber)],
    }
    # dict_of_flux_lists = {
    #     "fiber_flux": [np.array([np.nan, np.nan, np.nan])] * n_fiber,
    #     "total_flux": [np.array([np.nan, np.nan, np.nan])] * n_fiber,
    #     "psf_flux": [np.array([np.nan, np.nan, np.nan])] * n_fiber,
    #     "fiber_flux_err": [np.array([np.nan, np.nan, np.nan])] * n_fiber,
    #     "total_flux_err": [np.array([np.nan, np.nan, np.nan])] * n_fiber,
    #     "psf_flux_err": [np.array([np.nan, np.nan, np.nan])] * n_fiber,
    #     "filter_names": [["none", "none", "none"]] * n_fiber,
    # }

    # print(dict_of_flux_lists)

    # total_flux = [np.array([np.nan, np.nan, np.nan])] * n_fiber
    # psf_flux = [np.array([np.nan, np.nan, np.nan])] * n_fiber
    # filter_names = [["none", "none", "none"]] * n_fiber

    if not is_no_target:

        gfm = FiberIds()  # 2604
        cobra_ids = gfm.cobraId
        scifiber_ids = gfm.scienceFiberId

        for tidx, cidx in vis.items():

            # print(cidx)

            idx_fiber = (
                cobra_ids[np.logical_and(scifiber_ids >= 0, scifiber_ids <= n_fiber)]
                == cidx + 1
            )

            i_fiber = idx_array[idx_fiber][0]

            ra[idx_fiber] = tgt[tidx].ra
            dec[idx_fiber] = tgt[tidx].dec
            # netflow's Target class convert object IDs to string.
            obj_id[idx_fiber] = np.int64(tgt[tidx].ID)
            pfi_nominal[idx_fiber] = [tp[tidx].real, tp[tidx].imag]
            target_type[idx_fiber] = tgt_class_dict[tgt[tidx].targetclass]

            idx_target = np.logical_and(
                df_targets["obj_id"] == np.int64(tgt[tidx].ID),
                df_targets["target_type_id"] == tgt_class_dict[tgt[tidx].targetclass],
            )
            idx_fluxstd = np.logical_and(
                df_fluxstds["obj_id"] == np.int64(tgt[tidx].ID),
                df_fluxstds["target_type_id"] == tgt_class_dict[tgt[tidx].targetclass],
            )

            if np.any(idx_target):
                cat_id[i_fiber] = df_targets["input_catalog_id"][idx_target].values[0]
                # dict_of_flux_lists["total_flux"][i_fiber] = [
                #     np.nan for _ in filter_band_names
                # ]
                dict_of_flux_lists["psf_flux"][i_fiber] = np.array(
                    [
                        df_targets[f"psf_flux_{band}"][idx_target].values[0]
                        if df_targets[f"psf_flux_{band}"][idx_target].values[0]
                        is not None
                        else np.nan
                        for band in filter_band_names
                    ]
                )
                dict_of_flux_lists["filter_names"][i_fiber] = [
                    df_targets[f"filter_{band}"][idx_target].values[0]
                    if df_targets[f"filter_{band}"][idx_target].values[0] is not None
                    else "none"
                    for band in filter_band_names
                ]
                # total_flux[i_fiber] = df_targets["totalFlux"][idx_target][0]
                # filter_names[i_fiber] = df_targets["filterNames"][idx_target][0].tolist()
            if np.any(idx_fluxstd):
                cat_id[i_fiber] = df_fluxstds["input_catalog_id"][idx_fluxstd].values[0]
                # dict_of_flux_lists["total_flux"][i_fiber] = [
                #     np.nan for band in filter_band_names
                # ]
                dict_of_flux_lists["psf_flux"][i_fiber] = np.array(
                    [
                        df_fluxstds[f"psf_flux_{band}"][idx_fluxstd].values[0]
                        if df_fluxstds[f"psf_flux_{band}"][idx_fluxstd].values[0]
                        is not None
                        else np.nan
                        for band in filter_band_names
                    ]
                )
                dict_of_flux_lists["filter_names"][i_fiber] = [
                    df_fluxstds[f"filter_{band}"][idx_fluxstd].values[0]
                    if df_fluxstds[f"filter_{band}"][idx_fluxstd].values[0] is not None
                    else "none"
                    for band in filter_band_names
                ]
            # psf_flux[i_fiber] = df_fluxstds["psfFlux"][idx_fluxstd][0]
            # filter_names[i_fiber] = df_fluxstds["filterNames"][idx_fluxstd][0].tolist()

            if is_raster:
                idx_raster = np.logical_and(
                    df_raster["obj_id"] == np.int64(tgt[tidx].ID),
                    df_raster["target_type_id"]
                    == tgt_class_dict[tgt[tidx].targetclass],
                )
                if np.any(idx_raster):
                    cat_id[i_fiber] = df_raster["input_catalog_id"][idx_raster].values[
                        0
                    ]
                    dict_of_flux_lists["psf_flux"][i_fiber] = np.array(
                        [
                            df_raster["g_flux_njy"][idx_raster].values[0],
                            df_raster["bp_flux_njy"][idx_raster].values[0],
                            df_raster["rp_flux_njy"][idx_raster].values[0],
                        ]
                    )
                    dict_of_flux_lists["filter_names"][i_fiber] = [
                        "g_gaia",
                        "bp_gaia",
                        "rp_gaia",
                    ]

    # print(dict_of_flux_lists)

    pfs_design = makePfsDesign(
        pfi_nominal,
        ra,
        dec,
        raBoresight=tel._ra,
        decBoresight=tel._dec,
        posAng=tel._posang,
        arms=arms,
        # tract=1,
        # patch="1,1",
        catId=cat_id,
        objId=obj_id,
        targetType=target_type,
        # fiberStatus=FiberStatus.GOOD,
        # fiberFlux=dict_of_flux_lists["fiber_flux"],
        psfFlux=dict_of_flux_lists["psf_flux"],
        # psfFlux=psf_flux,
        # totalFlux=dict_of_flux_lists["total_flux"],
        # fiberFluxErr=np.NaN,
        # psfFluxErr=np.NaN,
        # totalFluxErr=np.NaN,
        filterNames=dict_of_flux_lists["filter_names"],
        # filterNames=filter_names,
        # guideStars=None,
        designName=design_name,
    )

    return pfs_design


def generate_guidestars_from_gaiadb(
    ra,
    dec,
    pa,
    observation_time,
    telescope_elevation=None,
    conf=None,
    guidestar_mag_min=12.0,
    guidestar_mag_max=19.0,
    guidestar_neighbor_mag_min=21.0,
    guidestar_minsep_deg=1.0 / 3600,
    fp_radius_degree=260.0 * 10.2 / 3600,  # "Radius" of PFS FoV in degree (?)
    fp_fudge_factor=1.5,  # fudge factor for search widths
    search_radius=None,
    # gaiadb_epoch=2015.0,
    gaiadb_input_catalog_id=2,
):
    # Get ra, dec and position angle from input arguments
    ra_tel_deg, dec_tel_deg, pa_deg = ra, dec, pa

    # Get telescope elevation from the observing time, target, and pointing
    pointing_center = FixedTarget(SkyCoord(ra * u.deg, dec * u.deg, frame="icrs"))
    observing_site = Observer.at_site("subaru")
    if telescope_elevation is None:
        telescope_elevation = observing_site.altaz(
            observation_time,
            pointing_center,
        ).alt.value
        print(
            f"Telescope elevation is set to {telescope_elevation:.1f} degrees from the pointing center ({ra:.5f}, {dec:.5f}) and observing time {observation_time} at Subaru Telescope"
        )

    # guidestar_mag_max = guidestar_mag_max
    # guidestar_neighbor_mag_min = guidestar_neighbor_mag_min
    # guidestar_minsep_deg = guidestar_minsep_deg

    # guide star cam geometries
    agcoord = guidecam_geometry()

    # internal, technical parameters
    # set focal plane radius
    # fp_radius_degree = 260.0 * 10.2 / 3600
    # fp_fudge_factor = 1.2

    if search_radius is None:
        search_radius = fp_radius_degree * fp_fudge_factor

    coldict = {
        "id": "source_id",
        "ra": "ra",
        "dec": "dec",
        "parallax": "parallax",
        "pmra": "pmra",
        "pmdec": "pmdec",
        "epoch": "ref_epoch",
        "mag": "phot_g_mean_mag",
        "color": "bp_rp",
    }
    racol, deccol = coldict["ra"], coldict["dec"]

    # Find guide star candidates
    conn = connect_subaru_gaiadb(conf)
    cur = conn.cursor()

    query_string = f"""SELECT source_id,ra,dec,parallax,pmra,pmdec,ref_epoch,phot_g_mean_mag,bp_rp
    FROM gaia3
    WHERE q3c_radial_query(ra, dec, {ra_tel_deg}, {dec_tel_deg}, {search_radius})
    AND {coldict['pmra']} IS NOT NULL
    AND {coldict['pmdec']} IS NOT NULL
    AND {coldict['parallax']} IS NOT NULL
    AND {coldict['parallax']} >= 0
    AND astrometric_excess_noise_sig < 2.0
    AND {coldict['mag']} BETWEEN {0.0} AND {guidestar_neighbor_mag_min}
    ;
    """
    print(query_string)
    cur.execute(query_string)

    df_res = pd.DataFrame(
        cur.fetchall(),
        columns=[
            "source_id",
            "ra",
            "dec",
            "parallax",
            "pmra",
            "pmdec",
            "ref_epoch",
            "phot_g_mean_mag",
            "bp_rp",
        ],
    )
    cur.close()
    conn.close()

    assert (
        np.unique(df_res["ref_epoch"]).size == 1
    ), "Non-unique epochs for sources from GaiaDB"

    gaiadb_epoch = np.unique(df_res["ref_epoch"])[0]

    res = {}
    for col in df_res.columns:
        res[col] = df_res[col].to_numpy()

    # MO: I'm not sure if the following FIXME is still valid or not.
    # # FIXME: run similar query, but without the PM requirement, to get a list of
    # # potentially too-bright neighbours

    # # adjust for proper motion
    # epoch = Time(observation_time).jyear
    # res[racol], res[deccol] = update_coords_for_proper_motion(
    #     res[racol],
    #     res[deccol],
    #     res[coldict["pmra"]],
    #     res[coldict["pmdec"]],
    #     gaiadb_epoch,  # Gaia DR2 uses 2015.5
    #     epoch,
    # )

    # compute PFI coordinates
    tmp = np.array([res[racol], res[deccol]])
    tmp = ctrans(
        xyin=tmp,
        mode="sky_pfi",
        pa=pa_deg,
        cent=np.array([ra_tel_deg, dec_tel_deg]).reshape((2, 1)),
        pm=np.stack([res[coldict["pmra"]], res[coldict["pmdec"]]], axis=0),
        par=res[coldict["parallax"]],
        time=observation_time,
        epoch=gaiadb_epoch,
    )

    res["xypos"] = np.array([tmp[0, :], tmp[1, :]]).T

    # determine the subset of sources falling within the guide cam FOVs
    # For the moment I'm using matplotlib's path functionality for this task
    # Once the "pfi_sky" transformation direction is available in
    # pfs_utils.coordinates, we can do a direct polygon query for every camera,
    # which should be more efficient.
    targets = {}
    tgtcam = []

    for i in range(agcoord.shape[0]):

        p = mppath.Path(agcoord[i])

        # find all targets in the slighty enlarged FOV
        tmp = p.contains_points(res["xypos"], radius=1.0)  # 1mm more
        tdict = {}

        for key, val in res.items():
            tdict[key] = val[tmp]

        # eliminate close neighbors
        flags = flag_close_pairs(tdict[racol], tdict[deccol], guidestar_minsep_deg)

        for key, val in tdict.items():
            tdict[key] = val[np.invert(flags)]

        # eliminate all targets which are not bright enough to be guide stars
        flags = (tdict["phot_g_mean_mag"] > guidestar_mag_min) * (
            tdict["phot_g_mean_mag"] < guidestar_mag_max
        )

        for key, val in tdict.items():
            tdict[key] = val[flags]

        # eliminate all targets which are not really in the camera's FOV
        flags = p.contains_points(tdict["xypos"])  # exact size

        for key, val in tdict.items():
            tdict[key] = val[flags]

        # add AG camera ID
        tdict["agid"] = [i] * len(tdict[coldict["id"]])

        # compute and add pixel coordinates
        tmp = []
        for pos in tdict["xypos"]:
            tmp.append(ag_pfimm_to_pixel(i, pos[0], pos[1]))
        tdict["agpix_x"] = np.array([x[0] for x in tmp])
        tdict["agpix_y"] = np.array([x[1] for x in tmp])

        # append the results for this camera to the full list
        tgtcam.append(tdict)
        for key, val in tdict.items():
            if key not in targets:
                targets[key] = val
            else:
                targets[key] = np.concatenate((targets[key], val))

    # Write the results to a new pfsDesign file. Data fields are according to
    # DAMD-101.
    # required data:
    # ra/dec of guide star candidates: in racol, deccol
    # PM information: in pmra, pmdec
    # parallax: currently N/A
    # flux: currently N/A
    # AgId: trivial to obtain from data structure
    # AgX, AgY (pixel coordinates): only computable with access to the full
    #   AG camera geometry
    # output_design = input_design

    ntgt = len(targets[coldict["id"]])

    guidestars = pfs.datamodel.guideStars.GuideStars(
        targets[coldict["id"]],
        np.full(ntgt, f"J{gaiadb_epoch:.1f}"),
        # np.full(ntgt, "J{:.1f}".format(epoch)),  # convert float epoch to string
        # FIXME: the ra/dec values below are _not_ corrected for proper motion
        #        any more! If corrected values are required, we might need
        #        a new mode "sky_skycorrected" (or similar)
        #        for pfs.utils.CoordinateTransform.
        #        On the other hand, since we store catalog and object ID here,
        #        most other columns are redundant anyway.
        targets[coldict["ra"]],
        targets[coldict["dec"]],
        targets[coldict["pmra"]],
        targets[coldict["pmdec"]],
        targets[coldict["parallax"]],
        targets[coldict["mag"]],
        np.full(ntgt, "g_gaia"),  # passband
        targets[coldict["color"]],  # color
        targets["agid"],  # AG camera ID
        targets["agpix_x"],  # AG x pixel coordinate
        targets["agpix_y"],  # AG y pixel coordinate
        telescope_elevation,
        gaiadb_input_catalog_id,  # numerical ID assigned to the GAIA catalogue
    )

    return guidestars
