import os

import matplotlib.path as mppath
import numpy as np
import pandas as pd
import pfs.datamodel
from astroplan import FixedTarget, Observer
from astropy import units as u
from astropy.coordinates import SkyCoord
from ets_shuffle.convenience import flag_close_pairs, guidecam_geometry
from logzero import logger
from pfs.utils.coordinates.CoordTransp import CoordinateTransform as ctrans
from pfs.utils.coordinates.CoordTransp import ag_pfimm_to_pixel
from pfs.utils.fiberids import FiberIds
from pfs.utils.pfsDesignUtils import makePfsDesign, setFiberStatus

from ..utils import get_pfs_utils_path
from .dbutils import connect_subaru_gaiadb


def generate_pfs_design(
    df_targets,
    df_fluxstds,
    df_sky,
    vis,
    tp,
    tel,
    tgt,
    tgt_class_dict,
    bench,  # won't be used
    arms="br",
    n_fiber=2394,
    df_filler=None,
    is_no_target=False,
    design_name=None,
    pfs_instdata_dir=None,
    obs_time="",
):

    gfm = FiberIds(path=get_pfs_utils_path())  # 2604
    cobra_ids = gfm.cobraId
    scifiber_ids = gfm.scienceFiberId

    is_filler = df_filler is not None

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
    # fiber_status = np.full(n_fiber, 1, dtype=int)  # filled as GOOD=1
    # for cidx in range(n_fiber):
    #     fidx = (
    #         cobra_ids[np.logical_and(scifiber_ids >= 0, scifiber_ids <= n_fiber)]
    #         == cidx + 1
    #     )
    #     fiber_status[fidx] = bench.cobras.status[cidx]
    # fiber_status[fiber_status > 1] = 2  # filled bad fibers as BROKENFIBER=2

    proposal_id = ["N/A" for _ in range(n_fiber)]
    ob_code = ["N/A" for _ in range(n_fiber)]
    epoch = ["J2000.0" for _ in range(n_fiber)]
    # proposal_id = np.full(len(n_fiber), "N/A", dtype="<U32")
    # ob_code = np.full(len(n_fiber), "N/A", dtype="<U64")
    # epoch = np.full(len(n_fiber), "J2000.0")
    pmRa = np.zeros(n_fiber, dtype=np.float32)
    pmDec = np.zeros(n_fiber, dtype=np.float32)
    parallax = np.full(n_fiber, 1.0e-5, dtype=np.float32)

    filter_band_names = ["g", "r", "i", "z", "y"]
    flux_default_values = np.full(len(filter_band_names), np.nan)
    filter_default_values = ["none" for _ in filter_band_names]

    # print(flux_default_values)
    # print(filter_default_values)

    dict_of_flux_lists = {
        "fiber_flux": [flux_default_values for _ in range(n_fiber)],
        "total_flux": [flux_default_values for _ in range(n_fiber)],
        "psf_flux": [flux_default_values for _ in range(n_fiber)],
        "fiber_flux_error": [flux_default_values for _ in range(n_fiber)],
        "psf_flux_error": [flux_default_values for _ in range(n_fiber)],
        "total_flux_error": [flux_default_values for _ in range(n_fiber)],
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
            # obj_id[idx_fiber] = np.int64(tgt[tidx].ID)
            obj_id[idx_fiber] = np.int64(tgt[tidx].ID.split("_")[0])
            pfi_nominal[idx_fiber] = [tp[tidx].real, tp[tidx].imag]
            target_type[idx_fiber] = tgt_class_dict[tgt[tidx].targetclass]

            idx_target = np.logical_and(
                # df_targets["obj_id"] == np.int64(tgt[tidx].ID),
                df_targets["obj_id"].map(str)
                + "_"
                + df_targets["input_catalog_id"].map(str)
                == tgt[tidx].ID,
                df_targets["target_type_id"] == tgt_class_dict[tgt[tidx].targetclass],
            )
            idx_fluxstd = np.logical_and(
                # df_fluxstds["obj_id"] == np.int64(tgt[tidx].ID),
                df_fluxstds["obj_id"].map(str)
                + "_"
                + df_fluxstds["input_catalog_id"].map(str)
                == tgt[tidx].ID,
                df_fluxstds["target_type_id"] == tgt_class_dict[tgt[tidx].targetclass],
            )
            idx_sky = np.logical_and(
                # df_sky["obj_id"] == np.int64(tgt[tidx].ID),
                df_sky["obj_id"].map(str) + "_" + df_sky["input_catalog_id"].map(str)
                == tgt[tidx].ID,
                df_sky["target_type_id"] == tgt_class_dict[tgt[tidx].targetclass],
            )

            if np.any(idx_target):
                proposal_id[i_fiber] = df_targets["proposal_id"][idx_target].values[0]
                ob_code[i_fiber] = df_targets["ob_code"][idx_target].values[0]
                epoch[i_fiber] = df_targets["epoch"][idx_target].values[0]
                pmRa[i_fiber] = df_targets["pmra"][idx_target].values[0]
                pmDec[i_fiber] = df_targets["pmdec"][idx_target].values[0]
                parallax[i_fiber] = df_targets["parallax"][idx_target].values[0]

                cat_id[i_fiber] = df_targets["input_catalog_id"][idx_target].values[0]
                # dict_of_flux_lists["total_flux"][i_fiber] = [
                #     np.nan for _ in filter_band_names
                # ]

                dict_of_flux_lists["psf_flux"][i_fiber] = np.array(
                    [
                        (
                            df_targets[f"psf_flux_{band}"][idx_target].values[0]
                            if (
                                df_targets[f"psf_flux_{band}"][idx_target].values[0]
                                is not None
                            )
                            and (
                                df_targets[f"psf_flux_{band}"][idx_target].values[0]
                                > 0.0
                            )
                            else np.nan
                        )
                        for band in filter_band_names
                    ]
                )
                dict_of_flux_lists["psf_flux_error"][i_fiber] = np.array(
                    [
                        (
                            df_targets[f"psf_flux_error_{band}"][idx_target].values[0]
                            if (
                                df_targets[f"psf_flux_error_{band}"][idx_target].values[
                                    0
                                ]
                                is not None
                            )
                            and (
                                df_targets[f"psf_flux_error_{band}"][idx_target].values[
                                    0
                                ]
                                > 0.0
                            )
                            else np.nan
                        )
                        for band in filter_band_names
                    ]
                )
                msk = dict_of_flux_lists["psf_flux_error"][i_fiber] <= 0
                dict_of_flux_lists["psf_flux_error"][i_fiber][msk] = np.nan
                # FIXME: filter names should be in targetDB
                if cat_id[i_fiber] >= 5 and cat_id[i_fiber] <= 12:
                    dict_of_flux_lists["filter_names"][i_fiber] = [
                        "g_hsc",
                        "r2_hsc",
                        "i2_hsc",
                        "z_hsc",
                        "y_hsc",
                    ]
                else:
                    dict_of_flux_lists["filter_names"][i_fiber] = [
                        (
                            df_targets[f"filter_{band}"][idx_target].values[0]
                            if df_targets[f"filter_{band}"][idx_target].values[0]
                            is not None
                            else "none"
                        )
                        for band in filter_band_names
                    ]
                # FIXME: temporal workaround for co_field1
                if design_name == "field1_pa+00":
                    dict_of_flux_lists["psf_flux"][i_fiber] = [
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                    ]
                    dict_of_flux_lists["psf_flux_error"][i_fiber] = [
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                    ]
                # total_flux[i_fiber] = df_targets["totalFlux"][idx_target][0]
                # filter_names[i_fiber] = df_targets["filterNames"][idx_target][0].tolist()
            if np.any(idx_fluxstd):
                epoch[i_fiber] = df_fluxstds["epoch"][idx_fluxstd].values[0]
                pmRa[i_fiber] = df_fluxstds["pmra"][idx_fluxstd].values[0]
                pmDec[i_fiber] = df_fluxstds["pmdec"][idx_fluxstd].values[0]
                parallax[i_fiber] = df_fluxstds["parallax"][idx_fluxstd].values[0]

                cat_id[i_fiber] = df_fluxstds["input_catalog_id"][idx_fluxstd].values[0]
                # dict_of_flux_lists["total_flux"][i_fiber] = [
                #     np.nan for band in filter_band_names
                # ]
                dict_of_flux_lists["psf_flux"][i_fiber] = np.array(
                    [
                        (
                            df_fluxstds[f"psf_flux_{band}"][idx_fluxstd].values[0]
                            if df_fluxstds[f"psf_flux_{band}"][idx_fluxstd].values[0]
                            is not None
                            else np.nan
                        )
                        for band in filter_band_names
                    ]
                )
                dict_of_flux_lists["psf_flux_error"][i_fiber] = np.array(
                    [
                        (
                            df_fluxstds[f"psf_flux_error_{band}"][idx_fluxstd].values[0]
                            if df_fluxstds[f"psf_flux_error_{band}"][idx_fluxstd].values[0]
                            is not None
                            else np.nan
                        )
                        for band in filter_band_names
                    ]
                )
                msk = dict_of_flux_lists["psf_flux_error"][i_fiber] <= 0
                dict_of_flux_lists["psf_flux_error"][i_fiber][msk] = np.nan
                dict_of_flux_lists["filter_names"][i_fiber] = [
                    (
                        df_fluxstds[f"filter_{band}"][idx_fluxstd].values[0]
                        if df_fluxstds[f"filter_{band}"][idx_fluxstd].values[0]
                        is not None
                        else "none"
                    )
                    for band in filter_band_names
                ]

                # FIXME: temporal fix for gaia fluxstds
                #if "g_gaia" in dict_of_flux_lists["filter_names"][i_fiber]:
                #    dict_of_flux_lists["filter_names"][i_fiber] = list(dict_of_flux_lists["filter_names"][i_fiber][:3])
                #    dict_of_flux_lists["psf_flux"][i_fiber] = np.array(dict_of_flux_lists["psf_flux"][i_fiber][:3])
                #    dict_of_flux_lists["psf_flux_error"][i_fiber] = np.array(dict_of_flux_lists["psf_flux_error"][i_fiber][:3])
                #    dict_of_flux_lists["filter_names"][i_fiber][3] = 'none'
                #    dict_of_flux_lists["filter_names"][i_fiber][4] = 'none'

            # psf_flux[i_fiber] = df_fluxstds["psfFlux"][idx_fluxstd][0]
            # filter_names[i_fiber] = df_fluxstds["filterNames"][idx_fluxstd][0].tolist()
            if np.any(idx_sky):
                cat_id[i_fiber] = df_sky["input_catalog_id"][idx_sky].values[0]
                try:
                    dict_of_flux_lists["psf_flux"][i_fiber] = np.array(
                        [
                            10
                            ** (-0.4 * (df_sky["mag_thresh"][idx_sky].values[0] - 8.9))
                            * 1e09,
                            np.nan,
                            np.nan,
                            np.nan,
                            np.nan,
                        ]
                    )
                except:
                    dict_of_flux_lists["psf_flux"][i_fiber] = np.array(
                        [
                            np.nan,
                            np.nan,
                            np.nan,
                            np.nan,
                            np.nan,
                        ]
                    )
                dict_of_flux_lists["psf_flux_error"][i_fiber] = np.array(
                    [
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                    ]
                )
                dict_of_flux_lists["filter_names"][i_fiber] = [
                    "g_hsc",
                    "none",
                    "none",
                    "none",
                    "none",
                ]
            if is_filler:
                idx_filler = np.logical_and(
                    df_filler["obj_id"].map(str)
                    + "_"
                    + df_filler["input_catalog_id"].map(str)
                    == tgt[tidx].ID,
                    df_filler["target_type_id"]
                    == tgt_class_dict[tgt[tidx].targetclass],
                )
                if np.any(idx_filler):
                    proposal_id[i_fiber] = df_filler["proposal_id"][idx_filler].values[
                        0
                    ]
                    ob_code[i_fiber] = df_filler["ob_code"][idx_filler].values[0]
                    epoch[i_fiber] = df_filler["epoch"][idx_filler].values[0]
                    pmRa[i_fiber] = df_filler["pmra"][idx_filler].values[0]
                    pmDec[i_fiber] = df_filler["pmdec"][idx_filler].values[0]
                    parallax[i_fiber] = df_filler["parallax"][idx_filler].values[0]

                    cat_id[i_fiber] = df_filler["input_catalog_id"][idx_filler].values[
                        0
                    ]
                    dict_of_flux_lists["psf_flux"][i_fiber] = np.array(
                        [
                            df_filler["g_flux_njy"][idx_filler].values[0],
                            df_filler["bp_r_flux_njy"][idx_filler].values[0],
                            df_filler["rp_i_flux_njy"][idx_filler].values[0],
                            np.nan,
                            np.nan,
                        ]
                    )
                    dict_of_flux_lists["psf_flux_error"][i_fiber] = np.array(
                        [
                            df_filler["g_flux_err_njy"][idx_filler].values[0],
                            df_filler["bp_r_flux_err_njy"][idx_filler].values[0],
                            df_filler["rp_i_flux_err_njy"][idx_filler].values[0],
                            np.nan,
                            np.nan,
                        ]
                    )
                    dict_of_flux_lists["psf_flux_error"][i_fiber] = np.array(
                        [
                            df_filler["g_flux_err_njy"][idx_filler].values[0],
                            df_filler["bp_r_flux_err_njy"][idx_filler].values[0],
                            df_filler["rp_i_flux_err_njy"][idx_filler].values[0],
                            np.nan,
                            np.nan,
                        ]
                    )
                    dict_of_flux_lists["filter_names"][i_fiber] = [
                        "g_gaia",
                        "bp_gaia",
                        "rp_gaia",
                        "none",
                        "none",
                    ]

            # print(dict_of_flux_lists)

    for i in range(len(dict_of_flux_lists["filter_names"])):
        dict_of_flux_lists["psf_flux_error"][i][
            dict_of_flux_lists["psf_flux_error"][i] == 0.0
        ] = 1.0e-6

    # print(len(dict_of_flux_lists["psf_flux"]))
    # print(len(dict_of_flux_lists["filter_names"]))
    # with open('tmp1.dat', 'w') as f:
    #    for v1, v2, v3, v4 in zip(obj_id, dict_of_flux_lists["psf_flux"],
    # dict_of_flux_lists["psf_flux_error"], dict_of_flux_lists["filter_names"]):
    #    f.write(f'{v1}  {v2}  {v3}  {v4}\n')

    # sanity check for epoch
    for i, ep in enumerate(epoch):
        if ep[0] != "J":
            epoch[i] = "J" + ep

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
        # fiberStatus=fiber_status,
        epoch=epoch,
        pmRa=pmRa,
        pmDec=pmDec,
        parallax=parallax,
        proposalId=proposal_id,
        obCode=ob_code,
        # fiberFlux=dict_of_flux_lists["fiber_flux"],
        psfFlux=dict_of_flux_lists["psf_flux"],
        # psfFlux=psf_flux,
        # totalFlux=dict_of_flux_lists["total_flux"],
        # fiberFluxErr=np.NaN,
        psfFluxErr=dict_of_flux_lists["psf_flux_error"],
        # totalFluxErr=np.NaN,
        filterNames=dict_of_flux_lists["filter_names"],
        # filterNames=filter_names,
        # guideStars=None,
        designName=design_name,
        fiberidsPath=get_pfs_utils_path(),
        obstime=obs_time,
    )

    # Set the environment variables for the PFS instrument data and utilities directories
    if os.environ.get("PFS_UTILS_DIR") is None:
        pfs_utils_dir = pfs.utils.__path__
        if isinstance(pfs_utils_dir, list):
            pfs_utils_dir = pfs_utils_dir[0].replace("python/pfs/utils", "")
        else:
            pfs_utils_dir = pfs_utils_dir.replace("python/pfs/utils", "")
        fiber_id_path = os.path.join(pfs_utils_dir, "data", "fiberids")
    else:
        fiber_id_path = os.path.join(
            os.environ.get("PFS_UTILS_DIR"), "data", "fiberids"
        )

    if pfs_instdata_dir is None:
        if os.environ.get("PFS_INSTDATA_DIR") is None:
            raise ValueError("PFS_INSTDATA_DIR is not set.")
        pfs_instdata_dir = os.environ.get("PFS_INSTDATA_DIR")

    pfs_instdata_data_dir = os.path.join(pfs_instdata_dir, "data")

    pfs_design = setFiberStatus(
        pfs_design, configRoot=pfs_instdata_data_dir, fiberIdsPath=fiber_id_path
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
    gaiadb_input_catalog_id=4,
    guide_star_id_exclude=[],
    good_astrometry=False,
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
            f"Telescope elevation is set to {telescope_elevation:.1f} degrees \
                from the pointing center ({ra:.5f}, {dec:.5f}) and observing \
                    time {observation_time} at Subaru Telescope"
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

    sqlWhere = ""
    for gsId in guide_star_id_exclude:
        sqlWhere += f"AND source_id NOT EQUAL {gsId}"

    if good_astrometry is True:
        astrometric_flag = f"""AND  {coldict['pmra']} IS NOT NULL   
        AND {coldict['pmdec']} IS NOT NULL
        AND {coldict['parallax']} IS NOT NULL
        AND {coldict['parallax']} >= 0 
        AND astrometric_excess_noise_sig < 2.0 
        """
    else:
        astrometric_flag = ""

    query_string = f"""SELECT source_id,ra,dec,parallax,pmra,pmdec,ref_epoch,phot_g_mean_mag,bp_rp
    FROM gaia3
    WHERE q3c_radial_query(ra, dec, {ra_tel_deg}, {dec_tel_deg}, {search_radius})
    {astrometric_flag} AND {coldict['mag']} BETWEEN {guidestar_mag_min} AND {guidestar_mag_max} {sqlWhere}
    ;
    """
    
    #query_string = f"""SELECT source_id,ra,dec,parallax,pmra,pmdec,ref_epoch,phot_g_mean_mag,bp_rp
    #FROM gaia3
    #WHERE q3c_radial_query(ra, dec, {ra_tel_deg}, {dec_tel_deg}, {search_radius})
    #AND {coldict['mag']} BETWEEN {guidestar_mag_min} AND {guidestar_mag_max} {sqlWhere}
    #;
    #"""
    
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
