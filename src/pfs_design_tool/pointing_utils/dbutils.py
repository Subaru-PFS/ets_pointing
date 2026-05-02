#!/usr/bin/env python3

import time
import os

import numpy as np
import pandas as pd
import psycopg2
import psycopg2.extras
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.table import Table
from logzero import logger
from targetdb import targetdb
from glob import glob
from astropy.io import fits

def connect_subaru_gaiadb(conf=None):
    conn = psycopg2.connect(**dict(conf["gaiadb"]))
    return conn


def connect_targetdb(conf=None):
    db = targetdb.TargetDB(**dict(conf["targetdb"]["db"]))
    db.connect()
    return db


def connect_qadb(conf=None):
    conn = psycopg2.connect(**dict(conf["qadb"]))
    return conn


def generate_targets_from_targetdb(
    ra,
    dec,
    conf=None,
    arms="brn",
    tablename="target",
    fp_radius_degree=260.0 * 10.2 / 3600,  # "Radius" of PFS FoV in degree (?)
    fp_fudge_factor=1.5,  # fudge factor for search widths
    extra_where=None,
    force_priority=None,
    input_catalog=None,
    proposal_id=None,
    mag_min=None,
    mag_max=None,
    mag_filter=None,
    max_priority=None,
):
    db = connect_targetdb(conf)

    search_radius = fp_radius_degree * fp_fudge_factor

    if extra_where is None:
        extra_where = ""

    if "m" in arms:
        if "r" not in arms:
            extra_where = "AND is_medium_resolution IS TRUE"
    else:
        if "r" in arms:
            extra_where = "AND is_medium_resolution IS FALSE"

    query_string = f"""SELECT ob_code,obj_id,c.input_catalog_id,ra,dec,epoch,priority,pmra,pmdec,parallax,effective_exptime,single_exptime,qa_reference_arm,is_medium_resolution,proposal.proposal_id,rank,grade,allocated_time_lr+allocated_time_mr as \"allocated_time\",allocated_time_lr,allocated_time_mr,filter_g,filter_r,filter_i,filter_z,filter_y,psf_flux_g,psf_flux_r,psf_flux_i,psf_flux_z,psf_flux_y,psf_flux_error_g,psf_flux_error_r,psf_flux_error_i,psf_flux_error_z,psf_flux_error_y,total_flux_g,total_flux_r,total_flux_i,total_flux_z,total_flux_y,total_flux_error_g,total_flux_error_r,total_flux_error_i,total_flux_error_z,total_flux_error_y,target_type_id
    FROM {tablename} JOIN input_catalog AS c ON {tablename}.input_catalog_id = c.input_catalog_id JOIN proposal ON {tablename}.proposal_id=proposal.proposal_id
    WHERE q3c_radial_query(ra, dec, {ra}, {dec}, {search_radius})
    AND c.active
    """
    if extra_where is not None:
        query_string += extra_where

    if input_catalog is not None:
        query_string += (
            " AND ("
            + "OR".join([f" {tablename}.input_catalog_id={v} " for v in input_catalog])
            + ")"
        )

    if proposal_id is not None:
        query_string += (
            " AND (" + "OR".join([f" {tablename}.proposal_id='{v}' " for v in proposal_id]) + ")"
        )

    if max_priority is not None:
        query_string += f" AND priority <= {max_priority}"

    query_string += ";"

    #logger.info(f"Query string for targets:\n{query_string}")

    df = pd.DataFrame()

    t_begin = time.time()
    df = db.fetch_query(query_string)
    t_end = time.time()
    #logger.info(f"Time spent for querying (s): {t_end - t_begin:.3f}")

    # keep grade BCF in S25B and FG in S25A
    mask_keep = (
        ((df["proposal_id"].str.startswith("S25B")) & (df["grade"].isin(["B", "C", "F"])))
        | ((df["proposal_id"].str.startswith("S25A")) & (df["grade"].isin(["F", "G"])))
        | (df['proposal_id'].str.contains('EN16'))
    )
    
    df = df.loc[mask_keep].reset_index(drop=True)

    df.loc[df["pmra"].isna(), "pmra"] = 0.0
    df.loc[df["pmdec"].isna(), "pmdec"] = 0.0
    df.loc[df["parallax"].isna(), "parallax"] = 1.0e-7
    df.loc[df["rank"]<0, "rank"] = 10.0 # give highest rank to classic targets
    #logger.info(f"Fetched target DataFrame: \n{df}")

    if force_priority is not None:
        df["priority"] = force_priority

    # convert mag limits to flux (nJy)
    flux_max = (mag_min * u.ABmag).to(u.nJy).value
    flux_min = (mag_max * u.ABmag).to(u.nJy).value
    flux_limit_17mag = (17.0 * u.ABmag).to(u.nJy).value
    
    # --- build mask ---
    # case 1: grade == "G"  → flux in desired range
    mask_g = (df["grade"] == "G") & df[mag_filter].between(flux_min, flux_max)
    
    # case 2: grade != "G" → none of the bands brighter than 17 mag
    flux_cols = ["total_flux_g", "total_flux_r", "total_flux_i", "total_flux_z", "total_flux_y"]
    
    # --- case 2: grade != "G" ---
    # we build a per-row mask that depends on proposal_id
    mask_not_g = np.zeros(len(df), dtype=bool)
    
    for i, (_, row) in enumerate(df.iterrows()):
        if row["proposal_id"] == "S25A-119QF":
            # interpret as magnitudes → keep if all bands ≥ 17.0
            if not np.any([row[col] < 17.0 for col in flux_cols]):
                mask_not_g[i] = True
        elif row["grade"] in ["B", "C", "F"]:
            # interpret as fluxes → keep if all bands ≤ flux_limit_17mag
            if not np.any([
                (row[col] is not None) and np.isfinite(row[col]) and (row[col] > flux_limit_17mag)
                for col in flux_cols
            ]):
                mask_not_g[i] = True
        elif "EN16" in row["proposal_id"]:
            mask_not_g[i] = True
        else:
            continue
                
    # --- combine both ---
    df = df[mask_g | mask_not_g].reset_index(drop=True)

    db.close()

    return df


def generate_fluxstds_from_targetdb(
    ra,
    dec,
    conf=None,
    tablename="fluxstd",
    fp_radius_degree=260.0 * 10.2 / 3600,  # "Radius" of PFS FoV in degree (?)
    fp_fudge_factor=1.5,  # fudge factor for search widths
    good_fluxstd=False,
    flags_dist=False,
    flags_ebv=False,
    mag_min=None,
    mag_max=None,
    flux_min=None,
    flux_max=None,
    select_by_flux=None,
    mag_filter=None,
    min_prob_f_star=None,
    min_teff=None,
    max_teff=None,
    extra_where=None,
    write_csv=False,
    ignore_prob_f_star=False,
    select_from_gaia=False,
):
    try:
        fluxstd_versions = conf["targetdb"]["fluxstd"]["version"]
    except Exception:
        fluxstd_versions = None

    db = connect_targetdb(conf)

    search_radius = fp_radius_degree * fp_fudge_factor

    query_string = f"""SELECT *
    FROM {tablename}
    WHERE q3c_radial_query(ra, dec, {ra}, {dec}, {search_radius})
    """

    if extra_where is None:
        extra_where = ""

    if not ignore_prob_f_star:
        extra_where = f"""
        AND prob_f_star BETWEEN {min_prob_f_star} AND 1.0
        """

    if good_fluxstd:
        extra_where += """
        AND flags_dist IS FALSE
        AND flags_ebv IS FALSE
        """
        if select_by_flux:
            extra_where += (
                f"""AND psf_flux_{mag_filter} BETWEEN {flux_min} AND {flux_max}"""
            )
        else:
            extra_where += (
                f"""AND psf_mag_{mag_filter} BETWEEN {mag_min} AND {mag_max}"""
            )

    if not good_fluxstd:
        if select_by_flux:
            extra_where += (
                f"""AND psf_flux_{mag_filter} BETWEEN {flux_min} AND {flux_max}"""
            )
        else:
            extra_where += (
                f"""AND psf_mag_{mag_filter} BETWEEN {mag_min} AND {mag_max}"""
            )

        if flags_dist:
            extra_where += """
            AND flags_dist IS FALSE
            """
        if flags_ebv:
            extra_where += """
            AND flags_ebv IS FALSE
            """
    if fluxstd_versions is not None:
        for fluxstd_version in fluxstd_versions:
            try:
                if float(fluxstd_version) >= 3.0:
                    if not select_from_gaia:
                        extra_where += f"""
                AND teff_brutus BETWEEN {min_teff} AND {max_teff}
                """
                        break
                    else:
                        extra_where += f"""
                AND teff_gspphot BETWEEN {min_teff} AND {max_teff}
                """

            except:
                extra_where += f""

    if fluxstd_versions is not None:
        version_condition = "("
        first_condition = True
        for fluxstd_version in fluxstd_versions:
            if first_condition:
                first_condition = False
            else:
                version_condition += " OR "
            version_condition += f"version = '{fluxstd_version}'"
        version_condition += ")"

        extra_where += f"""
        AND {version_condition
        }"""

    query_string += extra_where

    query_string += ";"

    logger.info(f"Query string for fluxstd: \n{query_string}")

    t_begin = time.time()
    df = db.fetch_query(query_string)

    if len(df) == 0:
        # select gaia fstar when no PS1 fstar is selected
        flux_max = (mag_min * u.ABmag).to(u.nJy).value
        flux_min = (mag_max * u.ABmag).to(u.nJy).value

        query_string = f"""SELECT *
            FROM {tablename}
            WHERE q3c_radial_query(ra, dec, {ra}, {dec}, {search_radius})
            AND is_fstar_gaia
            AND teff_gspphot BETWEEN {min_teff} AND {max_teff}
            AND psf_flux_r BETWEEN {flux_min} AND {flux_max};
            """
        logger.info(f"Query string for fluxstd (Gaia): \n{query_string}")

        df = db.fetch_query(query_string)

    t_end = time.time()
    logger.info(f"Time spent for querying (s): {t_end - t_begin:.3f}")

    df.loc[df["pmra"].isna(), "pmra"] = 0.0
    df.loc[df["pmdec"].isna(), "pmdec"] = 0.0
    df.loc[df["parallax"].isna(), "parallax"] = 1.0e-7
    logger.info(f"Fetched target DataFrame: \n{df}")

    db.close()

    n_fluxstd_ori = len(df)

    # check if there is cluster
    ra = df["ra"].values
    dec = df["dec"].values

    bins = 7  # adjust for resolution (7×7 works well)
    H, ra_edges, dec_edges = np.histogram2d(ra, dec, bins=bins)
    d_ra = ra_edges[1] - ra_edges[0]
    d_dec = dec_edges[1] - dec_edges[0]
    local_density = H / (d_ra * d_dec)

    density_global = np.mean(local_density)
    print(f"Average density ≈ {density_global:.2f} / deg²")

    threshold = 5.0 * density_global  # keep regions up to 5.0 × mean
    overdense = local_density > threshold

    valid_densities = local_density[~overdense]
    density_global_clean = np.mean(valid_densities[valid_densities > 0])

    if np.any(overdense):
        logger.warning("There may be clusters of fluxstds")

        keep_idx = []
    
        for i in range(bins):
            for j in range(bins):
                # points inside bin
                in_bin = (
                    (ra >= ra_edges[i]) & (ra < ra_edges[i+1]) &
                    (dec >= dec_edges[j]) & (dec < dec_edges[j+1])
                )
                idx = np.where(in_bin)[0]
                if len(idx) == 0:
                    continue
        
                # expected count per bin given global density
                expected = density_global_clean * d_ra * d_dec
                n_expected = int(round(expected))
        
                if len(idx) > n_expected:
                    keep_idx.extend(np.random.choice(idx, n_expected, replace=False))
                else:
                    keep_idx.extend(idx)
        
        df = df.iloc[keep_idx].reset_index(drop=True)
        print(f"Kept {len(df)} / {n_fluxstd_ori} flux standards")

    if write_csv:
        df.to_csv("fluxstd.csv")

    return df


def generate_skyobjects_from_targetdb(
    ra,
    dec,
    conf=None,
    tablename="sky",
    fp_radius_degree=260.0 * 10.2 / 3600,  # "Radius" of PFS FoV in degree (?)
    fp_fudge_factor=1.5,  # fudge factor for search widths
    # extra_where=None,
):
    db = connect_targetdb(conf)

    search_radius = fp_radius_degree * fp_fudge_factor

    try:
        sky_versions = conf["targetdb"]["sky"]["version"]
    except Exception:
        sky_versions = None

    where_condition = f"WHERE q3c_radial_query(ra, dec, {ra}, {dec}, {search_radius})"

    if sky_versions is not None:
        version_condition = "("
        first_condition = True
        for sky_version in sky_versions:
            if first_condition:
                first_condition = False
            else:
                version_condition += " OR "
            if sky_version == "20220915":
                # use only HSC sky catalog in the older version
                version_condition += (
                    f"(version = '{sky_version}' AND input_catalog_id=1001)"
                )
            else:
                version_condition += f"version = '{sky_version}'"
        version_condition += ")"

        where_condition += f" AND {version_condition}"

    query_string = f"""SELECT *
    FROM {tablename}
    {where_condition}
    """

    query_string += ";"

    logger.info(f"Query string for sky: \n{query_string}")

    t_begin = time.time()
    df = db.fetch_query(query_string)
    t_end = time.time()
    logger.info(f"Time spent for querying (s): {t_end - t_begin:.3f}")

    df["pmra"] = np.zeros(df.index.size, dtype=float)
    df["pmdec"] = np.zeros(df.index.size, dtype=float)
    df["parallax"] = np.full(df.index.size, 1.0e-7)
    logger.info(f"Fetched target DataFrame: \n{df}")

    # Replacing obj_id with sky_id as currently (obj_id, cat_id) pairs can be duplicated for sky.
    # In the version 20220915, obj_ids are not unique and sometimes not integer.

    is_old_version = df["version"] == "20220915"

    if np.any(is_old_version):
        df.loc[is_old_version, "obj_id"] = df.loc[is_old_version, "sky_id"]
        logger.warning(
            "obj_id is forced to be replaced to sky_id for sky objects with version=20220915"
        )

    db.close()

    return df


def generate_random_skyobjects(
    ra,
    dec,
    n_sky_target,
):
    dw = 0.75
    cos_term = 1.0 / np.cos(dec * u.deg)
    dw_ra = dw * cos_term
    df = pd.DataFrame()
    df["sky_id"] = np.arange(n_sky_target)
    df["obj_id"] = np.arange(n_sky_target)
    df["ra"] = np.random.uniform(ra - dw_ra, ra + dw_ra, n_sky_target)
    df["dec"] = np.random.uniform(dec - dw, dec + dw, n_sky_target)
    df["epoch"] = "J2016.0"
    df["tract"] = None
    df["patch"] = None
    df["target_type_id"] = None
    df["input_catalog_id"] = None
    df["mag_thresh"] = None
    df["version"] = None
    df["created_at"] = None
    df["updated_at"] = None

    # print(df)
    return df


def generate_targets_from_gaiadb(
    ra,
    dec,
    conf=None,
    fp_radius_degree=260.0 * 10.2 / 3600,  # "Radius" of PFS FoV in degree (?)
    fp_fudge_factor=1.5,  # fudge factor for search widths
    search_radius=None,
    band_select="phot_g_mean_mag",
    mag_min=0.0,
    mag_max=99.0,
    good_astrometry=False,
    write_csv=False,
):
    conn = connect_subaru_gaiadb(conf)
    cur = conn.cursor()

    if search_radius is None:
        search_radius = fp_radius_degree * fp_fudge_factor

    # Query for fillers:
    # astrometric_excess_noise_sig (D) < 2
    # 12 <= phot_g_mean_mag <=20

    query_string = f"""SELECT
    source_id,ref_epoch,ra,dec,pmra,pmdec,parallax,
    phot_g_mean_mag,phot_bp_mean_mag,phot_rp_mean_mag,
    phot_g_mean_flux_over_error, phot_bp_mean_flux_over_error, phot_rp_mean_flux_over_error
    FROM gaia3
    WHERE q3c_radial_query(ra, dec, {ra}, {dec}, {search_radius})
    AND {band_select} BETWEEN {mag_min} AND {mag_max}
    """

    if good_astrometry:
        query_string += "AND astrometric_excess_noise < 1.0"

    query_string += ";"

    #logger.info(query_string)

    cur.execute(query_string)

    df_res = pd.DataFrame(
        cur.fetchall(),
        columns=[
            "source_id",
            "ref_epoch",
            "ra",
            "dec",
            "pmra",
            "pmdec",
            "parallax",
            "phot_g_mean_mag",
            "phot_bp_mean_mag",
            "phot_rp_mean_mag",
            "phot_g_mean_flux_over_error",
            "phot_bp_mean_flux_over_error",
            "phot_rp_mean_flux_over_error",
        ],
    )

    cur.close()
    conn.close()

    # logger.info(df_res)
    if write_csv:
        df_res.to_csv("gaia.csv")

    return df_res


def fixcols_gaiadb_to_targetdb(
    df,
    proposal_id=None,
    target_type_id=None,
    input_catalog_id=None,
    exptime=900.0,
    priority=1,
):
    df.rename(columns={"source_id": "obj_id", "ref_epoch": "epoch"}, inplace=True)

    if df["epoch"].dtype != "O":
        df["epoch"] = df["epoch"].apply(lambda x: f"J{x:.1f}")

    if "proposal_id" not in df.columns:
        df["proposal_id"] = proposal_id

    if "is_medium_resolution" not in df.columns:
        df["is_medium_resolution"] = "L/M"

    df["ob_code"] = df["obj_id"].astype("str")
    df["target_type_id"] = target_type_id

    if "input_catalog_id" not in df.columns:
        df["input_catalog_id"] = input_catalog_id

    df["effective_exptime"] = exptime
    df["priority"] = priority

    tb = Table([])

    # ZPs are taken from Weiler (2018, A&A, 617, A138)
    tb["g_mag_ab"] = (df["phot_g_mean_mag"].to_numpy() + (25.7455 - 25.6409)) * u.ABmag
    tb["bp_mag_ab"] = (
        df["phot_bp_mean_mag"].to_numpy() + (25.3603 - 25.3423)
    ) * u.ABmag
    tb["rp_mag_ab"] = (
        df["phot_rp_mean_mag"].to_numpy() + (25.1185 - 24.7600)
    ) * u.ABmag

    ## FIXME? ##
    df["psf_flux_g"] = tb["g_mag_ab"].to("nJy").value
    df["psf_flux_r"] = tb["bp_mag_ab"].to("nJy").value
    df["psf_flux_i"] = tb["rp_mag_ab"].to("nJy").value
    df["psf_flux_z"] = np.full(len(tb), np.nan)
    df["psf_flux_y"] = np.full(len(tb), np.nan)
    df["psf_flux_error_g"] = df["psf_flux_g"] / df["phot_g_mean_flux_over_error"]
    df["psf_flux_error_r"] = df["psf_flux_r"] / df["phot_bp_mean_flux_over_error"]
    df["psf_flux_error_i"] = df["psf_flux_i"] / df["phot_rp_mean_flux_over_error"]
    df["psf_flux_error_z"] = np.full(len(tb), np.nan)
    df["psf_flux_error_y"] = np.full(len(tb), np.nan)
    df["filter_g"] = ["g_gaia" for _ in range(len(tb))]
    df["filter_r"] = ["bp_gaia" for _ in range(len(tb))]
    df["filter_i"] = ["rp_gaia" for _ in range(len(tb))]
    df["filter_z"] = ["none" for _ in range(len(tb))]
    df["filter_y"] = ["none" for _ in range(len(tb))]
    # df["priority"] = np.array(tb["g_mag_ab"].value, dtype=int)
    # df["priority"][tb["g_mag_ab"].value > 12] = 9999
    # df["priority"] = np.array(tb["g_mag_ab"].value - 7, dtype=int)
    # df["priority"][tb["g_mag_ab"].value - 7 > 12] = 9999
    # df["priority"][np.isnan(tb["g_mag_ab"])] = 9

    return df


def generate_fillers_from_targetdb(
    ra,
    dec,
    conf=None,
    fp_radius_degree=260.0 * 10.2 / 3600,  # "Radius" of PFS FoV in degree (?)
    fp_fudge_factor=1.5,  # fudge factor for search widths
    search_radius=None,
    band_select="total_flux_g",
    mag_min=0.0,
    mag_max=99.0,
    write_csv=False,
):
    db = connect_targetdb(conf)

    if search_radius is None:
        search_radius = fp_radius_degree * fp_fudge_factor

    query_string = f"""SELECT
    ob_code,obj_id,epoch,ra,dec,pmra,pmdec,parallax,
    psf_flux_g,psf_flux_r,psf_flux_i,psf_flux_z,psf_flux_y,
    psf_flux_error_g, psf_flux_error_r, psf_flux_error_i, psf_flux_error_z, psf_flux_error_y, 
    total_flux_g,total_flux_r,total_flux_i,total_flux_z,total_flux_y,
    total_flux_error_g, total_flux_error_r, total_flux_error_i, total_flux_error_z, total_flux_error_y, 
    filter_g, filter_r, filter_i, filter_z, filter_y,
    proposal.proposal_id, proposal.grade, c.input_catalog_id, is_medium_resolution
    FROM target JOIN proposal ON target.proposal_id=proposal.proposal_id JOIN input_catalog AS c ON target.input_catalog_id = c.input_catalog_id
    WHERE q3c_radial_query(ra, dec, {ra}, {dec}, {search_radius})
    AND c.active
    """

    query_string += ";"

    logger.info(query_string)

    df_res = pd.DataFrame(
        db.fetch_query(query_string),
        columns=[
            "ob_code",
            "obj_id",
            "epoch",
            "ra",
            "dec",
            "pmra",
            "pmdec",
            "parallax",
            "psf_flux_g",
            "psf_flux_r",
            "psf_flux_i",
            "psf_flux_z",
            "psf_flux_y",
            "psf_flux_error_g",
            "psf_flux_error_r",
            "psf_flux_error_i",
            "psf_flux_error_z",
            "psf_flux_error_y",
            "total_flux_g",
            "total_flux_r",
            "total_flux_i",
            "total_flux_z",
            "total_flux_y",
            "total_flux_error_g",
            "total_flux_error_r",
            "total_flux_error_i",
            "total_flux_error_z",
            "total_flux_error_y",
            "filter_g",
            "filter_r",
            "filter_i",
            "filter_z",
            "filter_y",
            "proposal_id",
            "grade",
            "input_catalog_id",
            "is_medium_resolution",
        ],
    )

    #df_res = df_res[df_res["grade"] != "G"] # do not include obs. fillers

    # convert mag limits to flux (nJy)
    flux_max = (mag_min * u.ABmag).to(u.nJy).value
    flux_min = (mag_max * u.ABmag).to(u.nJy).value
    flux_limit_17mag = (17.0 * u.ABmag).to(u.nJy).value
    
    # --- build mask ---
    # case 1: grade == "G"  → flux in desired range
    mask_g = (df_res["grade"] == "G") & df_res[band_select].between(flux_min, flux_max)
    
    # case 2: grade != "G" → none of the bands brighter than 17 mag
    flux_cols = ["total_flux_g", "total_flux_r", "total_flux_i", "total_flux_z", "total_flux_y"]
    
    # --- case 2: grade != "G" ---
    # we build a per-row mask that depends on proposal_id
    mask_not_g = np.zeros(len(df_res), dtype=bool)
    
    for i, (_, row) in enumerate(df_res.iterrows()):
        if row["proposal_id"] == "S25A-119QF":
            # interpret as magnitudes → keep if all bands ≥ 17.0
            if not np.any([row[col] < 17.0 for col in flux_cols]):
                mask_not_g[i] = True
        else:
            # interpret as fluxes → keep if all bands ≤ flux_limit_17mag
            if not np.any([
                (row[col] is not None) and np.isfinite(row[col]) and (row[col] > flux_limit_17mag)
                for col in flux_cols
            ]):
                mask_not_g[i] = True
                
    # --- combine both ---
    df_res_magcut = df_res[mask_g | mask_not_g].reset_index(drop=True)

    db.close()

    # logger.info(df_res)
    if write_csv:
        df_res.to_csv("userfiller.csv")

    return df_res_magcut, df_res


def fixcols_filler_targetdb(
    df,
    df_no_mag_cut,
    conf=None,
    target_type_id=None,
    exptime=900.0,
    priority_obs=1,
    priority_usr=1,
    priority_obs_done=1,
    priority_usr_done=1,
    dup_obs_filler_remove=False,
    obs_filler_done_remove=False,
    workDir=None,
):
    """
    # only for gaia
    df.rename(
        columns={
            "psf_flux_g": "g_flux_njy",
            "psf_flux_r": "bp_r_flux_njy",
            "psf_flux_i": "rp_i_flux_njy",
            "psf_flux_error_g": "g_flux_err_njy",
            "psf_flux_error_r": "bp_r_flux_err_njy",
            "psf_flux_error_i": "rp_i_flux_err_njy",
        },
        inplace=True,
    )
    #"""

    if df["epoch"].dtype != "O":
        df["epoch"] = df["epoch"].apply(lambda x: f"J{x:.1f}")

    df["target_type_id"] = target_type_id

    df["effective_exptime"] = exptime

    df_filler_obs = df[df["grade"].isin(["G"])]
    df_filler_usr = df[
        ((df["grade"] == "C") & df["proposal_id"].str.startswith("S25B"))
        | ((df["grade"] == "F") & df["proposal_id"].str.startswith("S25A"))
    ]
    df_sci = df_no_mag_cut[df_no_mag_cut["grade"].isin(["B", "C", "F"])]

    df_filler_obs["observed"] = False  # ensure column exists
    df_filler_usr["observed"] = False  # ensure column exists

    if dup_obs_filler_remove:
        n_obs_filler_orig = len(df_filler_obs)
        # Build SkyCoord for df_filler_obs
        coords_obs = SkyCoord(
            ra=df_filler_obs["ra"].values * u.deg,
            dec=df_filler_obs["dec"].values * u.deg,
        )

        # Build SkyCoord for df_filler_usr (user-filler) + df_sci (science)
        coords_sci = SkyCoord(
            ra=df_sci["ra"].values * u.deg, dec=df_sci["dec"].values * u.deg
        )

        # Match df_filler_obs → df_sci
        idx_sci, sep2d_sci, _ = coords_obs.match_to_catalog_sky(coords_sci)
        mask_sci = sep2d_sci < (1.0 * u.arcsec)

        # Keep only those not duplicated in either catalog
        mask_keep = ~mask_sci
        df_filler_obs = df_filler_obs.loc[mask_keep].reset_index(drop=True)
        n_obs_filler_red = len(df_filler_obs)
        logger.info(
            f"Duplicates in obs. filler removed: {n_obs_filler_orig} --> {n_obs_filler_red}"
        )

    if obs_filler_done_remove:
        # check observed obs filler
        try:
            df_obs_filler_done = pd.read_csv(os.path.join(workDir, "ppp/df_obsfiller_done.csv"))
        except:
            # query qaDB to get executed pfsdesign
            conn = connect_qadb(conf)
            cur = conn.cursor()
        
            sql = f'''
            SELECT pfs_design_id 
            FROM exposure_time 
                JOIN pfs_visit ON exposure_time.pfs_visit_id = pfs_visit.pfs_visit_id 
                JOIN onsite_processing_status ON onsite_processing_status.pfs_visit_id = pfs_visit.pfs_visit_id
            WHERE pfs_visit.pfs_visit_id >=129587
            ORDER BY pfs_visit.pfs_visit_id DESC;
            '''
        
            cur.execute(sql)
        
            df_design_done = pd.DataFrame(
                cur.fetchall(),
                columns=["pfs_design_id"],
            )
        
            cur.close()
            conn.close()

            # search for design files under /work/wanqqq/ and make a df of observed obs filler
            df_list_obs_filler = []
            cols = ["ra", "dec", "catId", "objId", "targetType", "proposalId", "obcode"]

            for design_id in set(df_design_done["pfs_design_id"]):
                # construct expected filename
                fname = f"pfsDesign-0x{design_id:016x}.fits"
            
                base_dir = "/work/wanqqq/"
                
                for root, dirs, files in os.walk(base_dir):
                    if fname in files:
                        filepath = os.path.join(root, fname)
            
                        with fits.open(filepath) as hdul:
                            data = hdul[1].data
                            mask_obs_filler = (data["targetType"] == 1) & (np.in1d(data["proposalId"], conf["sfa"]["proposalIds_obsFiller"]))
                            if sum(mask_obs_filler) > 0:
                                df_obs_filler_ = pd.DataFrame({col: data[mask_obs_filler][col] for col in cols})
                                df_list_obs_filler.append(df_obs_filler_)
    
            df_obs_filler_done = pd.concat(df_list_obs_filler, ignore_index=True).drop_duplicates(subset=["objId", "obcode"])
            df_obs_filler_done.to_csv(os.path.join(workDir, "ppp/df_obsfiller_done.csv"))

        # match observed obs filler with df_filler_obs, and set the observed ones to be true
        mask = df_filler_obs.set_index(["obj_id", "ob_code"]).index.isin(
            df_obs_filler_done.set_index(["objId", "obcode"]).index
        )
        df_filler_obs.loc[mask, "observed"] = True

        logger.info(f"There are {sum(df_filler_obs['observed'])} / {len(df_filler_obs)} observed")

        # check observed user filler (including grade C)
        base_dir = os.path.join(workDir, "ppp")
        file_path = glob(os.path.join(base_dir, "tgt_queueDB*.csv"))

        if file_path:
            df_usr_filler_done = pd.read_csv(file_path[0])

        mask = df_filler_usr.set_index(["proposal_id", "ob_code"]).index.isin(
            df_usr_filler_done.set_index(["psl_id", "ob_code"]).index
        )
        df_filler_usr.loc[mask, "observed"] = True

        logger.info(f"There are {sum(df_filler_usr['observed'])} / {len(df_filler_usr)} observed")

    df_filler_obs["priority"] = np.where(
        df_filler_obs["observed"],
        priority_obs_done,
        priority_obs
    )
    
    df_filler_usr["priority"] = np.where(
        df_filler_usr["observed"],
        priority_usr_done,
        priority_usr
    )

    return df_filler_obs, df_filler_usr
