#!/usr/bin/env python3

# import configparser
# import tempfile
import time

import numpy as np
import pandas as pd
import psycopg2
import psycopg2.extras
from astropy import units as u
from astropy.table import Table
from logzero import logger
from targetdb import targetdb


def connect_subaru_gaiadb(conf=None):
    conn = psycopg2.connect(**dict(conf["gaiadb"]))
    return conn


def connect_targetdb(conf=None):
    db = targetdb.TargetDB(**dict(conf["targetdb"]["db"]))
    db.connect()
    return db


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

    query_string = f"""SELECT *
    FROM {tablename}
    WHERE q3c_radial_query(ra, dec, {ra}, {dec}, {search_radius})
    """
    if mag_filter is not None:
        extra_where += f"""
        AND psf_mag_{mag_filter} BETWEEN {mag_min} AND {mag_max}
        """

    if extra_where is not None:
        query_string += extra_where

    if input_catalog is not None:
        query_string += (
            " AND ("
            + "OR".join([f" input_catalog_id={v} " for v in input_catalog])
            + ")"
        )

    if proposal_id is not None:
        query_string += (
            " AND (" + "OR".join([f" proposal_id='{v}' " for v in proposal_id]) + ")"
        )

    if max_priority is not None:
        query_string += f" AND priority <= {max_priority}"

    query_string += ";"

    logger.info(f"Query string for targets:\n{query_string}")

    df = pd.DataFrame()

    t_begin = time.time()
    df = db.fetch_query(query_string)
    t_end = time.time()
    logger.info(f"Time spent for querying (s): {t_end - t_begin:.3f}")

    df.loc[df["pmra"].isna(), "pmra"] = 0.0
    df.loc[df["pmdec"].isna(), "pmdec"] = 0.0
    df.loc[df["parallax"].isna(), "parallax"] = 1.0e-7
    logger.info(f"Fetched target DataFrame: \n{df}")

    if force_priority is not None:
        df["priority"] = force_priority

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
        flux_max = (mag_max * u.ABmag).to(u.nJy).value
        flux_min = (mag_min * u.ABmag).to(u.nJy).value

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
        query_string += "AND astrometric_excess_noise_sig < 2.0"

    query_string += ";"

    logger.info(query_string)

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

    df["g_flux_njy"] = tb["g_mag_ab"].to("nJy").value
    df["bp_r_flux_njy"] = tb["bp_mag_ab"].to("nJy").value
    df["rp_i_flux_njy"] = tb["rp_mag_ab"].to("nJy").value
    df["g_flux_err_njy"] = df["g_flux_njy"] / df["phot_g_mean_flux_over_error"]
    df["bp_r_flux_err_njy"] = df["bp_r_flux_njy"] / df["phot_bp_mean_flux_over_error"]
    df["rp_i_flux_err_njy"] = df["rp_i_flux_njy"] / df["phot_rp_mean_flux_over_error"]

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
    band_select="psf_flux_g",
    mag_min=0.0,
    mag_max=99.0,
    write_csv=False,
):
    db = connect_targetdb(conf)

    if search_radius is None:
        search_radius = fp_radius_degree * fp_fudge_factor

    flux_max = (mag_min * u.ABmag).to(u.nJy).value
    flux_min = (mag_max * u.ABmag).to(u.nJy).value

    query_string = f"""SELECT
    ob_code,obj_id,epoch,ra,dec,pmra,pmdec,parallax,
    psf_flux_g,psf_flux_r,psf_flux_i,
    psf_flux_error_g, psf_flux_error_r, psf_flux_error_i, 
    proposal.proposal_id, c.input_catalog_id, is_medium_resolution
    FROM target JOIN proposal ON target.proposal_id=proposal.proposal_id JOIN input_catalog AS c ON target.input_catalog_id = c.input_catalog_id
    WHERE q3c_radial_query(ra, dec, {ra}, {dec}, {search_radius})
    AND proposal.grade IN ('C','F')
    AND c.active
    AND {band_select} BETWEEN {flux_min} AND {flux_max}
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
            "psf_flux_error_g",
            "psf_flux_error_r",
            "psf_flux_error_i",
            "proposal_id",
            "input_catalog_id",
            "is_medium_resolution",
        ],
    )

    db.close()

    # logger.info(df_res)
    if write_csv:
        df_res.to_csv("userfiller.csv")

    return df_res


def fixcols_filler_targetdb(
    df,
    target_type_id=None,
    exptime=900.0,
    priority=1,
):
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

    if df["epoch"].dtype != "O":
        df["epoch"] = df["epoch"].apply(lambda x: f"J{x:.1f}")

    df["target_type_id"] = target_type_id

    df["effective_exptime"] = exptime
    df["priority"] = priority

    return df
