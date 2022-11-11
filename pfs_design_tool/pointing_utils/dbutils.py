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
from astropy.time import Time
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
    arms="br",
    tablename="target",
    fp_radius_degree=260.0 * 10.2 / 3600,  # "Radius" of PFS FoV in degree (?)
    fp_fudge_factor=1.5,  # fudge factor for search widths
    extra_where=None,
    force_priority=None,
):

    db = connect_targetdb(conf)

    search_radius = fp_radius_degree * fp_fudge_factor

    if "m" in arms:
        extra_where = "AND is_medium_resolution IS TRUE"
    else:
        extra_where = "AND is_medium_resolution IS FALSE"

    query_string = f"""SELECT *
    FROM {tablename}
    WHERE q3c_radial_query(ra, dec, {ra}, {dec}, {search_radius})
    """

    if extra_where is not None:
        query_string += extra_where

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
    mag_filter=None,
    min_prob_f_star=None,
    extra_where=None,
):

    db = connect_targetdb(conf)

    search_radius = fp_radius_degree * fp_fudge_factor

    query_string = f"""SELECT *
    FROM {tablename}
    WHERE q3c_radial_query(ra, dec, {ra}, {dec}, {search_radius})
    """

    if extra_where is None:
        extra_where = ""

    if good_fluxstd:
        extra_where += f"""
        AND flags_dist IS FALSE
        AND flags_ebv IS FALSE
        AND prob_f_star > 0.5
        AND psf_mag_{mag_filter} BETWEEN {mag_min} AND {mag_max}
        """

    if not good_fluxstd:
        extra_where = f"""
        AND psf_mag_{mag_filter} BETWEEN {mag_min} AND {mag_max}
        AND prob_f_star > {min_prob_f_star}
        """
        if flags_dist:
            extra_where += f"""
            AND flags_dist IS FALSE
            """
        if flags_ebv:
            extra_where += f"""
            AND flags_ebv IS FALSE
            """

    try:
        fluxstd_versions = conf["targetdb"]["fluxstd"]["version"]
    except:
        fluxstd_versions = None

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
    t_end = time.time()
    logger.info(f"Time spent for querying (s): {t_end - t_begin:.3f}")

    df.loc[df["pmra"].isna(), "pmra"] = 0.0
    df.loc[df["pmdec"].isna(), "pmdec"] = 0.0
    df.loc[df["parallax"].isna(), "parallax"] = 1.0e-7
    logger.info(f"Fetched target DataFrame: \n{df}")

    db.close()

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
    except:
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

    print(df)
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

    # Query for raster scan stars:
    # astrometric_excess_noise_sig (D) < 2
    # 12 <= phot_g_mean_mag <=20

    query_string = f"""SELECT
    source_id,ref_epoch,ra,dec,pmra,pmdec,parallax,
    phot_g_mean_mag,phot_bp_mean_mag,phot_rp_mean_mag
    FROM gaia3
    WHERE q3c_radial_query(ra, dec, {ra}, {dec}, {search_radius})
    AND {band_select} BETWEEN {mag_min} AND {mag_max}
    """

    if good_astrometry:
        query_string += "AND astrometric_excess_noise_sig < 2.0"

    query_string += ";"

    print(query_string)

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
        ],
    )

    cur.close()
    conn.close()

    print(df_res)
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

    df["epoch"] = df["epoch"].apply(lambda x: f"J{x:.1f}")
    df["proposal_id"] = proposal_id
    df["target_type_id"] = target_type_id
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
    df["bp_flux_njy"] = tb["bp_mag_ab"].to("nJy").value
    df["rp_flux_njy"] = tb["rp_mag_ab"].to("nJy").value

    # df["priority"] = np.array(tb["g_mag_ab"].value, dtype=int)
    # df["priority"][tb["g_mag_ab"].value > 12] = 9999
    df["priority"] = np.array(tb["g_mag_ab"].value - 7, dtype=int)
    df["priority"][tb["g_mag_ab"].value - 7 > 12] = 9999

    return df
