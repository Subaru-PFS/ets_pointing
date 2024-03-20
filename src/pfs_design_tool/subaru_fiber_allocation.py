#!/usr/bin/env python3
#
# Necessary preparations for running:
#
# This script depends on several other modules from https://github.com/Subaru-PFS
# All of them were at the HEAD of the respective master branches, with the
# exception of "ets_fiber_assigner" and "pfs_utils" (must be on branch "tickets/INSTRM-1582").
#
# Also, the environment variable PFS_INSTDATA_DIR must be set correctly.
#

import argparse
import os

import numpy as np
import pandas as pd
from pfs_design_tool.pointing_utils import dbutils, designutils, nfutils
import toml
from astropy.time import Time
from astropy.utils import iers
from logzero import logger

# The following line seems to be needed to avoid IERS errors,
# though the default config is already `auto_download=True`.
iers.conf.auto_download = True
# iers.conf.iers_degraded_accuracy = "warn"

# netflow configuration (FIXME)
cobra_location_group = None
min_sky_targets_per_location = None
location_group_penalty = None
cobra_instrument_region = None
min_sky_targets_per_instrument_region = None
instrument_region_penalty = None

def get_arguments():
    parser = argparse.ArgumentParser()

    # telescope configurations
    parser.add_argument(
        "--ra",
        type=float,
        default=0.0,
        help="Telescope center RA [degrees] (default: 0.0)",
    )
    parser.add_argument(
        "--dec",
        type=float,
        default=0.0,
        help="Telescope center Dec [degrees] (default: 0.0)",
    )
    parser.add_argument(
        "--pa",
        type=float,
        default=-90.0,
        help="Telescope position angle [degrees] (default: -90.0)",
    )
    parser.add_argument(
        "--observation_time",
        type=str,
        default="2022-05-20T15:00:00Z",
        help="planned time of observation in UTC (default: 2022-05-20T15:00:00Z)",
    )
    parser.add_argument(
        "--telescope_elevation",
        type=float,
        default=None,
        help="Telescope elevation in degree \
        (default: None to set automatically from (ra, dec, observation_time))",
    )
    parser.add_argument(
        "--arms",
        type=str,
        default="br",
        help="Spectrograph arms to expose, such as 'brn' and 'bmn' (default: 'br')",
    )
    parser.add_argument(
        "--exptime",
        type=float,
        default=None,
        help="Override the exptime (seconds) obtained from the database (default: None)",
    )
    parser.add_argument(
        "--design_name",
        type=str,
        default=None,
        help="Human-readable design name (default: None)",
    )

    # configuration file
    parser.add_argument(
        "--conf",
        type=str,
        default="config.toml",
        help="Config file for the script to run. Must be a .toml file (default: config.toml)",
    )

    # output directories
    parser.add_argument(
        "--design_dir",
        type=str,
        default=".",
        help="directory for storing pfsDesign files (default: .)",
    )
    parser.add_argument(
        "--cobra_coach_dir",
        type=str,
        default=".",
        help="path for temporary cobraCoach files (default: .)",
    )

    # guide stars
    parser.add_argument(
        "--guidestar_mag_min",
        type=float,
        default=12.0,
        help="minimum magnitude for guide star candidates (default: 12.)",
    )
    parser.add_argument(
        "--guidestar_mag_max",
        type=float,
        default=19.0,
        help="maximum magnitude for guide star candidates (default: 19.)",
    )
    parser.add_argument(
        "--guidestar_neighbor_mag_min",
        type=float,
        default=21.0,
        help="minimum magnitude for objects in the vicinity of guide star candidates (default: 21.)",
    )
    parser.add_argument(
        "--guidestar_minsep_deg",
        type=float,
        default=1.0 / 3600,
        help="radius of guide star candidate vicinity (default: 1/3600)",
    )

    # science targets
    parser.add_argument(
        "--target_mag_max",
        type=float,
        default=19.0,
        help="Maximum (faintest) magnitude for stars in fibers (default: 19.)",
    )
    parser.add_argument(
        "--target_mag_min",
        type=float,
        default=0.0,
        help="Minimum (brightest) magnitude for stars in fibers (default: 0)",
    )
    parser.add_argument(
        "--target_mag_filter",
        type=str,
        default=None,
        help="Photometric band (grizyj of PS1) to apply magnitude cuts (default: None)",
    )
    parser.add_argument(
        "--target_priority_max",
        type=float,
        default=None,
        help="Maximum priority of the target (default: None)",
    )
    parser.add_argument(
        "--disable_force_priority",
        action="store_true",
        help="Disable the force_priority (default: False)",
    )
    parser.add_argument(
        "--degrade_priority",
        type=int,
        default=0,
        help="Degrade priority of all science targets (default: 0)",
    )
    parser.add_argument(
        "--degrade_priority_proposal",
        type=str,
        default=None,
        help="ProposalId for the priority degradation. If not specified, all proposals are degraded (default: None)",
    )
    parser.add_argument(
        "--skip_target",
        action="store_true",
        help="Skip science targets (default: False)",
    )

    # flux standards

    parser.add_argument(
        "--fluxstd_mag_max",
        type=float,
        default=19.0,
        help="Maximum (faintest) magnitude for stars in fibers (default: 19.)",
    )
    parser.add_argument(
        "--fluxstd_mag_min",
        type=float,
        default=14.0,
        help="Minimum (brightest) magnitude for stars in fibers (default: 14.0)",
    )
    parser.add_argument(
        "--fluxstd_mag_filter",
        type=str,
        default="g",
        help="Photometric band (grizyj of PS1) to apply magnitude cuts (default: g)",
    )
    parser.add_argument(
        "--fluxstd_flux_max",
        type=float,
        default=9120000.0,
        help="Maximum (brightest) flux for stars in fibers in [nJy] (default: 9120000.)",
    )
    parser.add_argument(
        "--fluxstd_flux_min",
        type=float,
        default=91200.0,
        help="Minimum (faintest) flux for stars in fibers in [nJy] (default: 91200.)",
    )
    parser.add_argument(
        "--fluxstd_select_by_flux",
        action="store_true",
        help="Select fluxstd stars by the flux range (default: False)",
    )
    parser.add_argument(
        "--good_fluxstd",
        action="store_true",
        help="Select fluxstd stars with prob_f_star>0.5, \
            flags_dist=False, and flags_ebv=False (default: False)",
    )
    parser.add_argument(
        "--fluxstd_min_prob_f_star",
        type=float,
        default=0.5,
        help="Minimum acceptable prob_f_star (default: 0.5)",
    )
    parser.add_argument(
        "--fluxstd_min_teff",
        type=float,
        default=3000.0,
        help="Minimum acceptable teff_brutus in [K] (default: 3000.)",
    )
    parser.add_argument(
        "--fluxstd_max_teff",
        type=float,
        default=10000.0,
        help="Maximum acceptable teff_brutus in [K] (default: 10000.)",
    )
    parser.add_argument(
        "--fluxstd_flags_dist",
        action="store_true",
        help="Select fluxstd stars with flags_dist=False (default: False)",
    )
    parser.add_argument(
        "--fluxstd_flags_ebv",
        action="store_true",
        help="Select fluxstd stars with flags_ebv=False (default: False)",
    )
    parser.add_argument(
        "--n_fluxstd",
        type=int,
        default=50,
        help="Number of FLUXSTD stars to be allocated. (default: 50)",
    )

    # filler targets from gaiaDB
    parser.add_argument(
        "--filler",
        action="store_true",
        help="Search stars for fillers (default: False)",
    )
    parser.add_argument(
        "--filler_mag_max",
        type=float,
        default=20.0,
        help="maximum magnitude in Gaia G for filler targets (default: 20.)",
    )
    parser.add_argument(
        "--filler_mag_min",
        type=float,
        default=12.0,
        help="minimum magnitude in Gaia G for filler targets (default: 12)",
    )
    parser.add_argument(
        "--filler_propid",
        default="S23A-EN16",
        help="Proposal-ID for filler targets (default: S23A-EN16)",
    )

    # sky fibers
    parser.add_argument(
        "--n_sky",
        type=int,
        default=0,
        help="Number of SKY fibers to be allocated. (default: 0)",
    )
    parser.add_argument(
        "--sky_random",
        action="store_true",
        help="Assign sky randomly (default: False)",
    )
    parser.add_argument(
        "--reduce_sky_targets",
        action="store_true",
        help="Reduce the number of sky targets randomly (default: False)",
    )
    parser.add_argument(
        "--n_sky_random",
        type=int,
        default=30000,
        help="Number of random (or randomly reduced) SKY fibers to be allocated. (default: 30000)",
    )
    parser.add_argument(
        "--filler_random",
        action="store_true",
        help="Assign fillers randomly (default: False)",
    )
    parser.add_argument(
        "--reduce_fillers",
        action="store_true",
        help="Reduce the number of fillers randomly (default: False)",
    )
    parser.add_argument(
        "--n_fillers_random",
        type=int,
        default=30000,
        help="Number of random (or randomly reduced) fillers to be allocated. (default: 30000)",
    )

    # instrument parameter files
    parser.add_argument(
        "--pfs_instdata_dir",
        type=str,
        default="/Users/monodera/Dropbox/NAOJ/PFS/Subaru-PFS/pfs_instdata/",
        help="Location of pfs_instdata (default: /Users/monodera/Dropbox/NAOJ/PFS/Subaru-PFS/pfs_instdata/)",
    )
    parser.add_argument(
        "--cobra_coach_module_version",
        type=str,
        default=None,
        help="version of the bench description file (default: None)",
    )
    parser.add_argument(
        "--sm",
        nargs="+",
        type=int,
        default=[1, 2, 3, 4],
        help="Spectral Modules(1 to 4) to be used (default: 1 2 3 4)",
    )
    parser.add_argument(
        "--dot_margin",
        type=float,
        default=1.0,
        help="Margin factor for dot avoidance (default: 1.0)",
    )
    parser.add_argument(
        "--dot_penalty",
        type=float,
        default=None,
        help="Cost for penalty of the dot proximity (default: None)",
    )
    parser.add_argument(
        "--input_catalog",
        nargs="+",
        type=int,
        default=None,
        help="Input catalog IDs for targets (default: None)",
    )
    parser.add_argument(
        "--proposal_id",
        nargs="+",
        type=str,
        default=None,
        help="Input proposal IDs for targets (default: None)",
    )

    parser.add_argument(
        "--guide_star_id_exclude",
        nargs="+",
        type=int,
        default=None,
        help="guide star ID to be excluded (default: None)",
    )

    args = parser.parse_args()

    # NOTE: astropy.time.Time.now() uses datetime.utcnow()
    if args.observation_time.lower() == "now":
        args.observation_time = Time.now().iso
        logger.info(
            f"Observation time is set to the current UTC {args.observation_time}."
        )

    return args


def read_conf(conf):
    config = toml.load(conf)
    return config


def main():

    args = get_arguments()

    print(args)
    # exit()

    conf = read_conf(args.conf)

    print(dict(conf["gurobi"]))

    for d in [args.design_dir, args.cobra_coach_dir]:
        try:
            os.makedirs(d, exist_ok=False)
        except BaseException:
            pass

    force_priority = 1
    if args.disable_force_priority:
        force_priority = None
    df_targets = dbutils.generate_targets_from_targetdb(
        args.ra, args.dec, conf=conf, arms=args.arms,
        force_priority=force_priority,
        input_catalog=args.input_catalog,
        proposal_id=args.proposal_id,
        mag_min=args.target_mag_min,
        mag_max=args.target_mag_max,
        mag_filter=args.target_mag_filter,
        max_priority=args.target_priority_max,
    )

    ## FIXME: temporal workaround for GE targets ##
    if args.input_catalog == [10] and args.proposal_id == ['S23A-QN900', 'S23A-QN901', 'S23A-QN902', 'S23A-QN903']:
        df_targets.priority[df_targets.priority == 2] = 5
        df_targets.priority[df_targets.priority == 3] = 6
        df_targets.priority[df_targets.priority == 4] = 3
        df_targets.priority[df_targets.priority < 1] = 7

    ## FIXME: temporal workaround for S24B-QT907 targets ##
    msk = (df_targets.proposal_id == 'S24B-QT907') * (df_targets.psf_flux_g > 1150)
    df_targets = df_targets[~msk].reset_index()

    # degrade priority of science targets (default: no degradation)
    if args.degrade_priority_proposal == None:
        df_targets.priority += args.degrade_priority
    else:
        df_targets.priority[df_targets.proposal_id == args.degrade_priority_proposal] += args.degrade_priority


    if args.skip_target:
        df_targets = df_targets[:0]
    df_fluxstds = dbutils.generate_fluxstds_from_targetdb(
        args.ra,
        args.dec,
        conf=conf,
        good_fluxstd=args.good_fluxstd,
        flags_dist=args.fluxstd_flags_dist,
        flags_ebv=args.fluxstd_flags_ebv,
        mag_min=args.fluxstd_mag_min,
        mag_max=args.fluxstd_mag_max,
        flux_min=args.fluxstd_flux_min,
        flux_max=args.fluxstd_flux_max,
        select_by_flux=args.fluxstd_select_by_flux,
        mag_filter=args.fluxstd_mag_filter,
        min_prob_f_star=args.fluxstd_min_prob_f_star,
        min_teff=args.fluxstd_min_teff,
        max_teff=args.fluxstd_max_teff,
        write_csv=True,
    )

    if args.n_sky == 0:
        logger.info("No sky object will be sent to netflow")
        df_sky = pd.DataFrame()
    elif args.sky_random:
        logger.info("Random sky objects will be generated.")
        # n_sky_target = (df_targets.size + df_fluxstds.size) * 2
        n_sky_target = args.n_sky_random  # this value can be tuned
        df_sky = dbutils.generate_random_skyobjects(
            args.ra,
            args.dec,
            n_sky_target,
        )
    else:
        logger.info("Sky objects will be generated using targetdb.")
        df_sky = dbutils.generate_skyobjects_from_targetdb(args.ra, args.dec, conf=conf)
        if args.reduce_sky_targets:
            n_sky_target = args.n_sky_random  # this value can be tuned
            if len(df_sky) > n_sky_target:
                df_sky = df_sky.sample(n_sky_target,
                                       ignore_index=True,
                                       random_state=1
                                       )
        logger.info(f"Fetched sky target DataFrame: \n{df_sky}")
        # df_sky = dbutils.generate_skyobjects_from_targetdb(
        #    args.ra,
        #    args.dec,
        #    conf=conf,
        #    # extra_where="LIMIT 1000",
        # )

    # print(df_sky)
    # exit()

    if args.filler:
        df_filler = dbutils.generate_targets_from_gaiadb(
            args.ra,
            args.dec,
            conf=conf,
            band_select="phot_g_mean_mag",
            mag_min=args.filler_mag_min,
            mag_max=args.filler_mag_max,
            good_astrometry=False,  # select bright stars which may have large astrometric errors.
            write_csv=True,
        )
        df_filler = dbutils.fixcols_gaiadb_to_targetdb(
            df_filler,
            proposal_id=args.filler_propid,
            target_type_id=1,    # SCIENCE
            input_catalog_id=4,  # Gaia DR3
            exptime=60.0,
            priority=9999,
        )
        if args.reduce_fillers:
            n_fillers = args.n_fillers_random 
            if len(df_filler) > n_fillers:
                df_filler = df_filler.sample(n_fillers,
                                       ignore_index=True,
                                       random_state=1
                                       )
        logger.info(f"Fetched fillers DataFrame: \n{df_filler}")
    else:
        df_filler = None
       
    vis, tp, tel, tgt, tgt_class_dict, is_no_target, bench = nfutils.fiber_allocation(
        df_targets,
        df_fluxstds,
        df_sky,
        args.ra,
        args.dec,
        args.pa,
        args.n_fluxstd,
        args.n_sky,
        args.observation_time,
        conf["netflow"]["use_gurobi"],
        dict(conf["gurobi"]) if conf["netflow"]["use_gurobi"] else None,
        args.pfs_instdata_dir,
        args.cobra_coach_dir,
        args.cobra_coach_module_version,
        args.sm,
        args.dot_margin,
        args.dot_penalty,
        cobra_location_group=cobra_location_group,
        min_sky_targets_per_location=min_sky_targets_per_location,
        location_group_penalty=location_group_penalty,
        cobra_instrument_region=cobra_instrument_region,
        min_sky_targets_per_instrument_region=min_sky_targets_per_instrument_region,
        instrument_region_penalty=instrument_region_penalty,
        num_reserved_fibers=0,
        fiber_non_allocation_cost=0.0,
        df_filler=df_filler,
        force_exptime=args.exptime,
    )
    # print(vis, tp, tel, tgt, tgt_classdict)
    # print(vis.items())

    # print(is_no_target)

    design = designutils.generate_pfs_design(
        df_targets,
        df_fluxstds,
        df_sky,
        vis,
        tp,
        tel,
        tgt,
        tgt_class_dict,
        bench,
        arms=args.arms,
        df_filler=df_filler,
        is_no_target=is_no_target,
        design_name=args.design_name,
    )
    if args.guide_star_id_exclude is None:
        guide_star_id_exclude = []
    else:
        guide_star_id_exclude = args.guide_star_id_exclude
    guidestars = designutils.generate_guidestars_from_gaiadb(
        args.ra,
        args.dec,
        args.pa,
        args.observation_time,
        args.telescope_elevation,
        conf=conf,
        guidestar_mag_min=args.guidestar_mag_min,
        guidestar_mag_max=args.guidestar_mag_max,
        guidestar_neighbor_mag_min=args.guidestar_neighbor_mag_min,
        guidestar_minsep_deg=args.guidestar_minsep_deg,
        # gaiadb_epoch=2015.0,
        # gaiadb_input_catalog_id=2,
        guide_star_id_exclude=guide_star_id_exclude,
    )

    design.guideStars = guidestars

    design.write(dirName=args.design_dir, fileName=design.filename)

    logger.info(
        f"pfsDesign file {design.filename} is created in the {args.design_dir} directory."
    )
    logger.info(
        "Number of SCIENCE fibers: {:}".format(len(np.where(design.targetType == 1)[0]))
    )
    logger.info(
        "Number of FLUXSTD fibers: {:}".format(len(np.where(design.targetType == 3)[0]))
    )
    logger.info(
        "Number of SKY fibers: {:}".format(len(np.where(design.targetType == 2)[0]))
    )
    logger.info("Number of AG stars: {:}".format(len(guidestars.objId)))


if __name__ == "__main__":
    main()
