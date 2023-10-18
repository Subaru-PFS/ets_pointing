#!/usr/bin/env python3

import argparse
import functools
import os

import astropy.units as u
import numpy as np
import pandas as pd
import toml
from astropy.time import Time
from astropy.utils import iers
from IPython.display import clear_output
from logzero import logger
from pfs.datamodel import PfsDesign, TargetType

from .pointing_utils import dbutils, designutils, nfutils

# The following line seems to be needed to avoid IERS errors,
# though the default config is already `auto_download=True`.
iers.conf.auto_download = True
# iers.conf.iers_degraded_accuracy = "warn"


def get_arguments():
    parser = argparse.ArgumentParser()

    # input pfsDesign file
    parser.add_argument(
        "infile",
        # metavar="pfsDesignId",
        # type=functools.partial(int, base=0),
        type=str,
        help="Input file (supposed to be a CVS format) containing results from PPP.",
    )
    # parser.add_argument(
    #     "--design_indir",
    #     type=str,
    #     # dest="indir",
    #     default=".",
    #     help="Directory where the input pfsDesign file is stored. (default: .)",
    # )

    # New observation time in UTC
    parser.add_argument(
        "--observation_time",
        type=str,
        default="2022-05-20T15:00:00Z",
        help="Planned time of observation in UTC (default: 2022-05-20T15:00:00Z)",
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
        default="brn",
        help="Spectrograph arms to expose, such as 'brn' and 'bmn' (default: 'brn')",
    )
    parser.add_argument(
        "--design_name",
        type=str,
        default=None,
        help="Human-readable design name (default: None)",
    )

    # Set exposure time
    parser.add_argument(
        "--exptime",
        type=float,
        default=900.0,
        help="Override the exptime (seconds) obtained from the database (default: None)",
    )

    # Database and Gurobi configurations
    parser.add_argument(
        "--conf",
        type=str,
        default="config.toml",
        help="Config file for the script to run. Must be a .toml file (default: config.toml)",
    )

    # output directories
    # parser.add_argument(
    #     "--design_outdir",
    #     type=str,
    #     default=".",
    #     help="directory for storing the output pfsDesign file (default: .)",
    # )
    parser.add_argument(
        "--design_dir",
        type=str,
        default=".",
        help="directory for storing pfsDesign files (default: .)",
    )
    parser.add_argument(
        "--cobra_coach_dir",
        type=str,
        default="./coach",
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

    # raster scan stars from gaiaDB
    parser.add_argument(
        "--raster_scan",
        action="store_true",
        help="Search stars for raster scan test (default: False)",
    )
    parser.add_argument(
        "--raster_mag_max",
        type=float,
        default=20.0,
        help="maximum magnitude in Gaia G for raster scan stars (default: 20.)",
    )
    parser.add_argument(
        "--raster_mag_min",
        type=float,
        default=12.0,
        help="minimum magnitude in Gaia G for raster scan stars (default: 12)",
    )
    parser.add_argument(
        "--raster_propid",
        default="S23A-EN16",
        help="Proposal-ID for raster scan stars (default: S23A-EN16)",
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

    # instrument parameter files
    parser.add_argument(
        "--pfs_instdata_dir",
        type=str,
        # default="/Users/monodera/Dropbox/NAOJ/PFS/Subaru-PFS/pfs_instdata/",
        # default="/work/pfs/commissioning/2022sep/fiber_allocation/pfs_instdata/",
        default="/work/pfs/commissioning/2023jul/fiber_allocation/pfs_instdata/",
        help="Location of pfs_instdata (default: /work/pfs/commissioning/2023jul/fiber_allocation/pfs_instdata/)",
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

    args = parser.parse_args()

    # NOTE: astropy.time.Time.now() uses datetime.utcnow()
    if args.observation_time.lower() == "now":
        args.observation_time = Time.now().iso
        logger.info(
            f"Observation time is set to the current UTC {args.observation_time}."
        )

    # print(args)

    return args


def read_conf(conf):
    config = toml.load(conf)
    return config


def load_input_design(design_id, indir=".", exptime=None, bands=["g", "r", "i"]):
    pfs_design = PfsDesign.read(pfsDesignId=design_id, dirName=indir)

    fib = {
        "sci": pfs_design.select(targetType=TargetType.SCIENCE),
        "std": pfs_design.select(targetType=TargetType.FLUXSTD),
        "sky": pfs_design.select(targetType=TargetType.SKY),
    }

    dataframes = {}

    for k, v in fib.items():
        df_tmp = pd.DataFrame(
            {
                "obj_id": v.objId,
                "ra": v.ra,
                "dec": v.dec,
                "tract": v.tract,
                "patch": v.patch,
                "catalog_id": v.catId,
                "target_type_id": v.targetType,
                "input_catalog_id": v.catId,
            },
        )
        dataframes[k] = df_tmp.copy(deep=True)

    dataframes["sci"]["priority"] = np.full(fib["sci"].objId.size, 1, dtype=int)
    dataframes["sci"]["effective_exptime"] = np.full(
        fib["sci"].objId.size, exptime, dtype=float
    )

    for i, band in enumerate(bands):
        for target_type in ["sci", "std"]:
            dataframes[target_type][f"filter_{band}"] = [
                fib[target_type].filterNames[iobj][i]
                for iobj in range(fib[target_type].objId.size)
            ]

            dataframes[target_type][f"psf_flux_{band}"] = [
                fib[target_type].psfFlux[iobj][i]
                for iobj in range(fib[target_type].objId.size)
            ]

    # # print(dataframes["sci"]["total_flux"][0])
    # print(dataframes["sci"][["filter_g", "filter_r", "filter_i"]])
    # print(dataframes["sci"][["psf_flux_g", "psf_flux_r", "psf_flux_i"]])
    # exit()

    return pfs_design, dataframes["sci"], dataframes["std"], dataframes["sky"]


def load_ppp_results(infile: str):
    df = pd.read_csv(infile)
    # print(df)

    pointings = df["pointing"].unique()
    n_pointings = pointings.size
    priorities = df["target_class"].unique()
    print(pointings, n_pointings, priorities)

    dict_pointings = {}

    for i, pointing in enumerate(pointings):
        df_pointing = df.loc[df["pointing"] == pointing, :].copy().reset_index()
        n_obj = df_pointing.index.size
        pseudo_obj_ids = np.random.randint(
            0, high=np.iinfo(np.int64).max, size=n_obj, dtype=np.int64
        )
        df_tmp = pd.DataFrame(
            {
                "obj_id": pseudo_obj_ids,
                "ra": df_pointing["ra_target"],
                "dec": df_pointing["dec_target"],
                "pmra": df_pointing["pmra_target"],
                "pmdec": df_pointing["pmdec_target"],
                "parallax": df_pointing["parallax_target"],
                "epoch": df_pointing["equinox_target"],
                "tract": np.full(n_obj, 0),
                "patch": np.full(n_obj, 0),
                "catalog_id": np.full(n_obj, 9),  # HSC-SSP PDR3 Wide
                "target_type_id": np.full(n_obj, 1),  # SCIENCE
                "input_catalog_id": np.full(n_obj, 9),  # HSC-SSP PDR3 Wide
                "ob_code": df_pointing["ob_code"],
                "proposal_id": df_pointing["proposal_id"],
                "priority": [
                    int(p.replace("sci_P", "")) for p in df_pointing["target_class"]
                ],
                "effective_exptime": 900.0,
                "psf_flux_g": [None] * n_obj,
                "psf_flux_r": [None] * n_obj,
                "psf_flux_i": [None] * n_obj,
                "psf_flux_z": [None] * n_obj,
                "psf_flux_y": [None] * n_obj,
                "psf_flux_error_g": [None] * n_obj,
                "psf_flux_error_r": [None] * n_obj,
                "psf_flux_error_i": [None] * n_obj,
                "psf_flux_error_z": [None] * n_obj,
                "psf_flux_error_y": [None] * n_obj,
            },
        )
        # print(df_tmp)

        dict_pointings[pointing.lower()] = {
            "pointing_name": pointing,
            "ra_center": df_pointing["ra_center"][0],
            "dec_center": df_pointing["dec_center"][0],
            "pa_center": df_pointing["pa_center"][0],
            "sci": df_tmp,
            "obj_id_original": df_pointing["obj_id"],
            "obj_id_dummy": pseudo_obj_ids,
            # "observation_time": observation_time,
            "observation_time": df_pointing["obstime"],
            "observation_date_in_hst": df_pointing["obsdate_in_hst"],
        }

    # print(dict_pointings)

    return pointings, dict_pointings


def reconfigure(
    conf, workDir=".", infile="ppp+qplan_outout.csv", clearOutput=False, logger=logger
):
    try:
        list_pointings, dict_pointings = load_ppp_results(os.path.join(workDir, infile))
    except FileNotFoundError:
        list_pointings, dict_pointings = load_ppp_results(
            os.path.join(workDir, conf["ppp"]["outputDir"], infile)
        )

    # in_design, df_sci, df_std, df_sky = load_input_design(
    #     args.design_id, indir=args.design_indir, exptime=args.exptime
    # )

    obstime0 = Time("2023-06-15T10:00:00")
    # obstime0 = Time("2023-07-01T00:00:00.000")  # UTC
    d_obstime = 20 * u.min

    design_filenames = []
    observation_times = []
    observation_dates_in_hst = []
    if conf["sfa"]["cobra_coach_module_version"].lower == "none":
        cobra_coach_module_version = None
    else:
        cobra_coach_module_version = conf["sfa"]["cobra_coach_module_version"]

    if conf["sfa"]["dot_penalty"].lower == "none":
        dot_penalty = None
    else:
        dot_penalty = conf["sfa"]["dot_penalty"]

    design_ids = {}
    for i, pointing in enumerate(list_pointings):
        if clearOutput == True:
            clear_output()
        # observation_time = Time.now().iso
        observation_time = str(dict_pointings[pointing.lower()]["observation_time"][0])
        observation_time = observation_time.replace(" ", "T") + "Z"
        observation_date_in_hst = str(
            dict_pointings[pointing.lower()]["observation_date_in_hst"][0]
        )

        # get science targets
        df_sci = dict_pointings[pointing.lower()]["sci"]

        # get flux standards
        df_fluxstds = dbutils.generate_fluxstds_from_targetdb(
            dict_pointings[pointing.lower()]["ra_center"],
            dict_pointings[pointing.lower()]["dec_center"],
            conf=conf,
            good_fluxstd=conf["sfa"]["good_fluxstd"],
            flags_dist=conf["sfa"]["fluxstd_flags_dist"],
            flags_ebv=conf["sfa"]["fluxstd_flags_ebv"],
            mag_min=conf["sfa"]["fluxstd_mag_min"],
            mag_max=conf["sfa"]["fluxstd_mag_max"],
            mag_filter=conf["sfa"]["fluxstd_mag_filter"],
            min_prob_f_star=conf["sfa"]["fluxstd_min_prob_f_star"],
            write_csv=False,
        )

        # get sky targets
        if conf["sfa"]["n_sky"] == 0:
            logger.info("No sky object will be sent to netflow")
            df_sky = dbutils.generate_random_skyobjects(
                dict_pointings[pointing.lower()]["ra_center"],
                dict_pointings[pointing.lower()]["dec_center"],
                0,
            )
        elif conf["sfa"]["sky_random"]:
            logger.info("Random sky objects will be generated.")
            # n_sky_target = (df_targets.size + df_fluxstds.size) * 2
            n_sky_target = conf["sfa"]["n_sky_random"]  # this value can be tuned
            df_sky = dbutils.generate_random_skyobjects(
                dict_pointings[pointing.lower()]["ra_center"],
                dict_pointings[pointing.lower()]["dec_center"],
                n_sky_target,
            )
        else:
            logger.info("Sky objects will be generated using targetdb.")
            df_sky = dbutils.generate_skyobjects_from_targetdb(
                dict_pointings[pointing.lower()]["ra_center"],
                dict_pointings[pointing.lower()]["dec_center"],
                conf=conf,
            )
            if conf["sfa"]["reduce_sky_targets"]:
                n_sky_target = conf["sfa"]["n_sky_random"]  # this value can be tuned
                if len(df_sky) > n_sky_target:
                    df_sky = df_sky.sample(
                        n_sky_target, ignore_index=True, random_state=1
                    )
            logger.info(f"Fetched target DataFrame: \n{df_sky}")

        # get raster targets (optional)
        raster = conf["sfa"]["raster"]
        if conf["sfa"]["raster"] == True:
            df_raster = dbutils.generate_targets_from_gaiadb(
                dict_pointings[pointing.lower()]["ra_center"],
                dict_pointings[pointing.lower()]["dec_center"],
                conf=conf,
                band_select="phot_g_mean_mag",
                mag_min=conf["sfa"]["raster_mag_min"],
                mag_max=conf["sfa"]["raster_mag_max"],
                good_astrometry=False,  # select bright stars which may have large astrometric errors.
                write_csv=False,
            )
            df_raster = dbutils.fixcols_gaiadb_to_targetdb(
                df_raster,
                proposal_id="S23A-EN16",
                target_type_id=1,  # SCIENCE
                input_catalog_id=4,  # Gaia DR3
                exptime=900.0,
                priority=9999,
            )
        else:
            df_raster = None

        (
            vis,
            tp,
            tel,
            tgt,
            tgt_class_dict,
            is_no_target,
            bench,
        ) = nfutils.fiber_allocation(
            df_sci,
            df_fluxstds,
            df_sky,
            dict_pointings[pointing.lower()]["ra_center"],
            dict_pointings[pointing.lower()]["dec_center"],
            dict_pointings[pointing.lower()]["pa_center"],
            conf["sfa"]["n_fluxstd"],
            conf["sfa"]["n_sky"],
            observation_time,
            conf,
            conf["sfa"]["pfs_instdata_dir"],
            conf["sfa"]["cobra_coach_dir"],
            None,
            conf["sfa"]["sm"],
            conf["sfa"]["dot_margin"],
            None,
            df_raster=df_raster,
            force_exptime=conf["ppp"]["TEXP_NOMINAL"],
        )

        design = designutils.generate_pfs_design(
            df_sci,
            df_fluxstds,
            df_sky,
            vis,
            tp,
            tel,
            tgt,
            tgt_class_dict,
            bench,
            arms=conf["sfa"]["arms"],
            df_raster=df_raster,
            is_no_target=is_no_target,
            design_name=dict_pointings[pointing.lower()]["pointing_name"],
        )

        guidestars = designutils.generate_guidestars_from_gaiadb(
            dict_pointings[pointing.lower()]["ra_center"],
            dict_pointings[pointing.lower()]["dec_center"],
            dict_pointings[pointing.lower()]["pa_center"],
            observation_time,
            telescope_elevation=None,
            conf=conf,
            guidestar_mag_min=conf["sfa"]["guidestar_mag_min"],
            guidestar_mag_max=conf["sfa"]["guidestar_mag_max"],
            guidestar_neighbor_mag_min=conf["sfa"]["guidestar_neighbor_mag_min"],
            guidestar_minsep_deg=conf["sfa"]["guidestar_minsep_deg"],
            # gaiadb_epoch=2015.0,
            # gaiadb_input_catalog_id=2,
        )

        design.guideStars = guidestars
        design_dir = os.path.join(workDir, conf["ope"]["designPath"])
        design.write(dirName=design_dir, fileName=design.filename)

        design_filenames.append(design.filename)
        observation_times.append(observation_time)
        observation_dates_in_hst.append(observation_date_in_hst)

        design_ids[observation_time] = design.pfsDesignId

        df_obj_id = pd.DataFrame(
            {
                "obj_id_origial": dict_pointings[pointing.lower()]["obj_id_original"],
                "obj_id_dummy": dict_pointings[pointing.lower()]["obj_id_dummy"],
            }
        ).to_csv(
            os.path.join(design_dir, f"{design.filename}_obj_ids.csv"), index=False
        )

        print("### design saved ###")

        logger.info(
            f"pfsDesign file {design.filename} for {pointing} is created in the {design_dir} directory."
        )
        logger.info(
            "Number of SCIENCE fibers: {:}".format(
                len(np.where(design.targetType == 1)[0])
            )
        )
        logger.info(
            "Number of FLUXSTD fibers: {:}".format(
                len(np.where(design.targetType == 3)[0])
            )
        )
        logger.info(
            "Number of SKY fibers: {:}".format(len(np.where(design.targetType == 2)[0]))
        )
        logger.info("Number of AG stars: {:}".format(len(guidestars.objId)))
        logger.info(f"Observation Time: {observation_time}")

    df_summary = pd.DataFrame(
        {
            "pointing": list_pointings,
            "design_filename": design_filenames,
            "observation_time": observation_times,
            "observation_date_in_hst": observation_dates_in_hst,
        }
    )
    infile_base = os.path.splitext(os.path.basename(infile))[0]
    df_summary.to_csv(
        os.path.join(workDir, f"output/summary_reconfigure_ppp-{infile_base}.csv"),
        index=False,
    )

    return list_pointings, dict_pointings, design_ids, observation_dates_in_hst


def main():
    args = get_arguments()

    conf = read_conf(args.conf)
    print(conf["netflow"]["use_gurobi"])

    reconfigure(conf=conf)


if __name__ == "__main__":
    main()
