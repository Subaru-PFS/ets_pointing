#!/usr/bin/env python3

import argparse
import functools
import os

import numpy as np
import pandas as pd
import toml
from astropy.time import Time
from astropy.utils import iers
from logzero import logger
from pfs.datamodel import PfsDesign
from pfs.datamodel import TargetType

import pointing_utils.dbutils as dbutils
import pointing_utils.designutils as designutils
import pointing_utils.nfutils as nfutils

# The following line seems to be needed to avoid IERS errors,
# though the default config is already `auto_download=True`.
iers.conf.auto_download = True
# iers.conf.iers_degraded_accuracy = "warn"


def get_arguments():

    parser = argparse.ArgumentParser()

    # input pfsDesign file
    parser.add_argument(
        "design_id",
        metavar="pfsDesignId",
        type=functools.partial(int, base=0),
        help="pfsDesignId (hex string with a prefix of 0x)",
    )
    parser.add_argument(
        "--design_indir",
        type=str,
        # dest="indir",
        default=".",
        help="Directory where the input pfsDesign file is stored. (default: .)",
    )

    # New observation time in UTC
    parser.add_argument(
        "--observation_time",
        type=str,
        default="2022-05-20T15:00:00Z",
        help="Planned time of observation in UTC (default: 2022-05-20T15:00:00Z)",
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
    parser.add_argument(
        "--design_outdir",
        type=str,
        default=".",
        help="directory for storing the output pfsDesign file (default: .)",
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

    # instrument parameter files
    parser.add_argument(
        "--pfs_instdata_dir",
        type=str,
        default="/work/pfs/commissioning/2022sep/fiber_allocation/pfs_instdata/",
        # default="/Users/monodera/Dropbox/NAOJ/PFS/Subaru-PFS/pfs_instdata/",
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
        nargs="+",
        type=float,
        default=1.0,
        help="Margin factor for dot avoidance (default: 1.0)",
    )

    args = parser.parse_args()

    # NOTE: astropy.time.Time.now() uses datetime.utcnow()
    if args.observation_time.lower() == "now":
        args.observation_time = Time.now().iso
        logger.info(
            f"Observation time is set to the current UTC {args.observation_time}."
        )

    print(args)

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


def main():

    args = get_arguments()

    conf = read_conf(args.conf)
    print(conf["netflow"]["use_gurobi"])

    in_design, df_sci, df_std, df_sky = load_input_design(
        args.design_id, indir=args.design_indir, exptime=args.exptime
    )

    vis, tp, tel, tgt, tgt_class_dict, is_no_target = nfutils.fiber_allocation(
        df_sci,
        df_std,
        df_sky,
        in_design.raBoresight,
        in_design.decBoresight,
        in_design.posAng,
        df_std.index.size,
        df_sky.index.size,
        args.observation_time,
        conf,
        args.pfs_instdata_dir,
        args.cobra_coach_dir,
        args.cobra_coach_module_version,
        args.sm,
        args.dot_margin,
        force_exptime=args.exptime,
    )

    # print(vis)

    out_design = designutils.generate_pfs_design(
        df_sci,
        df_std,
        df_sky,
        vis,
        tp,
        tel,
        tgt,
        tgt_class_dict,
        arms=in_design.arms,
        design_name=in_design.designName,
    )

    guidestars = designutils.generate_guidestars_from_gaiadb(
        in_design.raBoresight,
        in_design.decBoresight,
        in_design.posAng,
        args.observation_time,
        telescope_elevation=None,
        conf=conf,
        guidestar_mag_min=args.guidestar_mag_min,
        guidestar_mag_max=args.guidestar_mag_max,
        guidestar_neighbor_mag_min=args.guidestar_neighbor_mag_min,
        guidestar_minsep_deg=args.guidestar_minsep_deg,
    )

    out_design.guideStars = guidestars

    out_design.write(dirName=args.design_outdir, fileName=out_design.filename)

    logger.info(
        f"pfsDesign file {out_design.filename} is created in the {args.design_outdir} directory."
    )
    logger.info(
        "Number of SCIENCE fibers: {:} --> {:}".format(
            len(np.where(in_design.targetType == 1)[0]),
            len(np.where(out_design.targetType == 1)[0]),
        )
    )
    logger.info(
        "Number of FLUXSTD fibers: {:} --> {:}".format(
            len(np.where(in_design.targetType == 3)[0]),
            len(np.where(out_design.targetType == 3)[0]),
        )
    )
    logger.info(
        "Number of SKY fibers: {:} --> {:}".format(
            len(np.where(in_design.targetType == 2)[0]),
            len(np.where(out_design.targetType == 2)[0]),
        )
    )
    logger.info(
        "Number of AG stars: {:} --> {:}".format(
            len(in_design.guideStars.objId), len(guidestars.objId)
        )
    )


if __name__ == "__main__":
    main()
