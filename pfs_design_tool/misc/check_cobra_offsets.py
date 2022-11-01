import os

import matplotlib.pyplot as plt
import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.table import Table
from astropy.utils import iers
from logzero import logger
from pfs.datamodel import PfsDesign
from pfs.utils.coordinates.CoordTransp import CoordinateTransform
from pfs.utils.fiberids import FiberIds

# import pfs.drp.stella.utils.raster as raster

iers.conf.auto_download = True


def patchPfsDesign(pfsConfig, visit, datadir="."):
    # md = butler.get("raw_md", spectrograph=1, arm="r", visit=visit)

    hdr = fits.getheader(os.path.join(datadir, f"PFSA{visit:06d}12.fits"))

    altitude = hdr["ALTITUDE"]
    pa = hdr["INST-PA"]
    insrot = hdr["INSROT"]

    utc = f"{hdr['DATE-OBS']} {hdr['UT']}"

    logger.info(
        f"(altitude, pa, insrot, utc) from raw data: {altitude}, {pa}, {insrot}, {utc}"
    )

    c = SkyCoord(
        f'{hdr["RA_CMD"]} {hdr["DEC_CMD"]}', frame="icrs", unit=(u.hourangle, u.deg)
    )
    boresight = [[c.ra.degree], [c.dec.degree]]

    logger.info(f'(RA_CMD, DEC_CMD) from raw data: {hdr["RA_CMD"]}, {hdr["DEC_CMD"]}')
    logger.info(boresight)

    # boresight = raster.raDecStrToDeg(md["RA_CMD"], md["DEC_CMD"])
    # boresight = [[boresight[0]], [boresight[1]]]

    xnom, ynom = pfsConfig.pfiNominal.T
    xmm, ymm = CoordinateTransform(
        np.stack((pfsConfig.ra, pfsConfig.dec)),
        mode="sky_pfi",
        # za=90.0 - altitude,
        # za=0.0,
        pa=pa,
        cent=boresight,
        time=utc,
        # epoch=2015.5,
    )[0:2]

    pfsConfig.pfiNominal.T[0] = xmm
    pfsConfig.pfiNominal.T[1] = ymm


def patch_pfsdesign_by_rawdata(datadir=".", skip_patch=False):
    pfsDesignId = 0x7637277BB44703B8
    # pfsDesignId = 0x51ED77B5D46117C6
    pfsConfig = PfsDesign.read(pfsDesignId, datadir)

    logger.info(
        f"(PA, ra_boresight, dec_boresight) in pfsDesign: ({pfsConfig.posAng}, {pfsConfig.raBoresight}, {pfsConfig.decBoresight}"
    )

    visit = 78155

    if not skip_patch:

        patchPfsDesign(pfsConfig, visit, datadir=datadir)
        pfsConfig.designName += " MO"

        pfsConfig.write(
            dirName=datadir,
            fileName="pfsDesign-0x{:016x}_MO.fits".format(pfsConfig.pfsDesignId),
            allowSubset=True,
        )

    return "pfsDesign-0x{:016x}.fits".format(
        pfsConfig.pfsDesignId
    ), "pfsDesign-0x{:016x}_MO.fits".format(pfsConfig.pfsDesignId)


def load_design(pfs_design, hdu=1):
    logger.info(f"Load the pfsDesign file ({pfs_design})")
    tb = Table.read(pfs_design, hdu=hdu)
    return tb


def plot_designs(input_design_file, patched_design_file, datadir=".", outdir="plots"):

    gfm = FiberIds()

    def plot_one_design(design_file, ax, **kwargs):
        tb_design = load_design(design_file, hdu=1)
        x_alloc = np.full(len(gfm.fiberId), np.nan)
        y_alloc = np.full(len(gfm.fiberId), np.nan)
        for i in range(x_alloc.size):
            idx_fiber = tb_design["fiberId"] == gfm.fiberId[i]
            if np.any(idx_fiber):
                x_alloc[i], y_alloc[i] = tb_design["pfiNominal"][idx_fiber][0]
        idx_sm = gfm.spectrographId == 1
        ax.scatter(x_alloc[idx_sm], y_alloc[idx_sm], **kwargs)

    with plt.style.context("tableau-colorblind10"):

        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 12))
        ax.set_aspect("equal")

        plot_one_design(
            os.path.join(datadir, input_design_file),
            ax,
            **dict(
                s=3**2,
                c="C0",
                marker="o",
                edgecolors="none",
                alpha=0.5,
                label="Original Design",
            ),
        )

        plot_one_design(
            os.path.join(datadir, patched_design_file),
            ax,
            **dict(
                s=5**2,
                facecolors="none",
                marker="o",
                edgecolors="C0",
                linewidth=0.5,
                label="Patched Design",
            ),
        )

        ax.legend(loc="upper right")

        plt.savefig(
            os.path.join(outdir, "comp_patched_design.pdf"), bbox_inches="tight"
        )


if __name__ == "__main__":

    datadir = "testdata"

    input_design, patched_design = patch_pfsdesign_by_rawdata(
        datadir=datadir, skip_patch=False
    )

    plot_designs(input_design, patched_design, datadir=datadir, outdir="plots")
