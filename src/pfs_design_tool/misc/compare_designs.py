#!/usr/bin/env python3

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


def load_design(pfs_design, hdu=1):
    logger.info(f"Load the pfsDesign file ({pfs_design})")
    tb = Table.read(pfs_design, hdu=hdu)
    return tb


def main(design_file1, design_file2, plotname=None, datadir="testdata", outdir="plots"):

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
            os.path.join(datadir, design_file1),
            ax,
            **dict(
                s=3**2,
                c="C0",
                marker="o",
                edgecolors="none",
                alpha=0.5,
                label=design_file1,
            ),
        )

        plot_one_design(
            os.path.join(datadir, design_file2),
            ax,
            **dict(
                s=5**2,
                facecolors="none",
                marker="o",
                edgecolors="C0",
                linewidth=0.5,
                label=design_file2,
            ),
        )

        ax.legend(loc="upper right")

        plt.savefig(os.path.join(outdir, plotname + ".pdf"), bbox_inches="tight")
        plt.savefig(
            os.path.join(outdir, plotname + ".png"), bbox_inches="tight", dpi=300
        )


if __name__ == "__main__":

    design_file1 = "pfsDesign-0x7637277bb44703b8.fits"
    design_file2 = "pfsDesign-0x2a3e0e7f53f47587.fits"
    main(design_file1, design_file2, plotname="comp_two_designs")

    # design_file1 = "pfsDesign-0x7637277bb44703b8.fits"
    # design_file2 = "pfsDesign-0x7637277bb44703b8_MO.fits"
    # main(design_file1, design_file2, plotname="comp_two_designs_patched")

    # design_file1 = "pfsDesign-0x7637277bb44703b8_MO.fits"
    # design_file2 = "pfsDesign-0x51ed77b5d46117c6.fits"
    # main(design_file1, design_file2, plotname="comp_two_designs_patchednew")
