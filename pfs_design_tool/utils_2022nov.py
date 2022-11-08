import logging
import sys
import os
import datetime
import numpy as np
import psycopg2
import pandas as pd
from astropy.io import fits
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.ticker import MultipleLocator
from matplotlib.patches import Circle
from matplotlib.collections import PatchCollection
from mpl_toolkits.axes_grid1 import make_axes_locatable
from astroquery.gaia import Gaia
from astropy.coordinates import SkyCoord
import astropy.units as u

from pfs.utils.fiberids import FiberIds
from pfs.datamodel import PfsDesign, FiberStatus, TargetType

from astropy.utils import iers
from astropy.time import Time
# os.path.append('/work/pfs/commissioning/2022nov/fiber_allocation/pfs_utils/python')
from pfs.utils.coordinates.CoordTransp import CoordinateTransform as ctrans

iers.conf.auto_download = True
logging.disable()

plt.rcParams["font.size"] = 12
plt.rcParams["xtick.direction"] = "in"
plt.rcParams["ytick.direction"] = "in"
plt.rcParams["xtick.top"] = True
plt.rcParams["xtick.bottom"] = True
plt.rcParams["ytick.left"] = True
plt.rcParams["ytick.right"] = True
plt.rcParams["xtick.major.size"] = 5.0
plt.rcParams["ytick.major.size"] = 5.0
plt.rcParams["xtick.major.width"] = 1.0
plt.rcParams["ytick.major.width"] = 1.0
plt.rcParams["xtick.minor.size"] = 3.0
plt.rcParams["ytick.minor.size"] = 3.0
plt.rcParams["xtick.minor.width"] = 0.6
plt.rcParams["ytick.minor.width"] = 0.6
plt.rcParams["xtick.labelsize"] = 12
plt.rcParams["ytick.labelsize"] = 12
plt.rcParams["axes.labelsize"] = 12
plt.rcParams["figure.facecolor"] = 'white'

sm = [1, 3]
gfm = FiberIds()


def is_smx(pfsDesign, moduleIds=[1, 3]):
    ''' isSmX '''
    isSmX = np.full(len(pfsDesign.fiberId), False)
    for x in moduleIds:
        cobrasForSmX = gfm.cobrasForSpectrograph(x)  # cobra index
        for i, fid in enumerate(pfsDesign.fiberId):
            cid = gfm.fiberIdToCobraId(fid)  # cobra ID
            if cid in cobrasForSmX+1:
                isSmX[i] = True
    return isSmX


def pfi2sky(pfsDesign, observation_time, epoch=2016.0):
    ''' coordinate transformation utils '''
    pmra = np.array([0.0 for _ in range(len(pfsDesign.ra))])
    pmdec = np.array([0.0 for _ in range(len(pfsDesign.ra))])
    parallax = np.array([1.0e-07 for _ in range(len(pfsDesign.ra))])

    tmp = np.array([pfsDesign.pfiNominal[:, 0], pfsDesign.pfiNominal[:, 1]])
    tmp = ctrans(
        xyin=tmp,
        mode="pfi_sky",
        pa=pfsDesign.posAng,
        cent=np.array([pfsDesign.raBoresight, pfsDesign.decBoresight]).reshape((2, 1)),
        pm=np.stack([pmra, pmdec], axis=0),
        par=parallax,
        time=observation_time,
        epoch=epoch,
    )

    sky_x = tmp[0, :]
    sky_y = tmp[1, :]

    return sky_x, sky_y


def pfi2sky_array(pfi_x, pfi_y, pfsDesign, observation_time, epoch=2016.0):
    pmra = np.array([0.0 for _ in range(len(pfi_x))])
    pmdec = np.array([0.0 for _ in range(len(pfi_x))])
    parallax = np.array([1.0e-07 for _ in range(len(pfi_x))])

    tmp = np.array([pfi_x, pfi_y])
    tmp = ctrans(
        xyin=tmp,
        mode="pfi_sky",
        pa=pfsDesign.posAng,
        cent=np.array([pfsDesign.raBoresight, pfsDesign.decBoresight]).reshape((2, 1)),
        pm=np.stack([pmra, pmdec], axis=0),
        par=parallax,
        time=observation_time,
        epoch=epoch,
    )

    sky_x = tmp[0, :]
    sky_y = tmp[1, :]

    return sky_x, sky_y


def sky2pfi(pfsDesign, observation_time, epoch=2016.0):
    pmra = np.array([0.0 for _ in range(len(pfsDesign.ra))])
    pmdec = np.array([0.0 for _ in range(len(pfsDesign.ra))])
    parallax = np.array([1.0e-07 for _ in range(len(pfsDesign.ra))])

    tmp = np.array([pfsDesign.ra, pfsDesign.dec])
    tmp = ctrans(
        xyin=tmp,
        mode="sky_pfi",
        pa=pfsDesign.posAng,
        cent=np.array([pfsDesign.raBoresight, pfsDesign.decBoresight]).reshape((2, 1)),
        pm=np.stack([pmra, pmdec], axis=0),
        par=parallax,
        time=observation_time,
        epoch=epoch,
    )

    pfi_x = tmp[0, :]
    pfi_y = tmp[1, :]

    return pfi_x, pfi_y


def sky2pfi_array(sky_x, sky_y, pfsDesign, observation_time, epoch=2016.0):
    pmra = np.array([0.0 for _ in range(len(sky_x))])
    pmdec = np.array([0.0 for _ in range(len(sky_x))])
    parallax = np.array([1.0e-07 for _ in range(len(sky_x))])

    tmp = np.array([sky_x, sky_y])
    tmp = ctrans(
        xyin=tmp,
        mode="sky_pfi",
        pa=pfsDesign.posAng,
        cent=np.array([pfsDesign.raBoresight, pfsDesign.decBoresight]).reshape((2, 1)),
        pm=np.stack([pmra, pmdec], axis=0),
        par=parallax,
        time=observation_time,
        epoch=epoch,
    )

    pfi_x = tmp[0, :]
    pfi_y = tmp[1, :]

    return pfi_x, pfi_y


def get_num_targets_in_patrol_region(bench, pfsDesign, gaia_info, cobra_ids_use):
    ''' get number of targets in the patrol region '''
    ''' get all gaia sources '''
    gaia_all_id = gaia_info[0]
    gaia_all_x = gaia_info[1]
    gaia_all_y = gaia_info[2]

    ''' get assigned gaia sources '''
    assigned_id = np.array(pfsDesign.objId, dtype='int64')
    assigned_x = np.array(pfsDesign.pfiNominal[:, 0])
    assigned_y = np.array(pfsDesign.pfiNominal[:, 1])

    gaia_assigned_in_fov_id = []
    gaia_assigned_in_fov_x = []
    gaia_assigned_in_fov_y = []
    for i, x, y in zip(assigned_id, assigned_x, assigned_y):
        msk = i in gaia_all_id
        if len(assigned_id[msk]) == 1:
            gaia_assigned_in_fov_id.append(assigned_id[msk][0])
            gaia_assigned_in_fov_x.append(assigned_x[msk][0])
            gaia_assigned_in_fov_y.append(assigned_y[msk][0])
    gaia_assigned_in_fov_id = np.array(gaia_assigned_in_fov_id)
    gaia_assigned_in_fov_x = np.array(gaia_assigned_in_fov_x)
    gaia_assigned_in_fov_y = np.array(gaia_assigned_in_fov_y)

    ''' get assigned gaia sources in patrol area of interested SMs '''
    flg = is_smx(pfsDesign, moduleIds=sm)
    assigned_id = assigned_id[flg]
    assigned_x = assigned_x[flg]
    assigned_y = assigned_y[flg]

    gaia_assigned_in_sms_id = []
    gaia_assigned_in_sms_x = []
    gaia_assigned_in_sms_y = []
    for i, x, y in zip(assigned_id, assigned_x, assigned_y):
        msk = i in gaia_all_id
        if len(assigned_id[msk]) == 1:
            gaia_assigned_in_sms_id.append(assigned_id[msk][0])
            gaia_assigned_in_sms_x.append(assigned_x[msk][0])
            gaia_assigned_in_sms_y.append(assigned_y[msk][0])
    gaia_assigned_in_sms_id = np.array(gaia_assigned_in_sms_id)
    gaia_assigned_in_sms_x = np.array(gaia_assigned_in_sms_x)
    gaia_assigned_in_sms_y = np.array(gaia_assigned_in_sms_y)

    ''' get all gaia sources in FoV '''
    gaia_all_in_fov_id = []
    gaia_all_in_fov_x = []
    gaia_all_in_fov_y = []
    for cobra_id, center in enumerate(bench.cobras.centers):
        for i, x, y in zip(gaia_all_id, gaia_all_x, gaia_all_y):
            if (center.real - x)**2 + (center.imag - y)**2 <= (9.5/2)**2:
                if i not in gaia_all_in_fov_id:
                    gaia_all_in_fov_id.append(i)
                    gaia_all_in_fov_x.append(x)
                    gaia_all_in_fov_y.append(y)
    gaia_all_in_fov_id = np.unique(np.array(gaia_all_in_fov_id))
    gaia_all_in_fov_x = np.unique(np.array(gaia_all_in_fov_x))
    gaia_all_in_fov_y = np.unique(np.array(gaia_all_in_fov_y))

    ''' get all gaia sources in patrol area of interested SMs '''
    gaia_all_in_sms_id = []
    gaia_all_in_sms_x = []
    gaia_all_in_sms_y = []
    for cobra_id, center in enumerate(bench.cobras.centers):
        if cobra_id in cobra_ids_use:
            for i, x, y in zip(gaia_all_id, gaia_all_x, gaia_all_y):
                if (center.real - x)**2 + (center.imag - y)**2 <= (9.5/2)**2:
                    if i not in gaia_all_in_sms_id:
                        gaia_all_in_sms_id.append(i)
                        gaia_all_in_sms_x.append(x)
                        gaia_all_in_sms_y.append(y)
    gaia_all_in_sms_id = np.unique(np.array(gaia_all_in_sms_id))
    gaia_all_in_sms_x = np.unique(np.array(gaia_all_in_sms_x))
    gaia_all_in_sms_y = np.unique(np.array(gaia_all_in_sms_y))

    num_gaia_all = len(gaia_all_id)
    num_gaia_all_in_fov = len(gaia_all_in_fov_id)
    num_gaia_assigned_in_fov = len(gaia_assigned_in_fov_id)
    num_gaia_all_in_sms = len(gaia_all_in_sms_id)
    num_gaia_assigned_in_sms = len(gaia_assigned_in_sms_id)

    return num_gaia_all, num_gaia_all_in_fov, num_gaia_assigned_in_fov, \
        num_gaia_all_in_sms, num_gaia_assigned_in_sms


class CheckDesign(object):

    def __init__(self,
                 pfsDesignId=None,
                 obsTime=None,
                 objId=None,
                 designDir='.',
                 dataDir='.',
                 repoDir='.',
                 dotMargin=1.0,
                 gaiaCsv=None,
                 gaia_gmag_min=12.0,
                 gaia_gmag_max=12.5,
                 ):
        self.pfsDesignId = pfsDesignId
        self.obsTime = obsTime
        self.objId = objId
        self.designDir = designDir
        self.dataDir = dataDir
        self.repoDir = repoDir
        self.dotMargin = dotMargin
        self.gaiaCsv = gaiaCsv
        self.gaia_gmag_min = gaia_gmag_min
        self.gaia_gmag_max = gaia_gmag_max

        if pfsDesignId is not None:
            self.setPfsDesignId(self.pfsDesignId)
        else:
            self.pfsDesignId = self.pfsDesignId

    def setPfsDesignId(self, pfsDesignId):
        self.pfsDesignId = pfsDesignId
        self.pfsDesign = PfsDesign.read(pfsDesignId=self.pfsDesignId, dirName=self.designDir)
        self.gs = self.pfsDesign.guideStars

    def updatePfsDesignId(self, pfsDesignId):
        self.setPfsDesignId(pfsDesignId)

    def configGeometry(self):
        ''' get cobra+dots geometry '''
        sys.path.append(os.path.join(self.repoDir, 'ics_fpsActor/python'))
        sys.path.append(os.path.join(self.repoDir, 'spt_operational_database/python'))
        sys.path.append(os.path.join(self.repoDir, 'ics_cobraCharmer/python'))
        sys.path.append(os.path.join(self.repoDir, 'ics_cobraOps/python'))

        from ics.cobraCharmer.pfiDesign import PFIDesign
        from ics.cobraOps.Bench import Bench
        from ics.cobraOps.BlackDotsCalibrationProduct import BlackDotsCalibrationProduct
        from ics.cobraOps.cobraConstants import NULL_TARGET_ID
        from ics.cobraOps.cobraConstants import NULL_TARGET_POSITION
        # from ics.cobraOps.CollisionSimulator2 import CollisionSimulator2
        # from ics.cobraOps.TargetGroup import TargetGroup
        from procedures.moduleTest.cobraCoach import CobraCoach

        def getBench(
            pfs_instdata_dir,
            cobra_coach_dir,
            cobra_coach_module_version,
            sm,
            black_dot_radius_margin,
        ):
            os.environ["PFS_INSTDATA_DIR"] = pfs_instdata_dir
            cobraCoach = CobraCoach(
                "fpga", loadModel=False, trajectoryMode=True, rootDir=cobra_coach_dir
            )
            cobraCoach.loadModel(version="ALL", moduleVersion=cobra_coach_module_version)
            # Get the calibration product
            calibrationProduct = cobraCoach.calibModel
            # Set some dummy center positions and phi angles for those cobras that have
            # zero centers
            zeroCenters = calibrationProduct.centers == 0
            calibrationProduct.centers[zeroCenters] = np.arange(np.sum(zeroCenters)) * 300j
            calibrationProduct.phiIn[zeroCenters] = -np.pi
            calibrationProduct.phiOut[zeroCenters] = 0
            print("Cobras with zero centers: %i" % np.sum(zeroCenters))
            # Use the median value link lengths in those cobras with zero link lengths
            zeroLinkLengths = np.logical_or(
                calibrationProduct.L1 == 0, calibrationProduct.L2 == 0
            )
            calibrationProduct.L1[zeroLinkLengths] = np.median(
                calibrationProduct.L1[~zeroLinkLengths]
            )
            calibrationProduct.L2[zeroLinkLengths] = np.median(
                calibrationProduct.L2[~zeroLinkLengths]
            )
            print("Cobras with zero link lengths: %i" % np.sum(zeroLinkLengths))
            # Use the median value link lengths in those cobras with too long link lengths
            tooLongLinkLengths = np.logical_or(
                calibrationProduct.L1 > 100, calibrationProduct.L2 > 100
            )
            calibrationProduct.L1[tooLongLinkLengths] = np.median(
                calibrationProduct.L1[~tooLongLinkLengths]
            )
            calibrationProduct.L2[tooLongLinkLengths] = np.median(
                calibrationProduct.L2[~tooLongLinkLengths]
            )
            print("Cobras with too long link lengths: %i" % np.sum(tooLongLinkLengths))

            # Limit spectral modules
            gfm = FiberIds()  # 2604
            self.cobra_ids_use = np.array([], dtype=np.uint16)
            for sm_use in sm:
                self.cobra_ids_use = np.append(self.cobra_ids_use, gfm.cobrasForSpectrograph(sm_use))

            # set Bad Cobra status for unused spectral modules
            for cobra_id in range(calibrationProduct.nCobras):
                if cobra_id not in self.cobra_ids_use:
                    calibrationProduct.status[cobra_id] = ~PFIDesign.COBRA_OK_MASK

            # Get the black dots calibration product
            calibrationFileName = os.path.join(
                os.environ["PFS_INSTDATA_DIR"], "data/pfi/dot", "black_dots_mm.csv"
            )
            blackDotsCalibrationProduct = BlackDotsCalibrationProduct(calibrationFileName)

            # Create the bench instance
            bench = Bench(
                layout="calibration",
                calibrationProduct=calibrationProduct,
                blackDotsCalibrationProduct=blackDotsCalibrationProduct,
                blackDotsMargin=black_dot_radius_margin,
            )
            print("Number of cobras:", bench.cobras.nCobras)

            return cobraCoach, bench

        ''' load bench information '''
        pfs_instdata_dir = os.path.join(self.repoDir, 'pfs_instdata')
        cobra_coach_dir = 'cobracoach'
        cobra_coach_module_version = None
        black_dot_radius_margin = self.dotMargin
        self.cobra_coach, self.bench = getBench(pfs_instdata_dir, cobra_coach_dir,
                                                cobra_coach_module_version,
                                                sm, black_dot_radius_margin)
        ''' get cobra+dots geometry '''
        self.cobra_mpl_patches = []
        self.cobra_mpl_id = []
        for i, center in enumerate(self.bench.cobras.centers):
            circle = Circle((center.real, center.imag), 9.5/2)
            self.cobra_mpl_patches.append(circle)
            self.cobra_mpl_id.append(i+1)
        self.dot_mpl_patches = []
        self.dot2_mpl_patches = []
        for center, radius in zip(self.bench.blackDots.centers, self.bench.blackDots.radius):
            circle = Circle((center.real, center.imag), radius)
            self.dot_mpl_patches.append(circle)
            circle2 = Circle((center.real, center.imag), radius*self.dotMargin)
            self.dot2_mpl_patches.append(circle2)

        ''' getGaiaSources '''
        self.getGaiaSources()

    def getGaiaSources(self):
        ''' read Gaia sources '''
        df = pd.read_csv(os.path.join(self.dataDir, self.gaiaCsv))
        gmag = np.array(df['phot_g_mean_mag'])
        msk = (gmag < self.gaia_gmag_max) * (gmag > self.gaia_gmag_min)

        id_gaia_bright = np.array(df[msk]['source_id'])
        ra_gaia_bright = df[msk]['ra']
        dec_gaia_bright = df[msk]['dec']
        self.x_gaia_bright, self.y_gaia_bright = sky2pfi_array(ra_gaia_bright,
                                                               dec_gaia_bright,
                                                               self.pfsDesign,
                                                               self.obsTime)
        self.gaia_info = (id_gaia_bright, self.x_gaia_bright, self.y_gaia_bright)

    def check_statistics(self):
        ''' check pfsDesign statistics '''

        ''' check target types '''
        isSm1 = is_smx(self.pfsDesign, moduleIds=[1])
        isSm3 = is_smx(self.pfsDesign, moduleIds=[3])
        # isSm13 = isSm1 + isSm3
        isSci = (self.pfsDesign.targetType == TargetType.SCIENCE)
        isFst = (self.pfsDesign.targetType == TargetType.FLUXSTD)
        isSky = (self.pfsDesign.targetType == TargetType.SKY)
        isUna = (self.pfsDesign.targetType == TargetType.UNASSIGNED)
        try:
            isTgt = (self.pfsDesign.targetType == TargetType.SCIENCE) * \
                (self.pfsDesign.objId == int(self.objId))
        except ValueError:
            isTgt = np.full(len(self.pfsDesign), False)

        print("=====================================")
        if len(isTgt[isTgt]) > 0:
            print("The number of MAIN       (SM1) : %4d" % (len(self.pfsDesign[isTgt*isSm1])))
        print("The number of SCIENCE    (SM1) : %4d" % (len(self.pfsDesign[isSci*isSm1])))
        print("The number of FLUXSTD    (SM1) : %4d" % (len(self.pfsDesign[isFst*isSm1])))
        print("The number of SKY        (SM1) : %4d" % (len(self.pfsDesign[isSky*isSm1])))
        print("The number of UNASSIGNED (SM1) : %4d" % (len(self.pfsDesign[isUna*isSm1])))
        print("=====================================")
        if len(isTgt[isTgt]) > 0:
            print("The number of MAIN       (SM3) : %4d" % (len(self.pfsDesign[isTgt*isSm3])))
        print("The number of SCIENCE    (SM3) : %4d" % (len(self.pfsDesign[isSci*isSm3])))
        print("The number of FLUXSTD    (SM3) : %4d" % (len(self.pfsDesign[isFst*isSm3])))
        print("The number of SKY        (SM3) : %4d" % (len(self.pfsDesign[isSky*isSm3])))
        print("The number of UNASSIGNED (SM3) : %4d" % (len(self.pfsDesign[isUna*isSm3])))
        print("=====================================")

        ''' check number of guide stars '''
        num_gs = []
        for i in range(6):
            msk = self.gs.agId == i
            num_gs.append(f'{len(self.gs.agId[msk])}')
        print("The number of GuideStars : %s" % ('/'.join(num_gs)))
        print("=====================================")

        ''' check bright targets (SM1) '''
        flg = (isSci+isFst)*isSm1
        f = np.array(self.pfsDesign[flg].psfFlux)
        gmag = -2.5*np.log10(f[:, 0]*1e-09) + 8.9  # conversion from nJy to ABmag
        fid = self.pfsDesign[flg].fiberId
        pos = self.pfsDesign[flg].pfiNominal
        msk = (gmag < self.gaia_gmag_max) * (gmag > self.gaia_gmag_min)
        bright_center_pfi_x = pos[msk].T[0]
        bright_center_pfi_y = pos[msk].T[1]
        print("The number of bright objects (%.1f<g<%.1f) (SM1): %d" %
              (self.gaia_gmag_min, self.gaia_gmag_max, len(fid[msk])))
        print("fiberId:", fid[msk])
        print("gmag:", gmag[msk])
        print("X:", bright_center_pfi_x)
        print("Y:", bright_center_pfi_y)
        print("=====================================")

        ''' check bright targets (SM3) '''
        flg = (isSci+isFst)*isSm3
        f = np.array(self.pfsDesign[flg].psfFlux)
        gmag = -2.5*np.log10(f[:, 0]*1e-09) + 8.9  # conversion from nJy to ABmag
        fid = self.pfsDesign[flg].fiberId
        pos = self.pfsDesign[flg].pfiNominal
        msk = (gmag < self.gaia_gmag_max) * (gmag > self.gaia_gmag_min)
        bright_center_pfi_x = pos[msk].T[0]
        bright_center_pfi_y = pos[msk].T[1]
        print("The number of bright objects (%.1f<g<%.1f) (SM3): %d" %
              (self.gaia_gmag_min, self.gaia_gmag_max, len(fid[msk])))
        print("fiberId:", fid[msk])
        print("gmag:", gmag[msk])
        print("X:", bright_center_pfi_x)
        print("Y:", bright_center_pfi_y)
        print("=====================================")

        num_gaia_all, num_gaia_all_in_fov, num_gaia_assigned_in_fov, \
            num_gaia_all_in_sms, num_gaia_assigned_in_sms = \
            get_num_targets_in_patrol_region(self.bench,
                                             self.pfsDesign,
                                             self.gaia_info,
                                             self.cobra_ids_use
                                             )
        gaia_completeness_in_fov_new = num_gaia_assigned_in_fov / num_gaia_all_in_fov
        gaia_completeness_in_sms_new = num_gaia_assigned_in_sms / num_gaia_all_in_sms
        print("Bright Gaia source coverage: %.2f" % (gaia_completeness_in_sms_new))
        return gaia_completeness_in_fov_new, gaia_completeness_in_sms_new

    def plot_pfi_fov(self):
        isSm1 = is_smx(self.pfsDesign, moduleIds=[1])
        isSm3 = is_smx(self.pfsDesign, moduleIds=[3])
        isSm13 = isSm1 + isSm3
        isSci = (self.pfsDesign.targetType == TargetType.SCIENCE)
        isFst = (self.pfsDesign.targetType == TargetType.FLUXSTD)
        isSky = (self.pfsDesign.targetType == TargetType.SKY)
        try:
            isTgt = (self.pfsDesign.targetType == TargetType.SCIENCE) * \
                (self.pfsDesign.objId == int(self.objId))
        except ValueError:
            isTgt = np.full(len(self.pfsDesign), False)

        fig = plt.figure(figsize=(6, 6))
        axe = fig.add_subplot(111)
        axe.set_title(f'{self.objId}')
        axe.set_xlabel('X (mm)')
        axe.set_ylabel('Y (mm)')
        axe.grid(color='gray', linestyle='dotted', linewidth=1)
        axe.set_xlim(-250., +250.)
        axe.set_ylim(-250., +250.)

        x = self.pfsDesign.pfiNominal[:, 0]
        y = self.pfsDesign.pfiNominal[:, 1]

        ''' plot SCIENCE & FLUXSTD & SKY '''
        if len(isTgt[isTgt]) > 0:
            axe.scatter(x[isTgt*isSm1], y[isTgt*isSm13],
                        marker='o', s=100, facecolor='red', edgecolor='k', alpha=0.7,
                        zorder=2,
                        label=f'MAIN targets ({len(x[isTgt*isSm13])})'
                        )
        axe.scatter(x[isSci*isSm1], y[isSci*isSm1],
                    marker='o', s=50, facecolor='C0', edgecolor='k', alpha=0.7,
                    zorder=1,
                    label=f'SCIENCE targets (SM1) ({len(x[isSci*isSm1])})'
                    )
        axe.scatter(x[isSci*isSm3], y[isSci*isSm3],
                    marker='^', s=50, facecolor='C0', edgecolor='k', alpha=0.7,
                    zorder=1,
                    label=f'SCIENCE targets (SM3) ({len(x[isSci*isSm3])})'
                    )
        axe.scatter(x[isFst*isSm1], y[isFst*isSm1],
                    marker='o', s=50, facecolor='C1', edgecolor='k', alpha=0.7,
                    zorder=1,
                    label=f'FLXSTD targets ({len(x[isFst*isSm1])})'
                    )
        axe.scatter(x[isFst*isSm3], y[isFst*isSm3],
                    marker='^', s=50, facecolor='C1', edgecolor='k', alpha=0.7,
                    zorder=1,
                    label=f'FLXSTD targets ({len(x[isFst*isSm3])})'
                    )
        axe.scatter(x[isSky*isSm1], y[isSky*isSm1],
                    marker='o', s=50, facecolor='C2', edgecolor='k', alpha=0.7,
                    zorder=1,
                    label=f'SKY targets ({len(x[isSky*isSm1])})'
                    )
        axe.scatter(x[isSky*isSm3], y[isSky*isSm3],
                    marker='^', s=50, facecolor='C2', edgecolor='k', alpha=0.7,
                    zorder=1,
                    label=f'SKY targets ({len(x[isSky*isSm3])})'
                    )

        ''' plot bright Gaia sources '''
        axe.scatter(self.x_gaia_bright, self.y_gaia_bright,
                    marker='o', s=50, facecolor='none', edgecolor='red',
                    linewidth=2, alpha=0.7,
                    label=f'All Gaia sources ({self.gaia_gmag_min}<g<{self.gaia_gmag_max} mag)'
                    )

        ''' plot cobra patrol regions and dots '''
        p1 = PatchCollection(self.cobra_mpl_patches, facecolor='none',
                             edgecolor='gray', alpha=0.5)
        axe.add_collection(p1)
        p2 = PatchCollection(self.dot_mpl_patches, facecolor='gray',
                             edgecolor='none', alpha=0.5)
        axe.add_collection(p2)
        for i in range(len(self.cobra_mpl_id)):
            center = self.cobra_mpl_patches[i].center
            axe.text(center[0], center[1], f'{self.cobra_mpl_id[i]}',
                     color='gray', fontsize=5)

        axe.legend(loc='upper left', fontsize=8)

        # plt.savefig(f'./figures/test_{self.objId}_pfi.pdf', dpi=150, bbox_inches='tight')
        # plt.savefig(f'./figures/test_{self.objId}_pfi.png', dpi=150, bbox_inches='tight')

    def plot_mag_hist(self):
        isSm1 = is_smx(self.pfsDesign, moduleIds=[1])
        isSm3 = is_smx(self.pfsDesign, moduleIds=[3])
        isSm13 = isSm1 + isSm3
        isSci = (self.pfsDesign.targetType == TargetType.SCIENCE)
        isFst = (self.pfsDesign.targetType == TargetType.FLUXSTD)
        isSky = (self.pfsDesign.targetType == TargetType.SKY)
        isUna = (self.pfsDesign.targetType == TargetType.UNASSIGNED)
        try:
            isTgt = (self.pfsDesign.targetType == TargetType.SCIENCE) * \
                (self.pfsDesign.objId == int(self.objId))
        except ValueError:
            isTgt = np.full(len(self.pfsDesign), False)

        magnitude = np.zeros(len(self.pfsDesign))
        for i, flx in enumerate(self.pfsDesign.psfFlux):
            if len(flx) > 0:
                m = -2.5*np.log10(flx[0]*1e-09) + 8.9
            else:
                m = np.nan
            magnitude[i] = m
            # mag=np.append(mag, m)

        xmin = 11.0
        xmax = 18.0
        fig = plt.figure(figsize=(6, 6))
        axe = fig.add_subplot(111)
        axe.set_title(f'{self.objId}')
        axe.set_xlabel('Magnitude (AB)')
        axe.set_ylabel('Number ')
        axe.grid(color='gray', linestyle='dotted', linewidth=1)
        ''' plot SCIENCE & FLUXSTD & SKY '''
        if len(isTgt[isTgt]) > 0:
            axe.hist(magnitude[isTgt*isSm13], histtype='step', bins=20, range=(xmin, xmax),
                     color='red', ls='solid', lw=2, zorder=3, label='MAIN (SM1/SM3)')
        axe.hist(magnitude[isSci*isSm1], histtype='step', bins=20, range=(xmin, xmax),
                 color='C0', ls='solid', lw=2, zorder=1, label='SCIENCE (SM1)')
        axe.hist(magnitude[isSci*isSm3], histtype='step', bins=20, range=(xmin, xmax),
                 color='C0', ls='dashed', lw=2, zorder=1, label='SCIENCE (SM3)')
        axe.hist(magnitude[isFst*isSm1], histtype='step', bins=20, range=(xmin, xmax),
                 color='C1', ls='solid', lw=2, zorder=2, label='FLUXSTD (SM1)')
        axe.hist(magnitude[isFst*isSm3], histtype='step', bins=20, range=(xmin, xmax),
                 color='C1', ls='dashed', lw=2, zorder=2, label='FLUXSTD (SM3)')

        axe.legend(loc='upper left', fontsize=10)
