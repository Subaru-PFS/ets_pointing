import os
import sys
import numpy as np


repoDir = '/work/pfs/commissioning/2022sep/fiber_allocation'

def test():
    return np.array([1,2,3,4,5])
    
''' isSM1 utils '''
gfmFilename = os.path.join(repoDir, 
                           'pfs_utils/data/fiberids',
                           'grandfibermap.20210314.txt')

gfmCob = {}
gfmSp = {}
gfmFh = {}
gfmX = {}
gfmY = {}

with open(gfmFilename, 'r') as file:
    for line in file:
        if '\cob' not in line:
            a = line.split()
            fid = int(a[15])
            sfib = a[14]
            gfmSp[fid] = int(a[12])
            if sfib != 'eng' and sfib != 'emp':
                gfmCob[fid] = int(a[0])
                gfmFh[fid] = int(a[13])
                gfmX[fid] = float(a[9])
                gfmY[fid] = float(a[10])               
            else:
                gfmCob[fid] = np.nan
                gfmFh[fid] = np.nan
                gfmX[fid] = np.nan
                gfmY[fid] = np.nan
                
def is_sm1(pfsDesign):
    isSm1 = np.empty(len(pfsDesign.fiberId), dtype='bool')
    for i, fid in enumerate(pfsDesign.fiberId):
        if gfmSp[fid]==1:
            isSm1[i] = True
        else:
            isSm1[i] = False
    return isSm1    

''' coordinate transformation utils '''

sys.path.append(os.path.join(repoDir, 'pfs_utils'))
from python.pfs.utils.coordinates.CoordTransp import CoordinateTransform as ctrans
from astropy.time import Time
from astropy.utils import iers
iers.conf.auto_download = True

def pfi2sky(pfsDesign, observation_time):
    pmra = np.array([0.0 for _ in range(len(pfsDesign.ra))])
    pmdec = np.array([0.0 for _ in range(len(pfsDesign.ra))])
    parallax = np.array([1.0e-07 for _ in range(len(pfsDesign.ra))])
    epoch = 2015.5

    tmp = np.array([pfsDesign.pfiNominal[:,0], pfsDesign.pfiNominal[:,1]])
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

    sky_x = tmp[0,:]
    sky_y = tmp[1,:]
        
    return sky_x, sky_y

def pfi2sky_array(pfi_x, pfi_y, pfsDesign, observation_time):
    pmra = np.array([0.0 for _ in range(len(pfi_x))])
    pmdec = np.array([0.0 for _ in range(len(pfi_x))])
    parallax = np.array([1.0e-07 for _ in range(len(pfi_x))])
    epoch = 2015.5

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

    sky_x = tmp[0,:]
    sky_y = tmp[1,:]
        
    return sky_x, sky_y

def sky2pfi(pfsDesign, observation_time):
    pmra = np.array([0.0 for _ in range(len(pfsDesign.ra))])
    pmdec = np.array([0.0 for _ in range(len(pfsDesign.ra))])
    parallax = np.array([1.0e-07 for _ in range(len(pfsDesign.ra))])
    epoch = 2015.5

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

    pfi_x = tmp[0,:]
    pfi_y = tmp[1,:]
    
    return pfi_x, pfi_y

def sky2pfi_array(sky_x, sky_y, pfsDesign, observation_time):
    pmra = np.array([0.0 for _ in range(len(sky_x))])
    pmdec = np.array([0.0 for _ in range(len(sky_x))])
    parallax = np.array([1.0e-07 for _ in range(len(sky_x))])
    epoch = 2015.5

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

    pfi_x = tmp[0,:]
    pfi_y = tmp[1,:]
    
    return pfi_x, pfi_y

''' get number of targets in the patrol region '''

def get_num_targets_in_patrol_region(bench, pfsDesign, gaia_info, cobra_ids_use):
    ''' get all gaia sources '''
    gaia_all_id = gaia_info[0]
    gaia_all_x = gaia_info[1]
    gaia_all_y = gaia_info[2]
    
    ''' get assigned gaia sources '''
    assigned_id = np.array(pfsDesign.objId, dtype='int64')
    assigned_x = np.array(pfsDesign.pfiNominal[:,0])
    assigned_y = np.array(pfsDesign.pfiNominal[:,1])
    
    gaia_assigned_in_fov_id = []
    gaia_assigned_in_fov_x = []
    gaia_assigned_in_fov_y = []
    for i, x, y in zip(assigned_id, assigned_x, assigned_y):
        msk = i in gaia_all_id
        if len(assigned_id[msk])==1:
            gaia_assigned_in_fov_id.append(assigned_id[msk][0])
            gaia_assigned_in_fov_x.append(assigned_x[msk][0])
            gaia_assigned_in_fov_y.append(assigned_y[msk][0])
    gaia_assigned_in_fov_id = np.array(gaia_assigned_in_fov_id)
    gaia_assigned_in_fov_x = np.array(gaia_assigned_in_fov_x)
    gaia_assigned_in_fov_y = np.array(gaia_assigned_in_fov_y)
    

    ''' get assigned gaia sources in patrol area of interested SMs '''
    flg = is_sm1(pfsDesign)
    assigned_id = assigned_id[flg]
    assigned_x = assigned_x[flg]
    assigned_y = assigned_y[flg]

    gaia_assigned_in_sms_id = []
    gaia_assigned_in_sms_x = []
    gaia_assigned_in_sms_y = []
    for i, x, y in zip(assigned_id, assigned_x, assigned_y):
        msk = i in gaia_all_id
        if len(assigned_id[msk])==1:
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
    for cobra_id,center in enumerate(bench.cobras.centers):
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
    for cobra_id,center in enumerate(bench.cobras.centers):
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
   
    return num_gaia_all, num_gaia_all_in_fov, num_gaia_assigned_in_fov, num_gaia_all_in_sms, num_gaia_assigned_in_sms