#!/bin/sh

# REPO=/work/pfs/commissioning/fiber_allocation/ets_target_database/examples/commissioning_2022jun
# DIR="yamashita"
NAME="fcal_4247355045211154560_pa+00"
OBSTIME="2022-06-21T14:00:00Z"
RA=$(echo "301.434 + 0.15" | bc) # for PA=0
DEC=$(echo "4.047 - 0.10" | bc)
PA=0
EXPTIME=10

python ./subaru_fiber_allocation.py \
  --design_dir "design" \
  --cobra_coach_dir "coach" \
  --ra ${RA} \
  --dec ${DEC} \
  --pa ${PA} \
  --raster_scan \
  --exptime ${EXPTIME} \
  --fluxstd_min_prob_f_star 0.0 \
  --n_fluxstd 40 \
  --fluxstd_mag_min 12.0 \
  --fluxstd_mag_max 16.00 \
  --guidestar_mag_min 10.0 \
  --guidestar_mag_max 20.0 \
  --raster_mag_min 0.0 \
  --raster_mag_max 16.0 \
  --observation_time ${OBSTIME} \
  --design_name ${NAME} \
  --conf "../../../database_configs/config_pfsa-db01-gb_commissioning_2022may.toml" \
  --pfs_instdata_dir "/Users/monodera/Dropbox/NAOJ/PFS/Subaru-PFS/pfs_instdata/"
