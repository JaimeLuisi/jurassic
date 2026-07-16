import sys, os
import warnings
warnings.filterwarnings('ignore')
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import jurassic as jmod

RAMP = ('/Users/rridden/Documents/work/code/jwst/ramps/trappist/'
        'jw01177007001_03101_00001-seg002_mirimage_ramp.fits')
OUT_DIR = os.path.dirname(__file__)
STAR_X, STAR_Y = 697, 515
AP_RADIUS = 5

j = jmod.Jurassic(file=RAMP, run=False, base_dir=os.path.join(OUT_DIR, 'test_jur_lc'),
                   data_dir=os.path.dirname(RAMP), correct_ramps=False)
j._assign_data()
j.data_cor = j.data
mask_path = os.path.join(os.path.dirname(__file__), 'full_MIRI_mask.npy')
full_mask = np.load(mask_path)
ny, nx = j.data.shape[2], j.data.shape[3]
j.gen_mask = full_mask

j.flux_calibrate(cube=j.data_cor)
print('flux_data shape:', j.flux_data.shape)

yy, xx = np.mgrid[:ny, :nx]
r_star = np.sqrt((yy - STAR_Y) ** 2 + (xx - STAR_X) ** 2)
ap_mask = r_star <= AP_RADIUS

# flux_data is (n_int, n_group, ny, nx), rate units
lc_ap_by_group = j.flux_data[:, :, ap_mask].sum(axis=2)   # (n_int, n_group)
print('\nAperture summed rate per group (median over integrations):')
for g in range(j.n_group):
    col = lc_ap_by_group[:, g]
    print(f'  group {g:2d}: median={np.median(col):12.4f}  std={np.std(col):10.4f}')

grad = np.diff(lc_ap_by_group, axis=1)  # (n_int, n_group-1)
print('\nGroup-to-group gradient of rate (median over integrations):')
for g in range(grad.shape[1]):
    col = grad[:, g]
    print(f'  grad {g:2d}: median={np.median(col):12.4f}  std={np.std(col):10.4f}')
