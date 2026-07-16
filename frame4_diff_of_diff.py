import warnings
warnings.filterwarnings('ignore')
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits

sys.path.insert(0, '/Users/rridden/Documents/work/code/jwst/ramps/rampdoctor')
from rampdoctor.ramp_correction import correct_bfe_rcd, fit_migration_params

RAMP = ('/Users/rridden/Documents/work/code/jwst/ramps/trappist/'
        'jw01177007001_03101_00001-seg002_mirimage_ramp.fits')
MASK_PATH = '/Users/rridden/Documents/work/code/jwst/jurassic/full_MIRI_mask.npy'
OUT_DIR = '/Users/rridden/Documents/work/code/jwst/jurassic'
G_IDX = 4
INT_IDX = 0
STAR_X, STAR_Y = 697, 515

with fits.open(RAMP) as hdul:
    cube = hdul[1].data.astype(float)
ny, nx = cube.shape[2], cube.shape[3]
sci_mask = np.load(MASK_PATH)

grads_raw = np.diff(cube, axis=1)
ref_raw = np.median(grads_raw[:, G_IDX], axis=0)
frame_raw = grads_raw[INT_IDX, G_IDX] - ref_raw
del grads_raw

yy, xx = np.mgrid[:ny, :nx]
r_star = np.sqrt((yy - STAR_Y) ** 2 + (xx - STAR_X) ** 2)
bg_mask = (r_star >= 20) & (r_star <= 60) & sci_mask

Mx_fit, My_fit, thr_fit, sx, sy = fit_migration_params(cube, sci_mask=sci_mask, verbose=True)

cube_cor = correct_bfe_rcd(cube, M_mig=Mx_fit, thr_mig=thr_fit, bg_mask=bg_mask,
                            sci_mask=sci_mask, fit_bfe=False, star_x=sx, star_y=sy, verbose=True)
del cube

grads_cor = np.diff(cube_cor, axis=1)
ref_cor = np.median(grads_cor[:, G_IDX], axis=0)
frame_cor = grads_cor[INT_IDX, G_IDX] - ref_cor
del grads_cor, cube_cor

diff = frame_cor - frame_raw
print('max abs diff:', np.nanmax(np.abs(diff)))
print('std diff:', np.nanstd(diff))
print('are they exactly equal?', np.allclose(frame_raw, frame_cor, equal_nan=True))

s = np.nanstd(diff)
cut = 40
fig, axes = plt.subplots(1, 2, figsize=(15, 7))
im0 = axes[0].imshow(diff, origin='lower', cmap='RdBu_r', vmin=-3*s if s > 0 else -1, vmax=3*s if s > 0 else 1)
axes[0].set_title(f'Full frame: (corrected - raw) ref-subtracted diff\nstd={s:.4f} DN/group')
fig.colorbar(im0, ax=axes[0], label='DN/group')

zoom = diff[STAR_Y-cut:STAR_Y+cut, STAR_X-cut:STAR_X+cut]
im1 = axes[1].imshow(zoom, origin='lower', cmap='RdBu_r',
                      vmin=-np.nanpercentile(np.abs(zoom), 99), vmax=np.nanpercentile(np.abs(zoom), 99),
                      extent=[-cut, cut, -cut, cut])
axes[1].set_title(f'Zoom on star (r={cut}px)\nstd={np.nanstd(zoom):.4f} DN/group')
fig.colorbar(im1, ax=axes[1], label='DN/group')

fig.suptitle('Does rampdoctor introduce new signal into the ref-subtracted (integration-vs-median) frame?',
             fontsize=11, fontweight='bold')
fig.tight_layout()
out = f'{OUT_DIR}/frame4_diff_of_diff.png'
fig.savefig(out, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f'Saved {out}')
