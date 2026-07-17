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

with fits.open(RAMP) as hdul:
    cube = hdul[1].data.astype(float)
ny, nx = cube.shape[2], cube.shape[3]
sci_mask = np.load(MASK_PATH)

grads_raw = np.diff(cube, axis=1)
ref_raw = np.median(grads_raw[:, G_IDX], axis=0)
frame_raw = grads_raw[INT_IDX, G_IDX] - ref_raw
del grads_raw

print('Fitting BFE parameters ...')
Mx_fit, My_fit, thr_fit, sx, sy = fit_migration_params(cube, sci_mask=sci_mask, verbose=True)
yy, xx = np.mgrid[:ny, :nx]
r_star = np.sqrt((yy - sy) ** 2 + (xx - sx) ** 2)
bg_mask = (r_star >= 20) & (r_star <= 60) & sci_mask

print('Running correct_bfe_rcd ...')
cube_cor = correct_bfe_rcd(cube, M_mig=Mx_fit, thr_mig=thr_fit, bg_mask=bg_mask,
                            sci_mask=sci_mask, fit_bfe=False, star_x=sx, star_y=sy, verbose=True)
del cube

grads_cor = np.diff(cube_cor, axis=1)
ref_cor = np.median(grads_cor[:, G_IDX], axis=0)
frame_cor = grads_cor[INT_IDX, G_IDX] - ref_cor
del grads_cor, cube_cor

s = np.nanstd(frame_raw)

fig, axes = plt.subplots(1, 2, figsize=(15, 7), sharey=True)
for ax, frame, title in [(axes[0], frame_raw, 'No rampdoctor correction'),
                          (axes[1], frame_cor, 'BFE + RCD corrected')]:
    im = ax.imshow(frame, origin='lower', cmap='RdBu_r', vmin=-3 * s, vmax=3 * s)
    ax.set_title(f'{title}\n(std={np.nanstd(frame):.2f} DN/group)', fontsize=10)
    ax.set_xlabel('x (px)')
axes[0].set_ylabel('y (px)')
fig.colorbar(im, ax=axes, label='DN/group (reference-subtracted)', pad=0.02)

fig.suptitle(f'TRAPPIST-1 seg002, integration {INT_IDX}, gradient index {G_IDX}, minus median reference',
             fontsize=11, fontweight='bold')
out = f'{OUT_DIR}/frame4_raw_vs_cor.png'
fig.savefig(out, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f'Saved {out}')
print(f'std raw={np.nanstd(frame_raw):.3f}  std corrected={np.nanstd(frame_cor):.3f}')
