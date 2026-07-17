"""
test_trappist_lc_compare.py

Reproduces exactly what Jurassic.ramp_correction() does (RampDoctor with the
full-frame science mask passed in as bg_mask, no BFE fit — see jurassic.py
lines 485-503) on the TRAPPIST-1 seg002 ramp, and compares the resulting
aperture lightcurve to raw and to the properly-fit rampdoctor correction
(M/threshold fit from data, bg_mask = a real background annulus).
"""
import sys, os
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from astropy.io import fits

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, '/Users/rridden/Documents/work/code/jwst/ramps/rampdoctor')

from rampdoctor import RampDoctor
from rampdoctor.ramp_correction import correct_bfe_rcd, fit_migration_params

RAMP = ('/Users/rridden/Documents/work/code/jwst/ramps/trappist/'
        'jw01177007001_03101_00001-seg002_mirimage_ramp.fits')
MASK_PATH = os.path.join(os.path.dirname(__file__), 'full_MIRI_mask.npy')
OUT_DIR = os.path.dirname(__file__)

AP_RADIUS = 5
BG_INNER = 20
BG_OUTER = 60

with fits.open(RAMP) as hdul:
    cube = hdul[1].data.astype(float)
n_int, n_groups, ny, nx = cube.shape
print(f'Loaded {os.path.basename(RAMP)}: {cube.shape}')

sci_mask = np.load(MASK_PATH)
print(f'Mask: {sci_mask.sum()} science pixels / {sci_mask.size} total')

grads_raw = np.diff(cube, axis=1)

# ---------------------------------------------------------------------------
# 1) Jurassic-style correction: exactly as in jurassic.py ramp_correction()
#    bg_mask = full science mask (includes the star + its wings), no BFE fit,
#    fixed default M/threshold, no star position passed through.
# ---------------------------------------------------------------------------
print('\n--- Jurassic-style correction (bg_mask=full sci mask, no BFE fit) ---')
rd = RampDoctor(cube=cube, bg_mask=sci_mask, verbose=True)
cube_cor_jurassic = rd.correct(diagnostics=True,
                                save_path=os.path.join(OUT_DIR, 'compare_jurassic_diag.png'))
grads_cor_jurassic = np.diff(cube_cor_jurassic, axis=1)

# ---------------------------------------------------------------------------
# 2) Correct rampdoctor usage: fit M/threshold from the data, real background
#    annulus for bg_mask, star position passed through.
# ---------------------------------------------------------------------------
print('\n--- Correct rampdoctor usage (fitted M/threshold, real bg annulus) ---')
Mx_fit, My_fit, thr_fit, sx, sy = fit_migration_params(
    cube, sci_mask=sci_mask, verbose=True)
print(f'Star detected at x={sx}, y={sy}')

yy, xx = np.mgrid[:ny, :nx]
r_star = np.sqrt((yy - sy) ** 2 + (xx - sx) ** 2)
ap_mask = r_star <= AP_RADIUS
bg_mask_good = (r_star >= BG_INNER) & (r_star <= BG_OUTER) & sci_mask

cube_cor_good = correct_bfe_rcd(
    cube, M_mig=Mx_fit, thr_mig=thr_fit, bg_mask=bg_mask_good, sci_mask=sci_mask,
    fit_bfe=False, star_x=sx, star_y=sy, verbose=True,
    diagnostics=True, save_path=os.path.join(OUT_DIR, 'compare_good_diag.png'))
grads_cor_good = np.diff(cube_cor_good, axis=1)

# ---------------------------------------------------------------------------
# Aperture lightcurves — all three
# ---------------------------------------------------------------------------
g_good = np.arange(1, n_groups - 2)

def lc_of(grads):
    lc = grads[:, g_good][:, :, ap_mask].sum(axis=2)
    return lc / np.median(lc)

lc_raw_n = lc_of(grads_raw)
lc_jur_n = lc_of(grads_cor_jurassic)
lc_good_n = lc_of(grads_cor_good)

rms_raw = np.std(lc_raw_n) * 100
rms_jur = np.std(lc_jur_n) * 100
rms_good = np.std(lc_good_n) * 100

print('\nAperture LC RMS:')
print(f'  Raw                    : {rms_raw:.3f}%')
print(f'  Jurassic-style correct : {rms_jur:.3f}%')
print(f'  Correct rampdoctor use : {rms_good:.3f}%')

integ = np.arange(n_int)
cmap_g = plt.colormaps['plasma'].resampled(n_groups)
colors = [cmap_g(g) for g in range(n_groups)]

fig, axes = plt.subplots(1, 3, figsize=(19, 5), sharey=True)
for ax, lc_n, title, rms in [
        (axes[0], lc_raw_n, 'Raw', rms_raw),
        (axes[1], lc_jur_n, 'Jurassic ramp_correction()', rms_jur),
        (axes[2], lc_good_n, 'Correct rampdoctor usage', rms_good),
]:
    for i, g in enumerate(g_good):
        ax.scatter(integ, lc_n[:, i], color=colors[g], s=6, alpha=0.7, zorder=g + 1)
    ax.axhline(1.0, color='k', lw=0.8, ls='--', alpha=0.4)
    ax.set_xlabel('Integration index')
    ax.set_title(f'{title}\n(RMS={rms:.3f}%)', fontsize=10)
axes[0].set_ylabel('Normalised aperture flux')

sm = plt.cm.ScalarMappable(cmap=cmap_g, norm=plt.Normalize(vmin=0, vmax=n_groups - 1))
sm.set_array([])
cbar = fig.colorbar(sm, ax=axes[-1], pad=0.01)
cbar.set_label('Group index')
cbar.set_ticks(np.arange(n_groups))

fig.suptitle('TRAPPIST-1 seg002: raw vs jurassic-invoked vs correctly-fit rampdoctor correction',
             fontsize=11, fontweight='bold')
fig.tight_layout()
out = os.path.join(OUT_DIR, 'trappist_lc_jurassic_vs_correct.png')
fig.savefig(out, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f'\nSaved {out}')
