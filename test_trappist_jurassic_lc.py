"""
test_trappist_jurassic_lc.py

Runs the actual Jurassic reduction steps (ramp_correction -> flux_calibrate ->
_make_cubes) on the TRAPPIST-1 seg002 ramp, for both corrected and
uncorrected cases, then extracts a stellar aperture lightcurve directly from
self.rampy_cube (the flux-calibrated, flattened frame-stream cube jurassic
itself uses for its transient search) to see whether the BFE+RCD correction
still improves the lightcurve once it goes through jurassic's own data
structures, rather than the standalone rampdoctor call.
"""
import sys, os
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
import jurassic as jmod

RAMP = ('/Users/rridden/Documents/work/code/jwst/ramps/trappist/'
        'jw01177007001_03101_00001-seg002_mirimage_ramp.fits')
OUT_DIR = os.path.dirname(__file__)

STAR_X, STAR_Y = 697, 515   # from fit_migration_params on this segment
AP_RADIUS = 5


def build_jurassic(correct_ramps):
    j = jmod.Jurassic(file=RAMP, run=False, base_dir=os.path.join(OUT_DIR, 'test_jur_lc'),
                       data_dir=os.path.dirname(RAMP), correct_ramps=correct_ramps)
    j._assign_data()
    if j.correct_ramps:
        j.ramp_correction(cube=j.data)
    else:
        j.data_cor = j.data
        mask_path = os.path.join(os.path.dirname(__file__), 'full_MIRI_mask.npy')
        full_mask = np.load(mask_path)
        ny, nx = j.data.shape[2], j.data.shape[3]
        j.gen_mask = full_mask if full_mask.shape == (ny, nx) else full_mask[
            j.substrt2 - 1:j.substrt2 - 1 + ny, j.substrt1 - 1:j.substrt1 - 1 + nx]
    j.flux_calibrate(cube=j.data_cor)
    j._make_cubes()
    return j


print('=== Running jurassic reduction WITHOUT correction ===')
j_raw = build_jurassic(correct_ramps=False)
print(f'rampy_cube shape: {j_raw.rampy_cube.shape}  n_int={j_raw.n_int}  n_group={j_raw.n_group}')

print('\n=== Running jurassic reduction WITH correction (ramp_correction) ===')
j_cor = build_jurassic(correct_ramps=True)
print(f'rampy_cube shape: {j_cor.rampy_cube.shape}')

n_int, n_group = j_raw.n_int, j_raw.n_group
ny, nx = j_raw.rampy_cube.shape[1:]

yy, xx = np.mgrid[:ny, :nx]
r_star = np.sqrt((yy - STAR_Y) ** 2 + (xx - STAR_X) ** 2)
ap_mask = r_star <= AP_RADIUS
print(f'\nAperture pixels: {ap_mask.sum()}')

# rampy_cube is already a per-group rate cube (flux_calibrate differences
# consecutive groups internally); reshape the flattened frame-stream
# (n_int*n_group, ny, nx) back to (n_int, n_group, ny, nx). Group index 0 is
# NaN (no preceding group to difference against).
def reshape(rampy_cube):
    return rampy_cube.reshape(n_int, n_group, ny, nx)

grads_raw = reshape(j_raw.rampy_cube)
grads_cor = reshape(j_cor.rampy_cube)

# exclude group 0 (NaN) and the last-frame anomaly (group n_group-1)
g_good = np.arange(1, n_group - 1)

def lc_of(grads):
    lc = grads[:, g_good][:, :, ap_mask].sum(axis=2)
    return lc / np.median(lc)

lc_raw_n = lc_of(grads_raw)
lc_cor_n = lc_of(grads_cor)

rms_raw = np.std(lc_raw_n) * 100
rms_cor = np.std(lc_cor_n) * 100
print('\nAperture LC RMS (via jurassic rampy_cube):')
print(f'  Raw (correct_ramps=False)  : {rms_raw:.3f}%')
print(f'  Corrected (ramp_correction): {rms_cor:.3f}%')

integ = np.arange(n_int)
cmap_g = plt.colormaps['plasma'].resampled(n_group)
colors = [cmap_g(g) for g in range(n_group)]

fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
for ax, lc_n, title, rms in [
        (axes[0], lc_raw_n, 'Raw (jurassic, no correction)', rms_raw),
        (axes[1], lc_cor_n, 'Jurassic ramp_correction()', rms_cor),
]:
    for i, g in enumerate(g_good):
        ax.scatter(integ, lc_n[:, i], color=colors[g], s=6, alpha=0.7, zorder=g + 1)
    ax.axhline(1.0, color='k', lw=0.8, ls='--', alpha=0.4)
    ax.set_xlabel('Integration index')
    ax.set_title(f'{title}\n(RMS={rms:.3f}%)', fontsize=10)
axes[0].set_ylabel('Normalised aperture flux')

sm = plt.cm.ScalarMappable(cmap=cmap_g, norm=plt.Normalize(vmin=0, vmax=n_group - 1))
sm.set_array([])
cbar = fig.colorbar(sm, ax=axes[-1], pad=0.01)
cbar.set_label('Group index')
cbar.set_ticks(np.arange(n_group))

fig.suptitle('TRAPPIST-1 seg002: lightcurve from jurassic rampy_cube, raw vs corrected',
             fontsize=11, fontweight='bold')
fig.tight_layout()
out = os.path.join(OUT_DIR, 'trappist_lc_via_jurassic_pipeline.png')
fig.savefig(out, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f'\nSaved {out}')
