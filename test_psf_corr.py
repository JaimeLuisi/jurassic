"""
Test that _psf_corr correctly scores injected PSF sources vs artifacts.

Injects into a noisy background:
  - PSF sources at various SNR (should score > 0.5)
  - Extended blobs (should score < 0.5)
  - Cosmic-ray-like streaks (should score < 0.5)
  - Hot pixels (should score < 0.5)

For each case with noise we repeat N_TRIALS times and report mean r.
"""
import sys, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

sys.path.insert(0, os.path.dirname(__file__))
os.environ.setdefault('WEBBPSF_PATH', '/Users/rridden/Documents/work/code/jwst/webbpsf-data')

from jurassic import _psf_fit

def _psf_corr_wrap(data_sub, x, y, kernel, a=None):
    r, x_fit, y_fit = _psf_fit(data_sub, x, y, kernel)
    if abs(x_fit - x) >= 0.9 or abs(y_fit - y) >= 0.9:
        return np.nan  # fit hit bound — not PSF-like
    # size filter: SEP a < 2 * sigma_psf
    if a is not None:
        sigma_exp = fwhm / (2 * np.sqrt(2 * np.log(2)))
        if a >= 2.0 * sigma_exp:
            return np.nan
    return r

_psf_corr = _psf_corr_wrap

try:
    import stpsf
except ModuleNotFoundError:
    import webbpsf as stpsf

# --- build kernel (F1500W, FWHM~4.4 px) ---
miri = stpsf.MIRI()
miri.filter = 'F1500W'
fwhm = 4.436
size = max(11, int(round(6 * fwhm)) | 1)
psf = miri.calc_psf(fov_pixels=size)
kernel = psf[3].data
kernel_norm = kernel / kernel.sum()
print(f'Kernel shape: {kernel.shape}  (fov_pixels={size})')

IMG = 256
BG_NOISE = 5.0
N_TRIALS = 20
THRESH = 0.5

rng = np.random.default_rng(0)

def blank_image():
    return rng.normal(0, BG_NOISE, (IMG, IMG))

def psf_peak_flux_for_snr(snr):
    """Flux that gives peak pixel SNR = snr."""
    return snr * BG_NOISE / kernel_norm.max()

def inject_psf(img, x, y, peak_snr):
    hs = kernel.shape[0] // 2
    y0, y1 = y - hs, y + hs + 1
    x0, x1 = x - hs, x + hs + 1
    if y0 < 0 or x0 < 0 or y1 > IMG or x1 > IMG:
        return
    img[y0:y1, x0:x1] += psf_peak_flux_for_snr(peak_snr) * kernel_norm

def inject_disk(img, x, y, radius):
    yy, xx = np.mgrid[:IMG, :IMG]
    disk = ((yy - y)**2 + (xx - x)**2) <= radius**2
    img[disk] += psf_peak_flux_for_snr(50) / disk.sum()

def inject_wide_gaussian(img, x, y, sigma):
    blob = np.zeros((IMG, IMG))
    blob[y, x] = 1.0
    blob = gaussian_filter(blob, sigma=sigma)
    img += psf_peak_flux_for_snr(50) * blob / blob.max()

def inject_streak(img, x, y, length):
    x0, x1 = max(0, x - length//2), min(IMG, x + length//2)
    img[y, x0:x1] += psf_peak_flux_for_snr(50) / (x1 - x0)

def mean_r(inject_fn, query_x=128, query_y=128, n=N_TRIALS, source_sigma=None):
    """source_sigma: known sigma of injected source (px); if None, skip size filter."""
    rs = []
    for _ in range(n):
        img = blank_image()
        inject_fn(img)
        r = _psf_corr_wrap(img, query_x, query_y, kernel, a=source_sigma)
        if r is not None and not np.isnan(r):
            rs.append(r)
    return np.mean(rs) if rs else np.nan

# ---- centroid offset test ----
# Inject PSF at known subpixel offsets; compare integer-rounded, fitted, and true position
print(f'\n{"True offset":>12}  {"SNR":>5}  {"Integer err":>12}  {"Fit err":>10}  {"Improvement":>12}')
print('-' * 60)
for true_offset in [0.1, 0.2, 0.3, 0.4, 0.5]:
    for snr in [10, 30, 100]:
        int_errs, fit_errs = [], []
        for _ in range(20):
            img = blank_image()
            # inject at 128 + true_offset (x), 128 (y)
            inject_psf(img, 128, 128, snr)  # inject at integer pixel
            # shift the image by true_offset to simulate subpixel centroid
            from scipy.ndimage import shift as ndshift
            img_shifted = ndshift(img, (0, true_offset), order=3, mode='reflect')
            # query at true position (128 + true_offset) — as SEP would report
            r, x_fit, y_fit = _psf_fit(img_shifted, 128 + true_offset, 128, kernel)
            int_err = abs(true_offset)                   # integer rounding always loses the offset
            fit_err = abs(x_fit - (128 + true_offset))  # fit error
            int_errs.append(int_err)
            fit_errs.append(fit_err)
        print(f'{true_offset:>12.1f}  {snr:>5d}  {np.mean(int_errs):>12.3f}  {np.mean(fit_errs):>10.3f}  {np.mean(int_errs)/np.mean(fit_errs):>12.1f}×')

# ---- SNR sweep (fine) ----
snr_vals = [2, 2.5, 3, 3.5, 4, 4.5, 5, 6, 7, 8, 10, 15, 20, 30, 50, 100]
snr_mean_r = []
sigma_psf = fwhm / (2 * np.sqrt(2 * np.log(2)))

for snr in snr_vals:
    r = mean_r(lambda img, s=snr: inject_psf(img, 128, 128, s), n=50,
               source_sigma=sigma_psf)
    snr_mean_r.append(r)

print(f'\n{"Peak SNR":>10}  {"mean r":>8}  {"Pass?":>6}')
print('-' * 30)
for snr, r in zip(snr_vals, snr_mean_r):
    passes = r >= THRESH
    print(f'{snr:>10.1f}  {r:>8.3f}  {"yes" if passes else "no":>6}')

# ---- cases ----
cases = []  # (type, description, mean_r, expected_pass)

# PSF sources at different peak SNRs
for snr in [3, 5, 10, 30, 100]:
    r = mean_r(lambda img, s=snr: inject_psf(img, 128, 128, s),
               source_sigma=sigma_psf)
    cases.append(('PSF', f'SNR={snr:3d}', r, snr >= 5))

# PSF at subpixel offsets (use high SNR to isolate offset effect from noise)
for offset in [0.3, 0.5, 0.8]:
    r = mean_r(lambda img: inject_psf(img, 128, 128, 30),
               query_x=128+offset, query_y=128, source_sigma=sigma_psf)
    cases.append(('PSF offset', f'{offset}px,SNR=30', r, True))

# Compact disk — radius close to PSF (edge case)
for r_px in [int(fwhm/2), int(fwhm), int(2*fwhm)]:
    disk_sigma = r_px / np.sqrt(2)  # rough sigma equivalent of disk radius
    r = mean_r(lambda img, rr=r_px: inject_disk(img, 128, 128, rr),
               source_sigma=disk_sigma)
    cases.append(('Disk', f'r={r_px}px ({r_px/fwhm*2:.1f}×FWHM)', r, False))

# Wide Gaussians
for sig_fac in [2, 3, 5]:
    sigma = sig_fac * sigma_psf
    r = mean_r(lambda img, s=sigma: inject_wide_gaussian(img, 128, 128, s),
               source_sigma=sigma)
    cases.append(('Wide Gauss', f'sig={sig_fac}×σ_PSF', r, False))

# Streaks
for length in [5, 15, 30]:
    r = mean_r(lambda img, l=length: inject_streak(img, 128, 128, l))
    cases.append(('Streak', f'len={length}px', r, False))

# Hot pixel
r = mean_r(lambda img: img.__setitem__((128, 128), img[128, 128] + psf_peak_flux_for_snr(50)))
cases.append(('Hot pixel', '1px', r, False))

# ---- report ----
print(f'\n{"Type":<16} {"Description":<24} {"mean r":>8}  {"Pass?":>6}  {"Correct?":>8}')
print('-' * 68)
n_correct = 0
for typ, desc, r, expected_pass in cases:
    passes = (not np.isnan(r) and r >= THRESH)
    correct = (passes == expected_pass)
    n_correct += correct
    mark = 'OK' if correct else 'FAIL'
    print(f'{typ:<16} {desc:<24} {r:>8.3f}  {"yes" if passes else "no":>6}  {mark:>8}')

print(f'\n{n_correct}/{len(cases)} cases correct  (threshold r >= {THRESH}, mean over {N_TRIALS} trials)')

# ---- diagnostic figure ----
fig, axes = plt.subplots(1, 5, figsize=(14, 3), constrained_layout=True)
hs = kernel.shape[0] // 2
slc = (slice(128-hs, 128+hs+1), slice(128-hs, 128+hs+1))

examples = [
    ('PSF\nSNR=30', lambda img: inject_psf(img, 128, 128, 30), 128, 128),
    ('PSF\nSNR=3',  lambda img: inject_psf(img, 128, 128, 3),  128, 128),
    (f'Disk\nr={int(fwhm)}px', lambda img: inject_disk(img, 128, 128, int(fwhm)), 128, 128),
    ('Wide Gauss\nsig=3×σ', lambda img: inject_wide_gaussian(img, 128, 128, 3*fwhm/2.355), 128, 128),
    ('Streak\nlen=15px', lambda img: inject_streak(img, 128, 128, 15), 128, 128),
]

rng2 = np.random.default_rng(7)
for ax, (label, fn, qx, qy) in zip(axes, examples):
    img = rng2.normal(0, BG_NOISE, (IMG, IMG))
    fn(img)
    r = _psf_corr(img, qx, qy, kernel)
    stamp = img[slc]
    vmax = np.percentile(np.abs(stamp), 99)
    ax.imshow(stamp, origin='lower', cmap='inferno', vmin=-vmax, vmax=vmax)
    col = 'lime' if (r is not None and r >= THRESH) else 'red'
    ax.set_title(f'{label}\nr = {r:.3f}', fontsize=8, color=col)
    ax.set_xticks([]); ax.set_yticks([])

out = os.path.join(os.path.dirname(__file__), 'test_psf_corr.png')
fig.savefig(out, dpi=130)
print(f'\nFigure → {out}')
os.system(f'open "{out}"')
