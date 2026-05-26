import matplotlib
matplotlib.use('Agg')
import numpy as np
import pandas as pd
import sys, unittest.mock

# Stub out unavailable heavy dependencies before importing jurassic
for mod in ('stpsf', 'sep', 'lacosmic', 'lacosmic.core', 'skimage', 'skimage.restoration'):
    sys.modules.setdefault(mod, unittest.mock.MagicMock())

from jurassic import Jurassic

# --- build a minimal Jurassic instance without running __init__ ---
j = object.__new__(Jurassic)
j.filename = 'test_observation'
j.fwhm = 2.445  # F770W FWHM in pixels

# Synthetic cube: 60 frames, 100x100 px, background noise
rng = np.random.default_rng(42)
n_frames, ny, nx = 60, 100, 100
cube = rng.normal(0, 5, (n_frames, ny, nx)).astype(np.float32)

# Inject a fake transient at (x=50, y=50) in frames 25-28
ty, tx = 50, 50
for frame in range(25, 29):
    yy, xx = np.ogrid[:ny, :nx]
    cube[frame] += 80 * np.exp(-((xx - tx)**2 + (yy - ty)**2) / (2 * 2**2))

j.clean_cube = cube

# MJD time array: cadence ~1.4 min per frame
mjd0 = 60000.0
cadence = 0.001  # days
frames = np.arange(n_frames)
mjds = mjd0 + frames * cadence
j.frame_mjd_df = pd.DataFrame({'frame': frames, 'mjd': mjds})

# Events dataframe: one object, one event spanning the transient frames
j.events = pd.DataFrame({
    'objid':  [1, 1, 1, 1],
    'event':  [0, 0, 0, 0],
    'x':      [tx, tx, tx, tx],
    'y':      [ty, ty, ty, ty],
    'frame':  [25, 26, 27, 28],
    'sep_flux': [80., 95., 70., 40.],
    'mjd':    [mjd0 + f * cadence for f in [25, 26, 27, 28]],
})

# Run and save
save_dir = 'test_detection_figures'
j.plot_detection(save_dir, latex=False)
print(f'Figures saved to {save_dir}/')
