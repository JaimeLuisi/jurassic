"""
Test jurassic + rampdoctor on a single Wolf 359 ramp file.
"""
import sys, os

os.environ.setdefault('WEBBPSF_PATH', '/Users/rridden/Documents/work/code/jwst/webbpsf-data')

try:
    import stpsf
except ModuleNotFoundError:
    import webbpsf
    sys.modules['stpsf'] = webbpsf

sys.path.insert(0, os.path.dirname(__file__))

import jurassic as jmod

RAMP = ('/Users/rridden/Documents/work/code/jwst/ramps/wolf-359/ramp-fits/'
        'jw06122002001_02101_00001_mirimage_ramp.fits')
OUT_DIR = os.path.join(os.path.dirname(__file__), 'test_wolf359')

print(f'Input:  {RAMP}')
print(f'Output: {OUT_DIR}')

j = jmod.Jurassic(file=RAMP, method='mega', num_cores=8,
                  base_dir=OUT_DIR,
                  data_dir=os.path.dirname(RAMP))
print('Done.')
