import unittest
from os.path import abspath, dirname, join, isfile, normpath, relpath
import os
import numpy as np
from numpy.testing import assert_allclose

from mhkit.tidal import resource, graphics, performance
from mhkit.dolfyn import load

testdir = dirname(abspath(__file__))
plotdir = join(testdir, 'plots')
isdir = os.path.isdir(plotdir)
if not isdir: os.mkdir(plotdir)
datadir = normpath(join(testdir,relpath('../../../examples/data/tidal')))


class TestResource(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        filename1 = join(datadir, 'adcp.enu.b1.20200815.nc')
        filename2 = join(datadir, 'adcp.principal.a1.20200815.nc') 
        self.ds_avg = load(filename1)
        self.ds = load(filename2)
        # Emulate power data
        self.power = self.ds_avg['U_mag'][10]**3 * 1e5

    @classmethod
    def tearDownClass(self):
        pass

    def test_power_curve(self,):
        df93_circ = performance.power_curve(
            power=self.power,
            velocity=self.ds['vel'].sel(dir='streamwise'),
            hub_height=4.2,
            doppler_cell_size=0.5, 
            sampling_frequency=1, 
            window_avg_time=600,
            turbine_profile='circular',
            diameter=3,
            height=None,
            width=None)
        test_circ = np.array([1.26250990e+00, 
                              1.09230978e+00, 
                              1.97168745e+05, 
                              1.04759930e+04, 
                              2.12352574e+05, 
                              1.79573443e+05])

        df93_rect = performance.power_curve(
            power=self.power,
            velocity=self.ds['vel'].sel(dir='streamwise'),
            hub_height=4.2,
            doppler_cell_size=0.5, 
            sampling_frequency=1, 
            window_avg_time=600,
            turbine_profile='rectangular',
            diameter=None,
            height=1,
            width=3)
        test_rect = np.array([1.15032239e+00, 
                              3.75747621e-01, 
                              1.79659574e+05, 
                              3.08413115e+04,
                              2.15625997e+05, 
                              1.32107771e+05])
        
        assert_allclose(df93_circ.values[-2], test_circ, atol=1e-5)
        assert_allclose(df93_rect.values[-3], test_rect, atol=1e-5)

    def test_velocity_profiles(self):
        df94 = performance.mean_velocity_profiles(
            velocity=self.ds['vel'].sel(dir='streamwise'), 
            hub_height=4.2,
            sampling_frequency=1, 
            window_avg_time=600)
        df95a = performance.rms_velocity_profiles(
            velocity=self.ds['vel'].sel(dir='streamwise'), 
            hub_height=4.2,
            sampling_frequency=1,
            window_avg_time=600)
        df95b = performance.std_velocity_profiles(
            velocity=self.ds['vel'].sel(dir='streamwise'), 
            hub_height=4.2, 
            sampling_frequency=1, 
            window_avg_time=600)
        
        test_df94 = np.array([0.32782955, 0.69326691, 1.00948623])
        test_df95a = np.array([0.3329345 , 0.69936798, 1.01762123])
        test_df95b = np.array([0.05635571, 0.08671777, 0.12735139])

        assert_allclose(df94.values[0], test_df94, atol=1e-5)
        assert_allclose(df95a.values[0], test_df95a, atol=1e-5)
        assert_allclose(df95b.values[0], test_df95b, atol=1e-5)
        

    def test_power_efficiency(self):
        df97 = performance.device_efficiency(
            self.power,
            velocity=self.ds['vel'].sel(dir='streamwise'),
            water_density=self.ds['water_density'],
            capture_area=np.pi*1.5**2,
            hub_height=4.2,
            sampling_frequency=1,
            window_avg_time=600)
        
        test_df97 = np.array(25.94345)
        assert_allclose(df97.values[-1], test_df97, atol=1e-5)


if __name__ == '__main__':
    unittest.main() 