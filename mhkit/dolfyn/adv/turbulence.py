import warnings
import numpy as np
import xarray as xr
from scipy.special import cbrt

from ..velocity import VelBinner
from ...utils.binning_tools.tools import slice1d_along_axis, _nans_like


class ADVBinner(VelBinner):
    def __init__(
        self,
        n_bin,
        fs,
        n_fft=None,
        n_fft_coh=None,
        noise=None,
    ):
        """
        A class that builds upon `VelBinner` for calculating turbulence
        statistics and velocity spectra from ADV data.

        Parameters
        ----------
        n_bin : int
          The length of each bin, in number of points, for this averaging
          operator.
        fs : int
          Instrument sampling frequency in Hz
        n_fft : int
          The length of the FFT for computing spectra (must be <= `n_bin`).
          Default = `n_fft` = `n_bin`
        n_fft_coh : int
          Number of data points to use for coherence and cross-spectra FFT's.
          Default = `n_fft_coh` = `n_fft`
        noise : float or array-like
          Instrument noise level in same units as velocity. Typically found from
          :func:`doppler_noise_level <mhkit.dolfyn.adv.turbulence.ADVBinner.doppler_noise_level>`.
          Default = None
        """

        VelBinner.__init__(self, n_bin, fs, n_fft, n_fft_coh, noise)

    def reynolds_stress(self, veldat, detrend=True):
        """
        Calculate the specific Reynolds shear stresses (:math:`\\overline{u'v'}`,
        :math:`\\overline{u'w'}`, :math:`\\overline{v'w'}`).

        Parameters
        ----------
        veldat : xr.DataArray
          A velocity data array. The last dimension is assumed to be time.
        detrend : bool
          Detrend the velocity data (True), or simply de-mean it
          (False), prior to computing stress. Note: the psd routines
          use detrend, so if you want to have the same amount of
          variance here as there use `detrend=True`.
          Default = True

        Returns
        -------
        out : xarray.DataArray
        """

        if not isinstance(veldat, xr.DataArray):
            raise TypeError("`veldat` must be an instance of `xarray.DataArray`.")

        time = self.mean(veldat["time"].values)
        vel = veldat.values

        out = np.empty(self._outshape(vel[:3].shape)[:-1])

        if detrend:
            vel = self.detrend(vel)
        else:
            vel = self.demean(vel)

        for idx, p in enumerate(self._cross_pairs):
            out[idx] = np.nanmean(vel[p[0]] * vel[p[1]], -1)

        da = xr.DataArray(
            out,
            dims=veldat.dims,
            attrs={"units": "m2 s-2", "long_name": "Specific Reynolds Stress Vector"},
        )

        tau = xr.DataArray(
            ["upvp_", "upwp_", "vpwp_"],
            dims=["tau"],
            name="tau",
            attrs={
                "units": "1",
                "long_name": "Reynolds Stress Vector Components",
                "coverage_content_type": "coordinate",
            },
        )
        da = da.rename({"dir": "tau"})
        da = da.assign_coords({"tau": tau, "time": time})

        return da

    def dissipation_rate_SF(self, vel_raw, U_mag, fs=None, freq_range=[2.0, 4.0]):
        """
        Calculate dissipation rate using the "structure function" (SF) method

        Parameters
        ----------
        vel_raw : xarray.DataArray (time)
          The raw velocity data upon which to perform the SF technique.
        U_mag : xarray.DataArray
          The bin-averaged horizontal velocity (i.e., computed using
          :func:`U_mag <mhkit.dolfyn.velocity.Velocity.U_mag>`)
        fs : float
          The sample rate of `vel_raw` in Hz
        freq_range : iterable(2)
          The frequency range over which to compute the SF in Hz
          (i.e. the frequency range within which the isotropic
          turbulence cascade falls).
          Default = [2., 4.] Hz

        Returns
        -------
        epsilon : xarray.DataArray
          dataArray of the dissipation rate
        """

        if not isinstance(vel_raw, xr.DataArray):
            raise TypeError("`vel_raw` must be an instance of `xarray.DataArray`.")
        if len(vel_raw["time"]) == len(U_mag["time"]):
            raise Exception("`U_mag` should be from ensembled-averaged dataset")
        if not hasattr(freq_range, "__iter__") or len(freq_range) != 2:
            raise ValueError("`freq_range` must be an iterable of length 2.")

        veldat = vel_raw.values
        if len(veldat.shape) > 1:
            raise Exception("Function input should be a 1D velocity vector")

        fs = self._parse_fs(fs)
        if freq_range[1] > fs:
            warnings.warn("Max freq_range cannot be greater than fs")

        dt = self.reshape(veldat)
        out = np.empty(dt.shape[:-1], dtype=dt.dtype)
        for slc in slice1d_along_axis(dt.shape, -1):
            up = dt[slc]
            lag = U_mag.values[slc[:-1]] / fs * np.arange(up.shape[0])
            DAA = _nans_like(lag)
            for L in range(int(fs / freq_range[1]), int(fs / freq_range[0])):
                DAA[L] = np.nanmean((up[L:] - up[:-L]) ** 2)
            cv2 = DAA / (lag ** (2 / 3))
            cv2m = np.median(cv2[np.logical_not(np.isnan(cv2))])
            out[slc[:-1]] = (cv2m / 2.1) ** (3 / 2)

        return xr.DataArray(
            out,
            coords=U_mag.coords,
            dims=U_mag.dims,
            attrs={
                "units": "m2 s-3",
                "long_name": "TKE Dissipation Rate",
                "standard_name": "specific_turbulent_kinetic_energy_dissipation_in_sea_water",
                "description": "TKE dissipation rate calculated using the "
                '"structure function" method',
            },
        )

    def _up_angle(self, U_complex):
        """
        Calculate the angle of the turbulence fluctuations.

        Parameters
        ----------
        U_complex  : numpy.ndarray (..., n_time * n_bin)
          The complex, raw horizontal velocity (non-binned)

        Returns
        -------
        theta : numpy.ndarray (..., n_time)
          The angle of the turbulence [rad]
        """

        dt = self.demean(U_complex)
        fx = dt.imag <= 0
        dt[fx] = dt[fx] * np.exp(1j * np.pi)

        return np.angle(np.mean(dt, -1, dtype=np.complex128))

    def _integral_TE01(self, I_tke, theta):
        """
        The integral, equation A13, in [TE01].

        Parameters
        ----------
        I_tke : numpy.ndarray
          (beta in TE01) is the turbulence intensity ratio:
          \\sigma_u / V
        theta : numpy.ndarray
          is the angle between the mean flow and the primary axis of
          velocity fluctuations
        """

        x = np.arange(-20, 20, 1e-2)  # I think this is a long enough range.
        out = np.empty_like(I_tke.flatten())
        for i, (b, t) in enumerate(zip(I_tke.flatten(), theta.flatten())):
            out[i] = np.trapezoid(
                cbrt(x**2 - 2 / b * np.cos(t) * x + b ** (-2)) * np.exp(-0.5 * x**2),
                x,
            )

        return out.reshape(I_tke.shape) * (2 * np.pi) ** (-0.5) * I_tke ** (2 / 3)

    def dissipation_rate_TE01(self, dat_raw, dat_avg, freq_range=[6.28, 12.57]):
        """
        Calculate the dissipation rate according to TE01.

        Parameters
        ----------
        dat_raw : xarray.Dataset
          The raw (off the instrument) adv dataset
        dat_avg : xarray.Dataset
          The bin-averaged adv dataset (calculated from `ADVBinner.calc_turbulence` or
          `VelBinner.bin_average`). The spectra (PSD) and Reynolds stresses
          (`tke_vec` and `stress_vec`) must already be computed.
        freq_range : iterable(2)
          The range over which to integrate/average the spectrum, in units
          of the psd frequency vector (Hz or rad/s).
          Default = [6.28, 12.57] rad/s

        Notes
        -----
        TE01 : Trowbridge, J and Elgar, S, "Turbulence measurements in
        the Surf Zone". JPO, 2001, vol31, pp2403-2417.
        """

        if not isinstance(dat_raw, xr.Dataset):
            raise TypeError("`dat_raw` must be an instance of `xarray.Dataset`.")
        if not isinstance(dat_avg, xr.Dataset):
            raise TypeError("`dat_avg` must be an instance of `xarray.Dataset`.")
        if not hasattr(freq_range, "__iter__") or len(freq_range) != 2:
            raise ValueError("`freq_range` must be an iterable of length 2.")
        if "tke_vec" not in dat_avg:
            raise Exception(
                "The bin-averaged dataset must have the `tke_vec` variable "
                "(i.e., calculated from `ADVBinner.calc_turbulence` or "
                "`VelBinner.bin_average`)."
            )
        # Assign local names
        U_mag = dat_avg.velds.U_mag
        I_tke = dat_avg.velds.I_tke.values
        theta = np.angle(dat_avg.velds.U.values) - self._up_angle(
            dat_raw.velds.U.values
        )
        freq = dat_avg["freq"].values

        # Calculate constants
        alpha = 1.5
        intgrl = self._integral_TE01(I_tke, theta)

        # Interpolate PSD to the same time dimension as dat_avg
        umag_time_dim = U_mag.dims[-1]
        psd_time_dim = dat_avg["psd"].dims[-2]
        # If overlap is not 0%
        if dat_avg[psd_time_dim].size != U_mag[umag_time_dim].size:
            psd = dat_avg["psd"].interp({psd_time_dim: dat_avg[umag_time_dim]}).values
        else:
            psd = dat_avg["psd"].values

        # Index data to be used
        inds = (freq_range[0] < freq) & (freq < freq_range[1])
        psd = psd[..., inds]
        freq = freq[inds].reshape([1] * (dat_avg["psd"].ndim - 2) + [sum(inds)])

        # Estimate values
        # u & v components (equation 6)
        out = (
            np.nanmean((psd[0] + psd[1]) * freq ** (5 / 3), -1)
            / (21 / 55 * alpha * intgrl)
        ) ** (3 / 2) / U_mag.values

        # Add w component
        out += (
            np.nanmean(psd[2] * freq ** (5 / 3), -1) / (12 / 55 * alpha * intgrl)
        ) ** (3 / 2) / U_mag.values

        # Average the two estimates
        out *= 0.5

        return xr.DataArray(
            out,
            coords={"time": dat_avg["time"]},
            dims="time",
            attrs={
                "units": "m2 s-3",
                "long_name": "TKE Dissipation Rate",
                "standard_name": "specific_turbulent_kinetic_energy_dissipation_in_sea_water",
                "description": "TKE dissipation rate calculated using the "
                "method from Trowbridge and Elgar, 2001",
            },
        )

    def integral_length_scales(self, a_cov, U_mag, fs=None):
        """
        Calculate integral length scales from the autocovariance (or autocorrelation).

        Parameters
        ----------
        a_cov : xarray.DataArray ([dir,] time, lag)
          The autocovariance or autocorrelation array
          (i.e., computed using
          :func:`autocovariance <mhkit.dolfyn.velocity.VelBinner.autocovariance>`)
        U_mag : xarray.DataArray (time)
          The bin-averaged horizontal velocity (i.e., computed using
          :func:`U_mag <mhkit.dolfyn.velocity.Velocity.U_mag>`)
        fs : numeric
          The raw sample rate

        Returns
        -------
        L_int : numpy.ndarray ([dir,] time)
          The integral length scale.

        Notes
        ----
        The integral time scale (:math:`T_{int}`) is integral of the normalized
        autocovariance (autocorrelation) function, which theoretically decays to
        zero over time. Practically, :math:`T_{int}` is the integral from zero to
        the first zero-crossing lag-time of the autocorrelation function. The
        integral length scale (:math:`L_{int}`) then is the integral time scale
        multiplied by the bin speed.
        """

        if not isinstance(a_cov, xr.DataArray):
            raise TypeError("`a_cov` must be an instance of `xarray.DataArray`.")
        if len(a_cov["time"]) != len(U_mag["time"]):
            raise Exception("`U_mag` should be from ensembled-averaged dataset")

        fs = self._parse_fs(fs)
        # Normalize autocovariance/autocorrelation
        acov = a_cov / a_cov[..., 0]

        # Calculate first zero crossing in auto-correlation
        zero_crossing = np.nanargmin(~(acov < 0), axis=-1)

        # Calculate integral time scale
        T_int = np.zeros(acov.shape[:2])
        for i in range(3):
            for t in range(a_cov["time"].size):
                T_int[i, t] = np.trapezoid(acov[i, t][: zero_crossing[i, t]], dx=1 / fs)

        L_int = U_mag.values * T_int

        return xr.DataArray(
            L_int,
            coords={"dir": a_cov["dir"], "time": a_cov["time"]},
            attrs={
                "units": "m",
                "long_name": "Integral Length Scale",
                "standard_name": "turbulent_mixing_length_of_sea_water",
            },
        )
