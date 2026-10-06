import numpy as np
import xarray as xr
import warnings
from scipy import signal
from .tools import detrend_array, slice1d_along_axis
from ..time_utils import epoch_to_dt64, dt64_to_epoch

warnings.simplefilter("ignore", RuntimeWarning)


class Binner:
    def __init__(self, n_bin, fs, n_fft=None, n_fft_coh=None, noise=[0, 0, 0]):
        """
        Initialize a binning object.

        Parameters
        ----------
        n_bin : int
          Number of data points to include in a 'bin' (ensemble), not the
          number of bins
        fs : int
          Instrument sampling frequency in Hz
        n_fft : int
          Number of data points to use for fft (`n_fft`<=`n_bin`).
          Default: `n_fft`=`n_bin`
        n_fft_coh : int
          Number of data points to use for coherence and cross-spectra ffts
          Default: `n_fft_coh`=`n_fft`
        noise : list or ndarray
          Instrument's doppler noise in same units as velocity
        """

        self.n_bin = n_bin
        self.fs = fs
        self.n_fft = n_fft
        self.n_fft_coh = n_fft_coh
        self.noise = noise
        if n_fft is None:
            self.n_fft = n_bin
        elif n_fft > n_bin:
            self.n_fft = n_bin
            warnings.warn("n_fft must be smaller than n_bin, setting n_fft = n_bin")
        if n_fft_coh is None:
            self.n_fft_coh = int(self.n_fft)
        elif n_fft_coh > n_bin:
            self.n_fft_coh = int(n_bin)
            warnings.warn(
                "n_fft_coh must be smaller than or equal to n_bin, "
                "setting n_fft_coh = n_bin"
            )

    def _outshape(self, inshape, n_pad=0, n_bin=None):
        """
        Returns `outshape` (the 'reshape'd shape) for an `inshape` array.
        """
        n_bin = int(self._parse_nbin(n_bin))
        return list(inshape[:-1]) + [int(inshape[-1] // n_bin), int(n_bin + n_pad)]

    def _outshape_fft(self, inshape, n_fft=None, n_bin=None, step=None):
        """
        Returns `outshape` (the fft 'reshape'd shape) for an `inshape` array.
        """
        n_fft = self._parse_nfft(n_fft)
        n_bin = self._parse_nbin(n_bin)
        if step is None:
            step = n_bin
        n_slices = (inshape[-1] - n_bin) // step + 1
        return list(inshape[:-1]) + [int(n_slices), int(n_fft // 2)]

    def _parse_fs(self, fs=None):
        if fs is None:
            return self.fs
        return fs

    def _parse_nbin(self, n_bin=None):
        if n_bin is None:
            return self.n_bin
        return n_bin

    def _parse_nfft(self, n_fft=None):
        if n_fft is None:
            return self.n_fft
        if n_fft > self.n_bin:
            n_fft = self.n_bin
            warnings.warn("n_fft must be smaller than n_bin, setting n_fft = n_bin")
        return n_fft

    def _parse_nfft_coh(self, n_fft_coh=None):
        if n_fft_coh is None:
            return self.n_fft_coh
        if n_fft_coh > self.n_bin:
            n_fft_coh = int(self.n_bin)
            warnings.warn(
                "n_fft_coh must be smaller than or equal to n_bin, "
                "setting n_fft_coh = n_bin"
            )
        return n_fft_coh

    def _check_ds(self, raw_ds, out_ds):
        """
        Check that the attributes between two datasets match up.

        Parameters
        ----------
        raw_ds : xarray.Dataset
          Input dataset
        out_ds : xarray.Dataset
          Dataset to append `raw_ds` to. If None is supplied, this
          dataset is created from `raw_ds`.

        Returns
        -------
        out_ds : xarray.Dataset
        """

        for v in raw_ds.data_vars:
            if np.any(np.array(raw_ds[v].shape) == 0):
                raise RuntimeError(f"{v} cannot be averaged " "because it is empty.")
        if (
            "DutyCycle_NBurst" in raw_ds.attrs
            and raw_ds.attrs["DutyCycle_NBurst"] < self.n_bin
        ):
            warnings.warn(
                f"The averaging interval (n_bin = {self.n_bin})"
                "is larger than the burst interval "
                "(NBurst = {dat.attrs['DutyCycle_NBurst']})"
            )
        if raw_ds.fs != self.fs:
            raise Exception(
                f"The input data sample rate ({raw_ds.fs}) does not "
                "match the sample rate of this binning-object "
                "({self.fs})"
            )

        if out_ds is None:
            out_ds = type(raw_ds)()

        o_attrs = out_ds.attrs

        props = {}
        props["fs"] = self.fs
        props["n_bin"] = self.n_bin
        props["n_fft"] = self.n_fft
        props["description"] = (
            "Binned averages calculated from " 'ensembles of size "n_bin"'
        )
        props.update(raw_ds.attrs)

        for ky in props:
            if ky in o_attrs and o_attrs[ky] != props[ky]:
                # The values in out_ds must match `props` (raw_ds.attrs,
                # plus those defined above)
                raise AttributeError(
                    "The attribute '{}' of `out_ds` is inconsistent "
                    "with this `VelBinner` or the input data (`raw_ds`)".format(ky)
                )
            else:
                o_attrs[ky] = props[ky]
        return out_ds

    def _new_coords(self, array):
        """
        Function for setting up a new xarray.DataArray regardless of how
        many dimensions the input data-array has
        """
        dims = array.dims
        dims_list = []
        coords_dict = {}
        if len(array.shape) == 1 & ("dir" in array.coords):
            array = array.drop_vars("dir")
        for ky in dims:
            dims_list.append(ky)
            if "time" in ky:
                coords_dict[ky] = self.mean(array.time.values)
            else:
                coords_dict[ky] = array.coords[ky].values

        return dims_list, coords_dict

    def reshape(self, arr, step=None, n_bin=None):
        """
        Reshape the array `arr` into sliding windows of shape (..., n_slices, n_bin).

        Parameters
        ----------
        arr : numpy.ndarray
        step : int
          Number of samples to advance between consecutive windows.
          Default: n_bin (non-overlapping windows).
        n_bin : int
          Window (bin) size. Default is `self.n_bin`

        Returns
        -------
        out : numpy.ndarray
          Shape (..., n_slices, n_bin) where
          n_slices = (N - n_bin) // step + 1
        """

        n_bin = int(self._parse_nbin(n_bin))
        if arr.shape[-1] < n_bin:
            raise Exception("n_bin is larger than length of input array")
        if step is None:
            step = n_bin
        step = int(step)
        sliding_window = np.lib.stride_tricks.sliding_window_view(arr, n_bin, axis=-1)
        out = sliding_window[..., ::step, :].copy()
        return out

    def detrend(self, arr, axis=-1, step=None, n_bin=None):
        """
        Reshape the array `arr` into sliding windows and remove the
        best-fit trend line from each window.

        Parameters
        ----------
        arr : numpy.ndarray
        axis : int
          Axis along which to detrend. Default = -1
        step : int
          Number of samples to advance between consecutive windows.
          Default: n_bin (non-overlapping windows).
        n_bin : int
          Override this binner's n_bin. Default is `self.n_bin`

        Returns
        -------
        out : numpy.ndarray
        """

        return detrend_array(self.reshape(arr, step=step, n_bin=n_bin), axis=axis)

    def demean(self, arr, axis=-1, step=None, n_bin=None):
        """
        Reshape the array `arr` into sliding windows and remove the
        mean from each window.

        Parameters
        ----------
        arr : numpy.ndarray
        axis : int
          Axis along which to take mean. Default = -1
        step : int
          Number of samples to advance between consecutive windows.
          Default: n_bin (non-overlapping windows).
        n_bin : int
          Override this binner's n_bin. Default is `self.n_bin`

        Returns
        -------
        out : numpy.ndarray
        """

        dt = self.reshape(arr, step=step, n_bin=n_bin)
        return dt - np.nanmean(dt, axis)[..., None]

    def mean(self, arr, axis=-1, step=None, n_bin=None):
        """
        Reshape the array `arr` to shape (...,n,n_bin+n_pad)
        and take the mean of each bin along the specified `axis`.

        Parameters
        ----------
        arr : numpy.ndarray
        axis : int
          Axis along which to take mean. Default = -1
        step : int
          Number of samples to advance between consecutive windows.
          Default: n_bin (non-overlapping windows).
        n_bin : int
          Override this binner's n_bin. Default is `self.n_bin`

        Returns
        -------
        out : numpy.ndarray
        """

        if np.issubdtype(arr.dtype, np.datetime64):
            return epoch_to_dt64(
                self.mean(dt64_to_epoch(arr), axis=axis, step=step, n_bin=n_bin)
            )
        if axis != -1:
            arr = np.swapaxes(arr, axis, -1)
        n_bin = self._parse_nbin(n_bin)
        tmp = self.reshape(arr, step=step, n_bin=n_bin)

        return np.nanmean(tmp, -1)

    def variance(self, arr, axis=-1, step=None, n_bin=None):
        """
        Reshape the array `arr` to shape (...,n,n_bin+n_pad)
        and take the variance of each bin along the specified `axis`.

        Parameters
        ----------
        arr : numpy.ndarray
        axis : int
          Axis along which to take variance. Default = -1
        step : int
          Number of samples to advance between consecutive windows.
          Default: n_bin (non-overlapping windows).
        n_bin : int
          Override this binner's n_bin. Default is `self.n_bin`

        Returns
        -------
        out : numpy.ndarray
        """

        return np.nanvar(self.reshape(arr, step=step, n_bin=n_bin), axis=axis)

    def standard_deviation(self, arr, axis=-1, step=None, n_bin=None):
        """
        Reshape the array `arr` to shape (...,n,n_bin+n_pad)
        and take the standard deviation of each bin along the
        specified `axis`.

        Parameters
        ----------
        arr : numpy.ndarray
        axis : int
          Axis along which to take std dev. Default = -1
        step : int
          Number of samples to advance between consecutive windows.
          Default: n_bin (non-overlapping windows).
        n_bin : int
          Override this binner's n_bin. Default is `self.n_bin`

        Returns
        -------
        out : numpy.ndarray
        """

        return np.nanstd(self.reshape(arr, step=step, n_bin=n_bin), axis=axis)

    def bin_average(self, raw_ds, out_ds=None, names=None):
        """
        Bin the dataset and calculate the ensemble averages of each
        variable.

        Parameters
        ----------
        raw_ds : xarray.Dataset
          The raw data structure to be binned
        out_ds : xarray.Dataset
          The binned (output) data object to which averaged data is added.
        names : list of strings
          The names of variables to be averaged.  If `names` is None,
          all data in `raw_ds` will be binned.

        Returns
        -------
        out_ds : xarray.Dataset
          The new (or updated when `out_ds` is not None) dataset
          with the averages of all the variables in `raw_ds`.

        Raises
        ------
        AttributeError : when `out_ds` is supplied as input (not None)
        and the values in ``out_ds.attrs`` are inconsistent with
        ``raw_ds.attrs`` or the properties of this VelBinner (`n_bin`,
        `n_fft`, `fs`, etc.)

        Notes
        -----
        ``raw_ds.attrs`` are copied to ``out_ds.attrs``. Inconsistencies
        between the two (when `out_ds` is specified as input) raise an
        AttributeError.
        """

        out_ds = self._check_ds(raw_ds, out_ds)

        if names is None:
            names = raw_ds.data_vars

        for ky in names:
            # set up dimensions and coordinates for Dataset
            dims_list = raw_ds[ky].dims
            if any([ar for ar in dims_list if "altraw" in ar]):
                continue
            coords_dict = {}
            for nm in dims_list:
                if "time" in nm:
                    coords_dict[nm] = self.mean(raw_ds[ky][nm].values)
                else:
                    coords_dict[nm] = raw_ds[ky][nm].values

            # create Dataset
            if "ensemble" not in ky:
                try:  # variables with time coordinate
                    out_ds[ky] = xr.DataArray(
                        self.mean(raw_ds[ky].values),
                        coords=coords_dict,
                        dims=dims_list,
                        attrs=raw_ds[ky].attrs,
                    ).astype("float32")
                except:  # variables not needing averaging
                    pass

        # Add standard deviation
        std = self.standard_deviation(raw_ds.velds.U_mag.values)
        out_ds["U_std"] = xr.DataArray(
            std.astype("float32"),
            dims=raw_ds.velds.U_mag.dims,
            attrs={
                "units": "m s-1",
                "long_name": "Water Velocity Standard Deviation",
            },
        )

        return out_ds

    def bin_variance(self, raw_ds, out_ds=None, names=None, suffix="_var"):
        """
        Bin the dataset and calculate the ensemble variances of each
        variable. Complementary to :func:`bin_average <mhkit.dolfyn.velocity.VelBinner.bin_average>`.

        Parameters
        ----------
        raw_ds : xarray.Dataset
          The raw data structure to be binned.
        out_ds : xarray.Dataset
          The binned (output) dataset to which variance data is added,
          nominally the dataset output from
          :func:`bin_average <mhkit.dolfyn.velocity.VelBinner.bin_average>`.
        names : list of strings
          The names of variables of which to calculate variance. If
          `names` is None, all data in `raw_ds` will be binned.

        Returns
        -------
        out_ds : xarray.Dataset
          The new (or updated when `out_ds` is not None) dataset
          with the variance of all the variables in `raw_ds`.

        Raises
        ------
        AttributeError : when `out_ds` is supplied as input (not None)
        and the values in ``out_ds.attrs`` are inconsistent with
        ``raw_ds.attrs`` or the properties of this VelBinner (`n_bin`,
        `n_fft`, `fs`, etc.)

        Notes
        -----
        ``raw_ds.attrs`` are copied to ``out_ds.attrs``. Inconsistencies
        between the two (when `out_ds` is specified as input) raise an
        AttributeError.
        """

        out_ds = self._check_ds(raw_ds, out_ds)

        if names is None:
            names = raw_ds.data_vars

        for ky in names:
            # set up dimensions and coordinates for dataarray
            dims_list = raw_ds[ky].dims
            if any([ar for ar in dims_list if "altraw" in ar]):
                continue
            coords_dict = {}
            for nm in dims_list:
                if "time" in nm:
                    coords_dict[nm] = self.mean(raw_ds[ky][nm].values)
                else:
                    coords_dict[nm] = raw_ds[ky][nm].values

            # create Dataset
            if "ensemble" not in ky:
                try:  # variables with time coordinate
                    out_ds[ky + suffix] = xr.DataArray(
                        self.variance(raw_ds[ky].values),
                        coords=coords_dict,
                        dims=dims_list,
                        attrs=raw_ds[ky].attrs,
                    ).astype("float32")
                except:  # variables not needing averaging
                    pass

        return out_ds

    def autocovariance(self, raw_data, n_bin=None):
        """
        Calculate the auto-covariance of the raw-signal `raw_data`

        Parameters
        ----------
        raw_data : xarray.DataArray
          The raw dataArray of which to calculate auto-covariance
        n_bin : float
          Number of data elements to use

        Returns
        -------
        da : xarray.DataArray
          The auto-covariance of raw_data

        Notes
        -----
        As opposed to cross-covariance, which returns the full
        cross-covariance between two arrays, this function only
        returns a quarter of the full auto-covariance. It computes the
        auto-covariance over half of the range, then averages the two
        sides (to return a 'quartered' covariance).

        This has the advantage that the 0 index is actually zero-lag.
        """

        indat = raw_data.values

        n_bin = self._parse_nbin(n_bin)
        out = np.empty(
            self._outshape(indat.shape, n_bin=n_bin)[:-1] + [int(n_bin // 4)],
            dtype=indat.dtype,
        )
        # Need to pad velocity timeseries with zeros to incoporate the full range of
        # the auto-covariance.
        n_pad = int(n_bin / 2 - 2)
        npd0 = n_pad // 2
        npd1 = (n_pad + 1) // 2
        # Pad with zeros at boundaries to replicate the original n_pad behavior
        indat_padded = np.pad(
            indat, pad_width=[(0, 0)] * (indat.ndim - 1) + [(npd0, npd1)]
        )
        dt1 = self.reshape(indat_padded, step=int(n_bin), n_bin=int(n_bin + n_pad))
        # Here we de-mean only on the 'valid' range:
        dt1 = dt1 - dt1[..., :, int(n_bin // 4) : int(-n_bin // 4)].mean(-1)[..., None]
        dt2 = self.demean(indat)
        se = slice(int(n_bin // 4) - 1, None, 1)
        sb = slice(int(n_bin // 4) - 1, None, -1)
        for slc in slice1d_along_axis(dt1.shape, -1):
            tmp = np.correlate(dt1[slc], dt2[slc], "valid")
            # The zero-padding in reshape means we compute coherence
            # from one-sided time-series for first and last points.
            if slc[-2] == 0:
                out[slc] = tmp[se]
            elif slc[-2] == dt2.shape[-2] - 1:
                out[slc] = tmp[sb]
            else:
                # For the others we take the average of the two sides.
                out[slc] = (tmp[se] + tmp[sb]) / 2

        dims_list, coords_dict = self._new_coords(raw_data)
        # tack on new coordinate
        dims_list.append("lag")
        coords_dict["lag"] = np.arange(n_bin // 4)

        da = xr.DataArray(
            out.astype("float32"),
            coords=coords_dict,
            dims=dims_list,
        )
        da["lag"].attrs["units"] = "timestep"

        return da

    def _psd_base(
        self,
        dat,
        fs=None,
        window="hann",
        noise=0,
        n_bin=None,
        n_fft=None,
        pct_overlap=0.5,
    ):
        """
        Calculate the power spectral density of `dat`

        Parameters
        ----------
        dat : xarray.DataArray
          The raw dataArray of which to calculate the psd.
        fs : float (optional)
          The sample rate (Hz).
        window : {None, 1, 'hann', numpy.ndarray}
          The window to use (default: 'hann'). Valid entries are:
            - None,1               : uses a 'boxcar' or ones window.
            - 'hann'               : hanning window.
            - a length(nfft) array : use this as the window directly.
        noise  : float
          The white-noise level of the measurement (in the same units
          as `dat`).
        n_bin : int
          n_bin of raw_data2, number of elements per bin if 'None' is taken
          from VelBinner
        n_fft : int
          n_fft of raw_data2, number of elements per bin if 'None' is taken
          from VelBinner
        pct_overlap : float
          The percent overlap between FFT windows (default: 0.5)

        Returns
        -------
        out : numpy.ndarray
          The power spectral density of `dat`

        Notes
        -----
        PSD's are calculated based on sample rate units
        """

        fs = self._parse_fs(fs)
        # n_bin determines the number of time bins in the output
        n_bin = self._parse_nbin(n_bin)
        # n_fft determines the length and resolution of the frequency vector
        n_fft = self._parse_nfft(n_fft)
        # step is the advance between consecutive bin slices.
        # For pct_overlap fraction of overlap: step = n_bin * (1 - pct_overlap).
        step = int((1 - pct_overlap) * n_bin)
        out = np.empty(
            self._outshape_fft(dat.shape, n_fft=n_fft, n_bin=n_bin, step=step)
        )
        n_samples = out.shape[-2]
        for i in range(n_samples):
            sample_slice = slice(i * step, i * step + int(n_bin))
            if any(np.isnan(dat[sample_slice])):
                warnings.warn("Skipped PSD window containing NaNs.")
                continue
            freq, psd = signal.welch(
                dat[sample_slice],
                fs=fs,
                window=window,
                nperseg=n_fft,
                noverlap=int(pct_overlap * n_fft),
                detrend="linear",
                return_onesided=True,
                scaling="density",
            )
            # Drop DC bin (index 0): always ~0 after linear detrending, excluded by convention
            out[i, :] = psd[1:]
        if np.any(noise):
            out -= noise**2 / (fs / 2)
            # Make sure all values of the PSD are >0 (but still small):
            out[out < 0] = np.min(np.abs(out)) / 100
        return freq[1:], out

    def _csd_base(
        self,
        dat1,
        dat2,
        fs=None,
        window="hann",
        n_fft=None,
        n_bin=None,
        pct_overlap=0.5,
    ):
        """
        Compute the cross power spectral density (CPSD) of the signals dat1 and dat2.

        Parameters
        ----------
        dat1 : numpy.ndarray
          The first raw dataArray of which to calculate the cpsd.
        dat2 : numpy.ndarray
          The second raw dataArray of which to calculate the cpsd.
        fs : float (optional)
          The sample rate (Hz).
        window : {None, 1, 'hann', numpy.ndarray}
          The window to use (default: 'hann'). Valid entries are:
            - None,1               : uses a 'boxcar' or ones window.
            - 'hann'               : hanning window.
            - a length(nfft) array : use this as the window directly.
        n_fft : int
          Number of elements in the FFT. If 'None', is taken
          from VelBinner (uses n_fft_coh).
        n_bin : int
          Number of elements per bin. If 'None', is taken
          from VelBinner.
        pct_overlap : float
          The percent overlap between sliding windows (default: 0.5).

        Returns
        -------
        out : numpy.ndarray
          The cross power spectral density of `dat1` and `dat2`

        Notes
        -----
        PSDs are calculated based on sample rate units.
        This removes a linear trend from the signals.
        The two signals must be the same length and both be real.

        This performs:

        .. math::

            fft(a)*conj(fft(b))

        This implementation is consistent with the numpy.correlate
        definition of correlation.  (The conjugate of D.B. Chelton's
        definition of correlation.)

        The units of the spectra is the product of the units of `a` and
        `b`, divided by the units of fs.
        """

        fs = self._parse_fs(fs)
        n_fft = self._parse_nfft_coh(n_fft)
        n_bin = self._parse_nbin(n_bin)

        if dat1.shape != dat2.shape:
            raise ValueError(
                "Cross-spectral density requires equal-length input arrays. "
                "Quasi-synchronized (different sample rate) inputs are not supported."
            )
        if np.iscomplexobj(dat1) or np.iscomplexobj(dat2):
            raise ValueError("Velocity cannot be complex")

        step = int((1 - pct_overlap) * n_bin)
        oshp = self._outshape_fft(dat1.shape, n_fft=n_fft, n_bin=n_bin, step=step)
        out = np.empty(oshp, dtype="c{}".format(dat1.dtype.itemsize * 2))
        n_samples = oshp[-2]
        for i in range(n_samples):
            sample_slice = slice(i * step, i * step + int(n_bin))
            if any(np.isnan(dat1[sample_slice])) or any(np.isnan(dat2[sample_slice])):
                warnings.warn("Skipped PSD window containing NaNs.")
                continue
            freq, cpsd = signal.csd(
                dat1[sample_slice],
                dat2[sample_slice],
                fs=fs,
                window=window,
                nperseg=n_fft,
                noverlap=int(pct_overlap * n_fft),
                detrend="linear",
                return_onesided=True,
                scaling="density",
            )
            # Drop DC bin (index 0): always ~0 after linear detrending, excluded by convention
            out[i, :] = cpsd[1:]
        return freq[1:], out

    def power_spectral_density(
        self,
        raw_data,
        freq_units="rad/s",
        fs=None,
        window="hann",
        noise=0,
        n_bin=None,
        n_fft=None,
        pct_overlap=0,
    ):
        """
        Calculate the power spectral density of velocity.

        Parameters
        ----------
        raw_data : xr.DataArray (dir, time)
          The raw velocity data
        freq_units : string
          Frequency units of the returned spectra in either Hz or rad/s
        fs : float (optional)
          The sample rate. Default is `binner.fs`
        window: str
          Type of window to apply to each FFT segment.
          Example options: 'boxcar', 'hann', 'hamming', 'blackman', 'bartlett'.
          See `scipy.signal.window` for more options.
          Default: 'hann'.
        noise : numeric or array
          Instrument noise level in same units as velocity.
          Default = 0 (ADCP) or [0, 0, 0] (ADV)
        n_bin : int (optional)
          The bin-size. Default = `self.n_bin`
        n_fft : int (optional)
          The fft size. Default = `self.n_fft`
        pct_overlap : float (optional)
          Fractional overlap between consecutive sliding windows, in [0, 1).
          Controls both the bin-to-bin advance and the within-bin FFT overlap
          passed to scipy.signal.welch. Industry standard is 50%.
          Default = 0 (0% overlap).

        Returns
        -------
        psd : xarray.DataArray (dir, time, freq)
          The spectra in the 'u', 'v', and 'w' directions.
        """

        fs_in = self._parse_fs(fs)
        n_fft = self._parse_nfft(n_fft)
        if "xarray" in type(raw_data).__module__:
            vel = raw_data.values
        if ("rad" not in freq_units) and ("Hz" not in freq_units):
            raise ValueError("`freq_units` should be one of 'Hz' or 'rad/s'")
        if (pct_overlap < 0) or (pct_overlap > 1):
            raise ValueError(
                f"pct_overlap must be between 0 and 1, received {pct_overlap}."
            )
        n_bin = self._parse_nbin(n_bin)
        step = int((1 - pct_overlap) * n_bin)

        # Set units correctly
        if "rad" in freq_units:
            fs = 2 * np.pi * fs_in
            freq_units = "rad s-1"
            units = "m2 s-1 rad-1"
        else:
            fs = fs_in
            freq_units = "Hz"
            units = "m2 s-2 Hz-1"

        # Spectra, if velocity is a 2D array (dir, time)
        if len(vel.shape) >= 2:
            if vel.shape[0] != 3:
                raise ValueError(
                    "Function can only handle 1D or 3D arrays."
                    " If ADCP data, please select a specific depth bin."
                )
            if np.array(noise).any():
                if np.size(noise) != 3:
                    raise ValueError("Noise is expected to be an array of 3 scalars")
            else:
                # Reset default to list of 3 zeros
                noise = np.array([0, 0, 0])
            # Set up input velocity array, coordinates, and dimensions
            vel_in = vel[:3]
            coords = {"S": self.S}
            dims = ["S"]

        # Spectra, if velocity is a single array
        else:
            if np.array(noise).any() and np.size(noise) > 1:
                raise ValueError("Noise is expected to be a scalar")
            # Add dummy axis
            vel_in = vel[np.newaxis]
            noise = np.atleast_1d(noise)
            coords = {}
            dims = []

        # Do power spectral density calculation for each velocity component
        out = np.empty(
            self._outshape_fft(vel_in.shape, n_fft=n_fft, n_bin=n_bin, step=step)
        )
        for idx in range(vel_in.shape[0]):
            f, out[idx] = self._psd_base(
                vel_in[idx],
                fs=fs,
                noise=noise[idx],
                window=window,
                n_bin=n_bin,
                n_fft=n_fft,
                pct_overlap=pct_overlap,
            )
        # If not 3D (ADV) data, remove the new axis
        if "S" not in dims:
            out = out[0]

        # Create frequency vector, also checks whether using f or omega
        freq = xr.DataArray(
            f,
            dims=["freq"],
            name="freq",
            attrs={
                "units": freq_units,
                "long_name": "FFT Frequency Vector",
                "coverage_content_type": "coordinate",
            },
        )

        # Update coordinates and dimensions
        time = raw_data[raw_data.dims[-1]].values
        time_coord = self.mean(time, step=step, n_bin=n_bin)
        coords.update(
            {
                "time_psd": time_coord,
                "freq": freq,
            }
        )
        dims += ["time_psd", "freq"]

        return xr.DataArray(
            out,
            coords=coords,
            dims=dims,
            attrs={
                "units": units,
                "n_fft": n_fft,
                "long_name": "Power Spectral Density",
            },
        )

    def cross_spectral_density(
        self,
        raw_data,
        freq_units="rad/s",
        fs=None,
        window="hann",
        n_bin=None,
        n_fft_coh=None,
        pct_overlap=0,
    ):
        """
        Calculate the cross-spectral density of velocity components.

        Parameters
        ----------
        raw_data : xarray.DataArray
          The raw 3D velocity data.
        freq_units : string
          Frequency units of the returned spectra in either Hz or rad/s
          (`f` or :math:`\\omega`)
        fs : float (optional)
          The sample rate. Default = `self.fs`
        window: str
          Type of window to apply to each FFT segment.
          Example options: 'boxcar', 'hann', 'hamming', 'blackman', 'bartlett'.
          See `scipy.signal.window` for more options.
          Default: 'hann'.
        n_bin : int (optional)
          The bin-size. Default = `self.n_bin`
        n_fft_coh : int (optional)
          The fft size. Default = `self.n_fft_coh`
        pct_overlap : float (optional)
          Fractional overlap between consecutive sliding windows, in [0, 1).
          Controls both the bin-to-bin advance and the within-bin FFT overlap
          passed to scipy.signal.welch. Industry standard is 50%.
          Default = 0 (0% overlap).

        Returns
        -------
        csd : xarray.DataArray (3, M, N_FFT)
          The first-dimension of the cross-spectrum is the three
          different cross-spectra: :math:`uv`, :math:`uw`, :math:`vw`.
        """

        if not isinstance(raw_data, xr.DataArray):
            raise TypeError("`raw_data` must be an instance of `xarray.DataArray`.")
        if ("rad" not in freq_units) and ("Hz" not in freq_units):
            raise ValueError("`freq_units` should be one of 'Hz' or 'rad/s'")
        if (pct_overlap < 0) or (pct_overlap > 1):
            raise ValueError(
                f"pct_overlap must be between 0 and 1, received {pct_overlap}."
            )

        fs_in = self._parse_fs(fs)
        n_bin = self._parse_nbin(n_bin)
        n_fft = self._parse_nfft_coh(n_fft_coh)
        step = int((1 - pct_overlap) * n_bin)

        # Get time coord before changing raw_data from xarray to numpy array
        time_coord = self.mean(raw_data["time"].values, step=step, n_bin=n_bin)
        raw_data = raw_data.values
        if len(np.shape(raw_data)) != 2:
            raise Exception(
                "This function is only valid for calculating TKE using "
                "the 3D velocity vector from an ADV."
            )

        out = np.empty(
            self._outshape_fft(raw_data[:3].shape, n_fft=n_fft, n_bin=n_bin, step=step),
            dtype="complex",
        )

        # Create frequency vector, also checks whether using f or omega
        if "rad" in freq_units:
            fs = 2 * np.pi * fs_in
            freq_units = "rad s-1"
            units = "m2 s-1 rad-1"
        else:
            fs = fs_in
            freq_units = "Hz"
            units = "m2 s-2 Hz-1"

        for ip, ipair in enumerate(self._cross_pairs):
            f, out[ip] = self._csd_base(
                raw_data[ipair[0]],
                raw_data[ipair[1]],
                fs=fs,
                window=window,
                n_bin=n_bin,
                n_fft=n_fft,
                pct_overlap=pct_overlap,
            )
        coh_freq = xr.DataArray(
            f,
            dims=["coh_freq"],
            name="coh_freq",
            attrs={
                "units": freq_units,
                "long_name": "FFT Frequency Vector",
                "coverage_content_type": "coordinate",
            },
        )

        csd = xr.DataArray(
            out.astype("complex64"),
            coords={"C": self.C, "time_psd": time_coord, "coh_freq": coh_freq},
            dims=["C", "time_psd", "coh_freq"],
            attrs={
                "units": units,
                "n_fft_coh": n_fft,
                "long_name": "Cross Spectral Density",
            },
        )
        csd["coh_freq"].attrs["units"] = freq_units

        return csd
