import numpy as np
import xarray as xr

from .rotate.api import rotate2, set_declination, set_inst2head_rotmat
from .io.api import save
from ..utils.binning_tools.binner import Binner
from ..utils.binning_tools.tools import convert_degrees
from ..utils.time_utils import dt64_to_epoch, dt64_to_datetime


@xr.register_dataset_accessor("velds")  # 'vel dataset'
class Velocity:
    """
    All ADCP and ADV xarray datasets wrap this base class.

    The turbulence-related attributes defined within this class
    assume that the  ``'tke_vec'`` and ``'stress_vec'`` data entries are
    included in the dataset. These are typically calculated using a
    :class:`VelBinner` tool, but the method for calculating these
    variables can depend on the details of the measurement
    (instrument, it's configuration, orientation, etc.).
    """

    ########
    # Major components of the dolfyn-API

    def rotate2(self, out_frame="earth", inplace=True):
        """
        Rotate the dataset to a new coordinate system.

        Parameters
        ----------
        out_frame : string {'beam', 'inst', 'earth', 'principal'}
          The coordinate system to rotate the data into.

        inplace : bool
          When True the existing data object is modified. When False
          a copy is returned. Default = True

        Returns
        -------
        ds : xarray.Dataset or None
          Returns the rotated dataset **when `inplace=False`**, otherwise
          returns None.

        Notes
        -----
        - This function rotates all variables in ``ds.attrs['rotate_vars']``.

        - To rotate to the 'principal' frame, a value of
          ``ds.attrs['principal_heading']`` must exist. The function
          :func:`calc_principal_heading <dolfyn.calc_principal_heading>`
          is recommended for this purpose, e.g.::

              ds.attrs['principal_heading'] = dolfyn.calc_principal_heading(ds['vel'].mean(range))

          where here we are using the depth-averaged velocity to calculate
          the principal direction.
        """

        return rotate2(self.ds, out_frame, inplace)

    def set_declination(self, declin, inplace=True):
        """
        Set the magnetic declination

        Parameters
        ----------
        declination : float
          The value of the magnetic declination in degrees (positive
          values specify that Magnetic North is clockwise from True North)

        inplace : bool
          When True the existing data object is modified. When False
          a copy is returned. Default = True

        Returns
        -------
        ds : xarray.Dataset or None
          Returns the rotated dataset **when ``inplace=False``**, otherwise
          returns None.

        Notes
        -----
        This method modifies the data object in the following ways:

        - If the dataset is in the *earth* reference frame at the time of
        setting declination, it will be rotated into the "*True-East*,
        *True-North*, Up" (hereafter, ETU) coordinate system

        - ``dat['orientmat']`` is modified to be an ETU to
        instrument (XYZ) rotation matrix (rather than the magnetic-ENU to
        XYZ rotation matrix). Therefore, all rotations to/from the 'earth'
        frame will now be to/from this ETU coordinate system.

        - The value of the specified declination will be stored in
        ``dat.attrs['declination']``

        - ``dat['heading']`` is adjusted for declination
        (i.e., it is relative to True North).

        - If ``dat.attrs['principal_heading']`` is set, it is
        adjusted to account for the orientation of the new 'True'
        earth coordinate system (i.e., calling set_declination on a
        data object in the principal coordinate system, then calling
        dat.rotate2('earth') will yield a data object in the new
        'True' earth coordinate system)
        """

        return set_declination(self.ds, declin, inplace)

    def set_inst2head_rotmat(self, rotmat, inplace=True):
        """
        Set the instrument to head rotation matrix for the Nortek ADV if it
        hasn't already been set through a '.userdata.json' file.

        Parameters
        ----------
        rotmat : float
          3x3 rotation matrix
        inplace : bool
          When True the existing data object is rotated. When False
          a copy is returned that is rotated. Default = True

        Returns
        -------
        ds : xarray.Dataset or None
          Returns the rotated dataset **when `inplace=False`**, otherwise
          returns None.

        Notes
        -----
        If the data object is in earth or principal coords, it is first
        rotated to 'inst' before assigning inst2head_rotmat, it is then
        rotated back to the coordinate system in which it was input. This
        way the inst2head_rotmat gets applied correctly (in inst
        coordinate system).
        """

        return set_inst2head_rotmat(self.ds, rotmat, inplace)

    def save(self, filename, **kwargs):
        """
        Save the data object (underlying xarray dataset) as netCDF (.nc).

        Parameters
        ----------
        filename : str
            Filename and/or path with the '.nc' extension
        **kwargs : dict
          These are passed directly to :func:`xarray.Dataset.to_netcdf`.

        Notes
        -----
        See DOLfYN's :func:`save <mhkit.dolfyn.io.api.save>` function for
        additional details.
        """

        save(self.ds, filename, **kwargs)

    ########
    # Magic methods of the API

    def __init__(self, ds, *args, **kwargs):
        self.ds = ds

    def __getitem__(self, key):
        return self.ds[key]

    def __contains__(self, val):
        return val in self.ds

    def __repr__(
        self,
    ):
        time_string = "{:.2f} {} (started: {})"
        time = "time" if "time" in self else "time_avg"
        if time not in self or dt64_to_epoch(self[time][0]) < 1:
            time_string = "-->No Time Information!<--"
        else:
            tm = self[time][[0, -1]].values
            dt = dt64_to_datetime(tm[0])[0]
            delta = (dt64_to_epoch(tm[-1]) - dt64_to_epoch(tm[0])) / (3600 * 24)  # days
            if delta > 1:
                units = "days"
            elif delta * 24 > 1:
                units = "hours"
                delta *= 24
            elif delta * 24 * 60 > 1:
                delta *= 24 * 60
                units = "minutes"
            else:
                delta *= 24 * 3600
                units = "seconds"
            try:
                time_string = time_string.format(
                    delta, units, dt.strftime("%b %d, %Y %H:%M")
                )
            except AttributeError:
                time_string = "-->Error in time info<--"

        p = self.ds.attrs
        t_shape = self[time].shape
        if len(t_shape) > 1:
            shape_string = "({} bins, {} pings @ {}Hz)".format(
                t_shape[0], t_shape, p.get("fs")
            )
        else:
            shape_string = "({} pings @ {}Hz)".format(t_shape[0], p.get("fs", "??"))
        _header = (
            "<%s data object>: "
            " %s %s\n"
            "  . %s\n"
            "  . %s-frame\n"
            "  . %s\n"
            % (
                p.get("inst_type"),
                self.ds.attrs["inst_make"],
                self.ds.attrs["inst_model"],
                time_string,
                p.get("coord_sys"),
                shape_string,
            )
        )
        _vars = "  Variables:\n"

        # Specify which variable show up in this view here.
        # * indicates a wildcard
        # This list also sets the display order.
        # Only the first 12 matches are displayed.
        show_vars = [
            "time*",
            "vel*",
            "range",
            "range_echo",
            "orientmat",
            "heading",
            "pitch",
            "roll",
            "temp",
            "press*",
            "amp*",
            "corr*",
            "accel",
            "angrt",
            "mag",
            "echo",
        ]
        n = 0
        for v in show_vars:
            if n > 12:
                break
            if v.endswith("*"):
                v = v[:-1]  # Drop the '*'
                for nm in self.variables:
                    if n > 12:
                        break
                    if nm.startswith(v):
                        n += 1
                        _vars += "  - {} {}\n".format(nm, self.ds[nm].dims)
            elif v in self.ds:
                _vars += "  - {} {}\n".format(v, self.ds[v].dims)
        if n < len(self.variables):
            _vars += "  ... and others (see `<obj>.variables`)\n"
        return _header + _vars

    ######
    # Duplicate valuable xarray properties here.
    @property
    def variables(
        self,
    ):
        """A sorted list of the variable names in the dataset."""
        return sorted(self.ds.variables)

    @property
    def attrs(
        self,
    ):
        """The attributes in the dataset."""
        return self.ds.attrs

    @property
    def coords(
        self,
    ):
        """The coordinates in the dataset."""
        return self.ds.coords

    ######
    # A bunch of DOLfYN specific properties
    @property
    def u(
        self,
    ):
        """
        The first velocity component.

        This is simply a shortcut to ``self['vel'][0]``. Therefore,
        depending on the coordinate system of the data object
        (``self.attrs['coord_sys']``), it is:

        - beam:      beam1
        - inst:      x
        - earth:     east
        - principal: streamwise
        """
        try:
            return self.ds["vel"][0].drop_vars("dir")
        except KeyError:
            return self.ds["vel_avg"][0].drop_vars("dir")

    @property
    def v(
        self,
    ):
        """
        The second velocity component.

        This is simply a shortcut to ``self['vel'][1]``. Therefore,
        depending on the coordinate system of the data object
        (``self.attrs['coord_sys']``), it is:

        - beam:      beam2
        - inst:      y
        - earth:     north
        - principal: cross-stream
        """
        try:
            return self.ds["vel"][1].drop_vars("dir")
        except KeyError:
            return self.ds["vel_avg"][1].drop_vars("dir")

    @property
    def w(
        self,
    ):
        """
        The third velocity component.

        This is simply a shortcut to ``self['vel'][2]``. Therefore,
        depending on the coordinate system of the data object
        (``self.attrs['coord_sys']``), it is:

        - beam:      beam3
        - inst:      z
        - earth:     up
        - principal: up
        """
        try:
            return self.ds["vel"][2].drop_vars("dir")
        except KeyError:
            return self.ds["vel_avg"][2].drop_vars("dir")

    @property
    def U(
        self,
    ):
        """Horizontal velocity as a complex quantity"""

        return xr.DataArray(
            (self.u + self.v * 1j).astype("complex64"),
            attrs={"units": "m s-1", "long_name": "Horizontal Water Velocity"},
        )

    @property
    def U_mag(
        self,
    ):
        """Horizontal velocity magnitude, i.e., speed"""

        return xr.DataArray(
            np.abs(self.U).astype("float32"),
            attrs={
                "units": "m s-1",
                "long_name": "Water Speed",
                "standard_name": "sea_water_speed",
            },
        )

    @property
    def U_dir(
        self,
    ):
        """
        Angle of horizontal velocity vector, i.e., direction. Direction is 'to',
        as opposed to 'from'. This function calculates angle as
        "degrees CCW from X/East/streamwise" and then converts it to
        "degrees CW from X/North/streamwise".
        """

        def convert_to_CW(angle):
            if self.ds.coord_sys == "earth":
                # Convert "deg CCW from East" to "deg CW from North" [0, 360]
                angle = convert_degrees(angle, tidal_mode=False)
                relative_to = self.ds.dir[1].values
            else:
                # Switch to clockwise and from [-180, 180] to [0, 360]
                angle *= -1
                angle[angle < 0] += 360
                relative_to = self.ds.dir[0].values
            return angle, relative_to

        # Convert from radians to degrees
        angle, rel = convert_to_CW(np.angle(self.U) * (180 / np.pi))

        return xr.DataArray(
            angle.astype("float32"),
            dims=self.U.dims,
            coords=self.U.coords,
            attrs={
                "units": "degree",
                "long_name": "Water Direction, CW from " + str(rel),
                "standard_name": "sea_water_to_direction",
            },
        )

    @property
    def E_coh(
        self,
    ):
        """
        Coherent turbulent energy

        Neil Kelley's 'coherent turbulence energy', which is the
        root-mean-square of the Reynold's stresses.

        See: NLR Technical Report TP-500-52353
        """
        E_coh = (self.upwp_**2 + self.upvp_**2 + self.vpwp_**2) ** (0.5)

        return xr.DataArray(
            E_coh.astype("float32"),
            coords={"time": self.ds["stress_vec"].time},
            dims=["time"],
            attrs={
                "units": self.ds["stress_vec"].units,
                "long_name": "Coherent Turbulence Energy",
            },
        )

    @property
    def I_tke(self, thresh=0):
        """
        Turbulent kinetic energy intensity.

        Ratio of sqrt(TKE) to horizontal velocity magnitude.
        """
        I_tke = np.ma.masked_where(
            self.U_mag < thresh, np.sqrt(2 * self.tke) / self.U_mag
        )
        return xr.DataArray(
            I_tke.data.astype("float32"),
            coords=self.U_mag.coords,
            dims=self.U_mag.dims,
            attrs={"units": "1", "long_name": "TKE Intensity"},
        )

    @property
    def I(self, thresh=0):
        """
        Turbulence intensity.

        Ratio of standard deviation of horizontal velocity
        to horizontal velocity magnitude.
        """
        I = np.ma.masked_where(self.U_mag < thresh, self.ds["U_std"] / self.U_mag)
        return xr.DataArray(
            I.data.astype("float32"),
            coords=self.U_mag.coords,
            dims=self.U_mag.dims,
            attrs={"units": "1", "long_name": "Turbulence Intensity"},
        )

    @property
    def tke(
        self,
    ):
        """Turbulent kinetic energy (sum of the three components)"""
        tke = self.ds["tke_vec"].sum("tke") / 2
        tke.name = "TKE"
        tke.attrs["units"] = self.ds["tke_vec"].units
        tke.attrs["long_name"] = "TKE"
        tke.attrs["standard_name"] = "specific_turbulent_kinetic_energy_of_sea_water"
        return tke

    @property
    def upvp_(
        self,
    ):
        """:math:`\\overline{u'v'}` Reynolds stress"""

        return self.ds["stress_vec"].sel(tau="upvp_").drop_vars("tau")

    @property
    def upwp_(
        self,
    ):
        """:math:`\\overline{u'w'}` Reynolds stress"""

        return self.ds["stress_vec"].sel(tau="upwp_").drop_vars("tau")

    @property
    def vpwp_(
        self,
    ):
        """:math:`\\overline{v'w'}` Reynolds stress"""

        return self.ds["stress_vec"].sel(tau="vpwp_").drop_vars("tau")

    @property
    def upup_(
        self,
    ):
        """:math:`\\overline{u'u'}` component of the TKE vector"""

        return self.ds["tke_vec"].sel(tke="upup_").drop_vars("tke")

    @property
    def vpvp_(
        self,
    ):
        """:math:`\\overline{v'v'}` component of the TKE vector"""

        return self.ds["tke_vec"].sel(tke="vpvp_").drop_vars("tke")

    @property
    def wpwp_(
        self,
    ):
        """:math:`\\overline{w'w'}` component of the TKE vector"""

        return self.ds["tke_vec"].sel(tke="wpwp_").drop_vars("tke")


class VelBinner(Binner):
    def __init__(
        self,
        n_bin,
        fs,
        n_fft=None,
        n_fft_coh=None,
        noise=None,
    ):
        """
        This is the base binning (averaging) tool.
        All DOLfYN binning tools derive from this base class.

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

        Examples
        --------
        The VelBinner class is used to compute averages and turbulence
        statistics from 'raw' (not averaged) ADV or ADP measurements, for
        example::

            # First read or load some data.
            rawdat = dolfyn.read_example('BenchFile01.ad2cp')

            # Now initialize the averaging tool:
            binner = dolfyn.VelBinner(n_bin=600, fs=rawdat.fs)

            # This computes the basic averages
            avg = binner.bin_average(rawdat)
        """

        Binner.__init__(self, n_bin, fs, n_fft, n_fft_coh, noise)

    def doppler_noise_level(self, psd, pct_fN=0.8):
        """
        Estimate bias due to Doppler noise using the noise floor
        of the velocity spectra.

        Parameters
        ----------
        psd : xarray.DataArray ([dir,] time, freq)
          The power spectral density of velocity (auto-spectra) from an ADV
          or from a single depth bin (range) of an ADCP.
        pct_fN : float
          Percent of Nyquist frequency to calculate characeristic frequency.
          Default = 0.8 (80%)

        Returns
        -------
        doppler_noise (xarray.DataArray):
          Doppler noise level in units of m/s

        Notes
        -----
        Approximates bias from

        .. math:: \\sigma^{2}_{noise} = N * f_{c}

        where :math:`\\sigma_{noise}` is the bias due to Doppler noise,
        :math:`N` is the constant variance or spectral density, and :math:`f_{c}`
        is the characteristic frequency.

        The characteristic frequency is then found as

        .. math:: f_{c} = pct_fN * (f_{s}/2)

        where :math:`f_{s}/2` is the Nyquist frequency.


        Richard, Jean-Baptiste, et al. "Method for identification of Doppler noise
        levels in turbulent flow measurements dedicated to tidal energy." International
        Journal of Marine Energy 3 (2013): 52-64.

        Thiébaut, Maxime, et al. "Investigating the flow dynamics and turbulence at a
        tidal-stream energy site in a highly energetic estuary." Renewable Energy 195
        (2022): 252-262.
        """

        if not isinstance(psd, xr.DataArray):
            raise TypeError("`psd` must be an instance of `xarray.DataArray`.")
        if not isinstance(pct_fN, float) or not 0 <= pct_fN <= 1:
            raise ValueError("`pct_fN` must be a float within the range [0, 1].")
        array_size = len(psd.shape)
        if array_size >= 3:
            if (psd.shape[0] != 3) or (array_size > 3):
                raise Exception("PSD should be 2-dimensional (time, frequency)")

        # Characteristic frequency set to 80% of Nyquist frequency
        fN = self.fs / 2
        fc = pct_fN * fN

        # Get units right
        if psd.freq.units == "Hz":
            f_range = slice(fc, fN)
        else:
            f_range = slice(2 * np.pi * fc, 2 * np.pi * fN)

        # Noise floor
        N2 = psd.sel(freq=f_range) * psd.freq.sel(freq=f_range)
        noise_level = np.sqrt(N2.mean(dim="freq"))

        time_dim = psd.dims[-2]
        if array_size == 2:
            coords = {time_dim: psd[time_dim]}
        elif array_size == 3:
            coords = {"S": psd["S"], time_dim: psd[time_dim]}

        return xr.DataArray(
            noise_level.values,
            coords=coords,
            attrs={
                "units": "m/s",
                "long_name": "Doppler Noise Level",
                "description": "Doppler noise level calculated from PSD white noise",
            },
        )

    def _interp_noise(self, noise, time):
        """Return noise as a numpy array, interpolating to binned `time` if needed."""
        noise_time_dim = noise.dims[-1]
        if time.size != noise[noise_time_dim].size:
            return noise.interp({noise_time_dim: time}).values
        return noise.values

    def turbulence_intensity(self, U_mag, noise=0, thresh=0, detrend=False):
        """
        Calculate noise-corrected turbulence intensity (TI).

        Parameters
        ----------
        U_mag : xarray.DataArray
          Raw horizontal velocity magnitude (i.e., computed using
          :func:`U_mag <mhkit.dolfyn.velocity.Velocity.U_mag>`)
        noise : numeric
          Instrument noise level in same units as velocity. Typically
          found from the ADV's
          :func:`doppler_noise_level <mhkit.dolfyn.adv.turbulence.ADVBinner.doppler_noise_level>`.
          or ADCP's
          :func:`doppler_noise_level <mhkit.dolfyn.adp.turbulence.ADPBinner.doppler_noise_level>`.
          Default = None
        thresh : numeric
          Theshold below which TI will not be calculated
        detrend : bool
          Detrend the velocity data (True), or simply de-mean it
          (False), prior to computing TI.
          Default = False
        """

        if "xarray" in type(U_mag).__module__:
            U = U_mag.values
        if "xarray" in type(noise).__module__:
            noise = self._interp_noise(noise, self.mean(U_mag["time"].values))

        if detrend:
            up = self.detrend(U)
        else:
            up = self.demean(U)

        # Take RMS and subtract noise
        u_rms = np.sqrt(np.nanmean(up**2, axis=-1) - noise**2)
        u_mag = self.mean(U)

        ti = np.ma.masked_where(u_mag < thresh, u_rms / u_mag)

        dims = U_mag.dims
        coords = {}
        for nm in U_mag.dims:
            if "time" in nm:
                coords[nm] = self.mean(U_mag[nm].values)
            else:
                coords[nm] = U_mag[nm].values

        return xr.DataArray(
            ti.data.astype("float32"),
            coords=coords,
            dims=dims,
            attrs={
                "units": "1",
                "long_name": "Turbulence Intensity",
                "comment": f"TI was corrected from a noise level of {noise} m/s",
            },
        )

    def turbulent_kinetic_energy(self, veldat, noise=None, detrend=True):
        """
        Calculate the turbulent kinetic energy (TKE) (:math:`\\overline{u'u'}`,
        :math:`\\overline{v'v'}`, :math:`\\overline{w'w'}`).

        Parameters
        ----------
        veldat : xarray.DataArray
          Velocity data array from ADV or single beam from ADCP.
          The last dimension is assumed to be time.
        noise : float or array-like
          Instrument noise level in same units as velocity. Typically
          found from
          :func:`doppler_noise_level <mhkit.dolfyn.VelBinner.doppler_noise_level>`.
          Default = None
        detrend : bool
          Detrend the velocity data (True), or simply de-mean it (False),
          prior to computing TKE. Default = False

          Note: the PSD routines use detrend, so if you want to have the same
          amount of variance here as there use ``detrend=True``.

        Returns
        -------
        tke_vec : xarray.DataArray
          dataArray containing ``u'u'_``, ``v'v'_`` and ``w'w'_``
        """

        if "xarray" in type(veldat).__module__:
            vel = veldat.values
        if "xarray" in type(noise).__module__:
            noise = self._interp_noise(noise, self.mean(veldat[veldat.dims[-1]].values))

        if len(np.shape(vel)) > 2:
            raise ValueError(
                "This function is only valid for calculating TKE using "
                "velocity from an ADV or a single ADCP beam."
            )

        tke = xr.DataArray(
            ["upup_", "vpvp_", "wpwp_"],
            dims=["tke"],
            name="tke",
            attrs={
                "units": "1",
                "long_name": "Turbulent Kinetic Energy Vector Components",
                "coverage_content_type": "coordinate",
            },
        )

        # Calc TKE
        if detrend:
            out = np.nanmean(self.detrend(vel) ** 2, axis=-1)
        else:
            out = np.nanmean(self.demean(vel) ** 2, axis=-1)

        if "dir" in veldat.dims:
            # Subtract noise
            if noise is not None:
                if np.shape(noise)[0] != 3:
                    raise Exception(
                        "Noise should have same first dimension as velocity"
                    )
                out[0] -= noise[0] ** 2
                out[1] -= noise[1] ** 2
                out[2] -= noise[2] ** 2
            # Set coords
            dims = ["tke", "time"]
            coords = {"tke": tke, "time": self.mean(veldat.time.values)}
        else:
            # Subtract noise
            if noise is not None:
                if np.shape(noise) > np.shape(vel):
                    raise Exception(
                        "Noise should have same or fewer dimensions as velocity"
                    )
                out -= noise**2
            # Set coords
            dims = veldat.dims
            coords = {}
            for nm in veldat.dims:
                if "time" in nm:
                    coords[nm] = self.mean(veldat[nm].values)
                else:
                    coords[nm] = veldat[nm].values

        return xr.DataArray(
            out,
            dims=dims,
            coords=coords,
            attrs={
                "units": "m2 s-2",
                "long_name": "TKE Vector",
                "standard_name": "specific_turbulent_kinetic_energy_of_sea_water",
            },
        )

    def check_turbulence_cascade_slope(self, psd, freq_range=[6.28, 12.57]):
        """
        This function calculates the slope of the PSD, the power spectra
        of velocity, within the given frequency range. The purpose of this
        function is to check that the region of the PSD containing the
        isotropic turbulence cascade decreases at a rate of :math:`f^{-5/3}`.

        Parameters
        ----------
        psd : xarray.DataArray ([time,] freq)
          The power spectral density (1D or 2D)
        freq_range : iterable(2)
          The range over which the isotropic turbulence cascade occurs, in
          units of the psd frequency vector (Hz or rad/s).
          Default = [6.28, 12.57] rad/s

        Returns
        -------
        (m, b): tuple (slope, y-intercept)
          A tuple containing the coefficients of the log-adjusted linear
          regression between PSD and frequency

        Notes
        -----
        Calculates slope based on the `standard` formula for dissipation:

        .. math:: S(k) = \\alpha \\epsilon^{2/3} k^{-5/3} + N

        The slope of the isotropic turbulence cascade, which should be
        equal to :math:`k^{-5/3}` or :math:`f^{-5/3}`, where k and f are
        the wavenumber and frequency vectors, is estimated using linear
        regression with a log transformation:

        .. math:: log10(y) = m*log10(x) + b

        Which is equivalent to

        .. math:: y = 10^{b} x^{m}

        Where :math:`y` is S(k) or S(f), :math:`x` is k or f, :math:`m`
        is the slope (ideally -5/3), and :math:`10^{b}` is the intercept of
        :math:`y` at :math:`x^{m}=1'.
        """

        if not isinstance(psd, xr.DataArray):
            raise TypeError("`psd` must be an instance of `xarray.DataArray`.")
        if not hasattr(freq_range, "__iter__") or len(freq_range) != 2:
            raise ValueError("`freq_range` must be an iterable of length 2.")

        idx = np.where((freq_range[0] < psd.freq) & (psd.freq < freq_range[1]))
        idx = idx[0]

        x = np.log10(psd["freq"].isel(freq=idx))
        y = np.log10(psd.isel(freq=idx))

        y_bar = y.mean("freq")
        x_bar = x.mean("freq")

        # using the formula to calculate the slope and intercept
        n = np.sum((x - x_bar) * (y - y_bar), axis=0)
        d = np.sum((x - x_bar) ** 2, axis=0)

        m = n / d
        b = y_bar - m * x_bar

        return m, b

    def dissipation_rate_LT83(
        self,
        psd,
        U_mag,
        freq_range=[6.28, 12.57],
        k_constant=[0.5, 0.67, 0.67],
        noise=None,
    ):
        """
        Calculate the dissipation rate from the power spectral density of velocity.

        Parameters
        ----------
        psd : xarray.DataArray ([dir,] time, freq)
          The power spectral density. For an ADCP, this should be a single depth bin
          (range) from the vertical beam.
        U_mag : xarray.DataArray (time)
          The bin-averaged horizontal velocity (speed) [m/s]. For an ADCP, this should
           be from a single depth bin. U_mag can be computed using
          :func:`U_mag <mhkit.dolfyn.velocity.Velocity.U_mag>`
        freq_range : iterable(2)
          The range over which to integrate/average the spectrum, in units
          of the psd frequency vector (Hz or rad/s).
          Default = [6.28, 12.57] rad/s
        k_constant : float or iterable(3)
          Kolmogorov Constant (\\alpha in Notes section below) to use. If a
          three dimensional PSD is provided, \\alpha defaults to [0.5, 0.67, 0.67];
          i.e. 0.5 for the streamwise PSD and 0.67 for the transverse and vertical
          PSDs. If the PSD is provided for a single velocity direction, \\alpha is
          taken to be 0.5 unless otherwise specified.
        noise : float or array-like
          Instrument noise level in same units as velocity. Can be computed from the
          power spectral density and
          :func:`doppler_noise_level <mhkit.dolfyn.VelBinner.doppler_noise_level>`
          Default: None.

        Returns
        -------
        epsilon : xarray.DataArray ([dir,] time)
          dataArray of the dissipation rate

        Notes
        -----
        This uses the `standard` formula for dissipation:

        .. math:: S(k) = \\alpha \\epsilon^{2/3} k^{-5/3} + N

        where :math:`\\alpha is the Kolmogorov constant, `k` is wavenumber,
        `S(k)` is the turbulent kinetic energy spectrum, and `N' is the
        doppler noise level associated with the TKE spectrum.

        With :math:`k \\rightarrow \\omega / U`, then -- to preserve variance --
        :math:`S(k) = U S(\\omega)`, and so this becomes:

        .. math:: S(\\omega) = \\alpha \\epsilon^{2/3} \\omega^{-5/3} U^{2/3} + N

        With :math:`k \\rightarrow (2\\pi f) / U`, then

        .. math:: S(\\omega) = \\alpha \\epsilon^{2/3} f^{-5/3} (U/(2*\\pi))^{2/3} + N

        LT83 : Lumley and Terray, "Kinematics of turbulence convected
        by a random wave field". JPO, 1983, vol13, pp2000-2007.
        """

        if not isinstance(psd, xr.DataArray):
            raise TypeError("`psd` must be an instance of `xarray.DataArray`.")
        if len(U_mag.shape) != 1:
            raise Exception("U_mag should be 1-dimensional (time).")
        if not hasattr(freq_range, "__iter__") or len(freq_range) != 2:
            raise ValueError("`freq_range` must be an iterable of length 2.")

        array_size = len(psd.shape)
        if array_size >= 3:
            if (psd.shape[0] != 3) or (array_size > 3):
                raise Exception("PSD should be 2-dimensional (time, frequency)")

        # if the spectra are 1D, then the first dimension should be time (any length)
        if (psd.shape[0] != 3) and (np.size(k_constant) != 1):
            raise ValueError(
                "`k_constant` should be a single value. If using streamwise "
                "velocity, set to 0.5. Otherwise set to 0.67."
            )
        elif (psd.shape[0] == 3) and (np.size(k_constant) != 3):
            raise ValueError("`k_constant` should be an iterable of length 3.")

        if noise is not None:
            if np.shape(noise)[0] != np.shape(psd)[0]:
                raise Exception("Noise should have same first dimension as `psd`.")
        else:
            if array_size == 3:  # ADV
                noise = np.array([0, 0, 0])[:, None, None]
            else:  # ADCP
                noise = np.array(0)

        # Noise subtraction
        psd = psd.copy()
        if noise is not None:
            psd -= noise**2 / (self.fs / 2)
            psd = psd.where(psd > 0, np.min(np.abs(psd)) / 100)

        freq = psd.freq
        idx = np.where((freq_range[0] < freq) & (freq < freq_range[1]))
        idx = idx[0]

        # Interpolate U_mag to the same time dimension as PSD
        umag_time_dim = U_mag.dims[-1]
        psd_time_dim = psd.dims[-2]
        # If overlap is not 0%
        if psd[psd_time_dim].size != U_mag[umag_time_dim].size:
            U_mag = U_mag.interp({umag_time_dim: psd[psd_time_dim].values}).values
        else:
            U_mag = U_mag.values

        # Set the correct magnitude whether the frequency is in Hz or rad/s
        if freq.units == "Hz":
            U = U_mag / (2 * np.pi)
        else:
            U = U_mag

        # Set Kolmogorov constant
        a = np.array(k_constant)
        if psd.shape[0] == 3:
            a = a[:, None, None]  # stack properly
        else:
            a = np.squeeze(k_constant)

        # Calculate dissipation
        out = (psd.isel(freq=idx) * freq.isel(freq=idx) ** (5 / 3) / a).mean(
            axis=-1
        ) ** (3 / 2) / U

        return xr.DataArray(
            out,
            attrs={
                "units": "m2 s-3",
                "long_name": "TKE Dissipation Rate",
                "standard_name": "specific_turbulent_kinetic_energy_dissipation_in_sea_water",
                "description": "TKE dissipation rate calculated using "
                "the method from Lumley and Terray, 1983",
            },
        )
