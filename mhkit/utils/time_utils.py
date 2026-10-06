"""
This module provides utility functions for converting datetime formats
from MATLAB and Excel to Python datetime formats.

Functions:
----------
- matlab_to_datetime: Converts MATLAB datenum format to Python datetime.
- excel_to_datetime: Converts Excel datenum format to Python datetime.
"""

import warnings
from typing import Union, List
from datetime import datetime, timedelta, timezone
import numpy as np
import pandas as pd
import xarray as xr

# pylint: disable=unused-import
from pecos.utils import index_to_datetime
from .binning_tools.tools import fillgaps


def matlab_to_datetime(
    matlab_datenum: Union[np.ndarray, list, float, int], to_pandas: bool = True
) -> Union[np.ndarray, pd.DatetimeIndex]:
    """
    Convert MATLAB datenum format to Python datetime

    Parameters
    ------------
    matlab_datenum : numpy array
        MATLAB datenum to be converted

    Returns
    ---------
    time : DateTimeIndex
        Python datetime values
    """
    # Check data types
    try:
        matlab_datenum = np.array(matlab_datenum, ndmin=1)
    except (TypeError, ValueError) as e:
        raise TypeError(f"Error converting to numpy array: {e}") from e
    if not isinstance(matlab_datenum, np.ndarray):
        raise TypeError(f"data must be of type np.ndarray. Got: {type(matlab_datenum)}")

    # Pre-allocate
    time = []
    # loop through dates and convert
    for i, t in enumerate(matlab_datenum):
        day = datetime.fromordinal(int(t))
        dayfrac = timedelta(days=t % 1) - timedelta(days=366)
        time.append(day + dayfrac)

        # Datenum is precise down to 100 microseconds - add difference to round
        us = int(round(time[i].microsecond / 100, 0)) * 100
        time[i] = time[i].replace(microsecond=time[i].microsecond) + timedelta(
            microseconds=us - time[i].microsecond
        )

    time = np.array(time)
    if to_pandas:
        time = pd.to_datetime(time)
    return time


def datetime_to_matlab(dt: List[datetime]):
    """
    Convert list of datetime objects to MATLAB datenum

    Parameters
    ----------
    dt : datetime.datetime
      List of datetime objects

    Returns
    -------
    time : float
      List of timestamps in MATLAB datnum format
    """

    time = []
    for t in dt:
        mdn = t + timedelta(days=366)
        frac_seconds = (t - datetime(t.year, t.month, t.day, 0, 0, 0)).seconds / (
            24 * 60 * 60
        )
        frac_microseconds = t.microsecond / (24 * 60 * 60 * 1000000)
        time.append(mdn.toordinal() + frac_seconds + frac_microseconds)

    return time


def excel_to_datetime(
    excel_num: Union[np.ndarray, list, float, int],
) -> pd.DatetimeIndex:
    """
    Convert Excel datenum format to Python datetime

    Parameters
    ------------
    excel_num : numpy array
        Excel datenums to be converted

    Returns
    ---------
    time : DateTimeIndex
        Python datetime values
    """
    # Check data types
    try:
        excel_num = np.array(excel_num)
    except (TypeError, ValueError) as e:
        raise TypeError(f"Error converting to numpy array: {e}") from e
    if not isinstance(excel_num, np.ndarray):
        raise TypeError(f"excel_num must be of type np.ndarray. Got: {type(excel_num)}")

    # Convert to datetime
    time = pd.to_datetime("1899-12-30") + pd.to_timedelta(excel_num, "D")

    return time


def epoch_to_dt64(ep_time: Union[np.ndarray, xr.DataArray, int]):
    """
    Convert from epoch time (seconds since 1/1/1970 00:00:00) to
    numpy.datetime64 array

    Parameters
    ----------
    ep_time : xarray.DataArray
      Time coordinate data-array or single time element

    Returns
    -------
    time : numpy.datetime64
      The converted datetime64 array
    """

    # assumes t0=1970-01-01 00:00:00
    out = np.array(ep_time.astype("int")).astype("datetime64[s]")
    out = out + ((ep_time % 1) * 1e9).astype("timedelta64[ns]")
    return out


def dt64_to_epoch(dt64: Union[np.ndarray, xr.DataArray, np.datetime64]):
    """
    Convert numpy.datetime64 array to epoch time
    (seconds since 1/1/1970 00:00:00)

    Parameters
    ----------
    dt64 : numpy.datetime64
      Single or array of datetime64 object(s)

    Returns
    -------
    time : float
      Epoch time (seconds since 1/1/1970 00:00:00)
    """

    return dt64.astype("datetime64[ns]").astype("float") / 1e9


def datetime_to_dt64(dt: Union[np.ndarray, xr.DataArray, datetime]):
    """
    Convert numpy.datetime64 array to list of datetime objects

    Parameters
    ----------
    time : datetime.datetime
      The converted datetime object

    Returns
    -------
    dt64 : numpy.datetime64
      Single or array of datetime64 object(s)
    """

    return np.array(dt).astype("datetime64[ns]")


def dt64_to_datetime(dt64: Union[np.ndarray, xr.DataArray, np.datetime64]):
    """
    Convert numpy.datetime64 array to list of datetime objects

    Parameters
    ----------
    dt64 : numpy.datetime64
      Single or array of datetime64 object(s)

    Returns
    -------
    time : datetime.datetime
      The converted datetime object
    """

    return epoch_to_datetime(dt64_to_epoch(dt64))


def epoch_to_datetime(
    ep_time: Union[np.ndarray, xr.DataArray, int],
    offset_hr: int = 0,
    to_str: bool = False,
):
    """
    Convert from epoch time (seconds since 1/1/1970 00:00:00) to a list
    of datetime objects

    Parameters
    ----------
    ep_time : xarray.DataArray
      Time coordinate data-array or single time element
    offset_hr : int
      Number of hours to offset time by (e.g. UTC -7 hours = PDT)
    to_str : logical
      Converts datetime object to a readable string

    Returns
    -------
    time : datetime.datetime
      The converted datetime object or list(strings)

    Notes
    -----
    The specific time instance is set during deployment, usually sync'd to the
    deployment computer. The time seen by DOLfYN is in the timezone of the
    deployment computer, which is unknown to DOLfYN.
    """

    try:
        ep_time = ep_time.values
    except AttributeError:
        pass

    if isinstance(ep_time, (np.ndarray)) and ep_time.ndim == 0:
        ep_time = [ep_time.item()]
    elif not isinstance(ep_time, (np.ndarray, list)):
        ep_time = [ep_time]

    if offset_hr != 0:
        delta = timedelta(hours=offset_hr)
        time = [
            datetime.fromtimestamp(t, timezone.utc).replace(tzinfo=None) + delta
            for t in ep_time
        ]
    else:
        time = [
            datetime.fromtimestamp(t, timezone.utc).replace(tzinfo=None)
            for t in ep_time
        ]

    if to_str:
        time = datetime_to_str(time)

    return time


def datetime_to_str(
    dt: Union[np.ndarray, xr.DataArray, datetime], format_str: Union[None, str] = None
):
    """
    Convert list of datetime objects to legible strings

    Parameters
    ----------
    dt : datetime.datetime
      Single or list of datetime object(s)
    format_str : string
      Timestamp string formatting. Default is '%Y-%m-%d %H:%M:%S.%f'
      See datetime.strftime documentation for timestamp string formatting.

    Returns
    -------
    time : string
      Converted timestamps
    """

    if format_str is None:
        format_str = "%Y-%m-%d %H:%M:%S.%f"

    if not isinstance(dt, list):
        dt = [dt]

    return [t.strftime(format_str) for t in dt]


def datetime_to_epoch(dt: Union[np.ndarray, xr.DataArray, int]):
    """
    Convert list of datetime objects to epoch time

    Parameters
    ----------
    dt : datetime.datetime
      Single or list of datetime object(s)

    Returns
    -------
    time : float
      Datetime converted to epoch time (seconds since 1/1/1970 00:00:00)
    """

    if not isinstance(dt, (list, np.ndarray)):
        dt = [dt]

    return [t.replace(tzinfo=timezone.utc).timestamp() for t in dt]


def _fill_time_gaps(
    epoch: Union[np.ndarray, xr.DataArray, list, int, float], sample_rate_hz: float
):
    """
    Fill gaps (NaN values) in the timeseries by simple linear
    interpolation.  The ends are extrapolated by stepping
    forward/backward by 1/sample_rate_hz.
    """

    # epoch is seconds since 1970
    dt = 1.0 / sample_rate_hz
    epoch = fillgaps(epoch)
    if np.isnan(epoch[0]):
        i0 = np.nonzero(~np.isnan(epoch))[0][0]
        delta = np.arange(-i0, 0, 1) * dt
        epoch[:i0] = epoch[i0] + delta
    if np.isnan(epoch[-1]):
        # Search backward through the array to get the 'negative index'
        ie = -np.nonzero(~np.isnan(epoch[::-1]))[0][0] - 1
        delta = np.arange(1, -ie, 1) * dt
        epoch[(ie + 1) :] = epoch[ie] + delta

    return epoch


## Function deprecations
def matlab2date(*args, **kwargs):
    """
    Deprecated function. Use `matlab_to_datetime` instead.
    """
    warnings.warn(
        "The 'matlab2date' function was renamed to 'matlab_to_datetime' "
        "and will be dropped in a future release.",
        DeprecationWarning,
    )
    return matlab_to_datetime(*args, **kwargs)


def date2matlab(*args, **kwargs):
    """
    Deprecated function. Use `datetime_to_matlab` instead.
    """
    warnings.warn(
        "The 'date2matlab' function was renamed to 'datetime_to_matlab' "
        "and will be dropped in a future release.",
        DeprecationWarning,
    )
    return matlab_to_datetime(*args, **kwargs)


def epoch2dt64(*args, **kwargs):
    """
    Deprecated function. Use `epoch_to_dt64` instead.
    """
    warnings.warn(
        "The 'epoch2dt64' function was renamed to 'epoch_to_dt64' "
        "and will be dropped in a future release.",
        DeprecationWarning,
    )
    return epoch_to_dt64(*args, **kwargs)


def dt642epoch(*args, **kwargs):
    """
    Deprecated function. Use `dt64_to_epoch` instead.
    """
    warnings.warn(
        "The 'dt642epoch' function was renamed to 'dt64_to_epoch' "
        "and will be dropped in a future release.",
        DeprecationWarning,
    )
    return dt64_to_epoch(*args, **kwargs)


def date2dt64(*args, **kwargs):
    """
    Deprecated function. Use `datetime_to_dt64` instead.
    """
    warnings.warn(
        "The 'date2dt64' function was renamed to 'datetime_to_dt64' "
        "and will be dropped in a future release.",
        DeprecationWarning,
    )
    return datetime_to_dt64(*args, **kwargs)


def dt642date(*args, **kwargs):
    """
    Deprecated function. Use `dt64_to_datetime` instead.
    """
    warnings.warn(
        "The 'dt642date' function was renamed to 'dt64_to_datetime' "
        "and will be dropped in a future release.",
        DeprecationWarning,
    )
    return dt64_to_datetime(*args, **kwargs)


def epoch2date(*args, **kwargs):
    """
    Deprecated function. Use `epoch_to_datetime` instead.
    """
    warnings.warn(
        "The 'epoch2date' function was renamed to 'epoch_to_datetime' "
        "and will be dropped in a future release.",
        DeprecationWarning,
    )
    return epoch_to_datetime(*args, **kwargs)


def date2str(*args, **kwargs):
    """
    Deprecated function. Use `datetime_to_str` instead.
    """
    warnings.warn(
        "The 'date2str' function was renamed to 'datetime_to_str' "
        "and will be dropped in a future release.",
        DeprecationWarning,
    )
    return datetime_to_str(*args, **kwargs)


def date2epoch(*args, **kwargs):
    """
    Deprecated function. Use `datetime_to_epoch` instead.
    """
    warnings.warn(
        "The 'dt642epoch' function was renamed to 'datetime_to_epoch' "
        "and will be dropped in a future release.",
        DeprecationWarning,
    )
    return datetime_to_epoch(*args, **kwargs)
