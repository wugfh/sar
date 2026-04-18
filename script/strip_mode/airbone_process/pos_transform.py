import math
import numpy as np
from scipy.interpolate import CubicSpline, interp1d

# pos_reader.py
# GitHub Copilot


WGS84_A = 6378137.0
WGS84_F = 1 / 298.257223563
WGS84_E2 = 2 * WGS84_F - WGS84_F * WGS84_F


def time2sec(time):
    """convert echo recorded time (HHMMSS[.sss]) to GNSS seconds (numpy array)"""
    t = np.asarray(time, dtype=float)
    hours = np.floor(t / 1e4)
    minutes = np.floor((t - hours * 1e4) / 1e2)
    seconds = t - hours * 1e4 - minutes * 1e2

    timezone = 8
    hours = hours - timezone
    seconds = seconds + hours * 3600 + minutes * 60
    seconds = seconds + 18  # leap seconds (as in original code)
    if seconds.size and seconds.flat[0] < 0:
        seconds = seconds + 3600 * 24

    # fit a line to index positions where seconds change to get smooth monotonic seconds
    if seconds.size > 1:
        sec_change_idx = np.nonzero(np.diff(seconds) != 0)[0] + 1
        if sec_change_idx.size == 0:
            seconds_new = seconds
        else:
            # include the first index to stabilize fit if needed
            idxs = sec_change_idx
            poly = np.polyfit(idxs, seconds[idxs], 1)
            seconds_new = np.polyval(poly, np.arange(seconds.size))
    else:
        seconds_new = seconds
    return seconds_new


def lla_to_ecef(lat_deg, lon_deg, alt_m):
    lat = np.deg2rad(lat_deg)
    lon = np.deg2rad(lon_deg)
    a = WGS84_A
    e2 = WGS84_E2
    sin_lat = np.sin(lat)
    N = a / np.sqrt(1 - e2 * sin_lat ** 2)
    x = (N + alt_m) * np.cos(lat) * np.cos(lon)
    y = (N + alt_m) * np.cos(lat) * np.sin(lon)
    z = (N * (1 - e2) + alt_m) * np.sin(lat)
    return np.vstack((x, y, z)).T  # shape (N,3)


def ecef_to_ned(ecef, ecef_ref, lat_ref_deg, lon_ref_deg):
    # ecef and ecef_ref are (N,3) and (3,) respectively
    lat = math.radians(lat_ref_deg)
    lon = math.radians(lon_ref_deg)
    sin_lat = math.sin(lat)
    cos_lat = math.cos(lat)
    sin_lon = math.sin(lon)
    cos_lon = math.cos(lon)

    # Rotation matrix from ECEF to NED
    R = np.array([
        [-sin_lat * cos_lon, -sin_lat * sin_lon, cos_lat],
        [-sin_lon,            cos_lon,           0      ],
        [-cos_lat * cos_lon, -cos_lat * sin_lon, -sin_lat]
    ])
    delta = (ecef - ecef_ref).T  # (3, N)
    ned = R.dot(delta)  # (3, N)
    return ned  # rows: north, east, down

def GNSS2UTC(time_stamp):
 
    # 减去闰秒补偿 (GNSS时间转回UTC)
    seconds = np.asarray(time_stamp) - 18
    
    # 将总秒数拆解为小时、分钟、秒
    hours = np.floor(seconds / 3600)
    remaining_seconds = seconds - hours * 3600
    minutes = np.floor(remaining_seconds / 60)
    sec_only = remaining_seconds - minutes * 60
    
    # 转换为东八区北京时间
    timezone = 8
    hours = hours + timezone
    
    # 处理跨天情况（完全保留原Matlab的判断逻辑）
    if np.any(hours >= 24):
        hours = np.mod(hours, 24)
    if np.any(hours < 0):
        hours = hours + 24
    
    # 重新组合为HHMMSS数值格式
    frametime = hours * 10000 + minutes * 100 + np.floor(sec_only)
    
    # 标量输入返回原生数值，数组输入返回numpy数组
    return frametime.item() if frametime.size == 1 else frametime

class PosReader:
    """
    PosReader: read POS binary double file with 17 rows per record (MATLAB-style)
    Provides get_coords(time_array) -> forward, right, down (arrays aligned to time_array)
    """

    def __init__(self, file_name):
        self.fileName = file_name
        data = np.fromfile(file_name, dtype=np.float64)
        if data.size == 0:
            raise IOError(f"Cannot open or empty file: {file_name}")
        # MATLAB reshapes column-wise: use order='F'
        if data.size % 17 != 0:
            raise ValueError("File length is not a multiple of 17 doubles.")
        data = data.reshape((17, -1), order='F')
        # timestamp row is first row
        day_offset = math.floor(data[0, 0] / (24 * 3600)) * 24 * 3600
        self.timestamp = data[0, :] - day_offset
        # lat, lng, alt rows (MATLAB indices 2,3,4 -> python 1,2,3)
        self.lat = data[1, :]
        self.lng = data[2, :]
        self.alt = data[3, :]

    def get_coords(self, time):
        """
        time: array-like of echo times (HHMMSS[.sss] format numbers)
        returns forward, right, down arrays corresponding to provided time entries
        """
        # time = np.asarray(time)
        time = GNSS2UTC(time)
        tsec = time2sec(time)

        # find slice bounds
        slcb_idx = np.nonzero(self.timestamp >= (tsec[0] - 1))[0]
        if slcb_idx.size > 0:
            slcb_idx = slcb_idx[0]
        if slcb_idx != 0:
            slcb = max(0, slcb_idx - 1)
        else:
            slcb = 0
        greater = np.nonzero(self.timestamp > (tsec[-1] + 1))[0]
        if greater.size == 0:
            slce = self.timestamp.size - 1
        else:
            slce = min(self.timestamp.size - 1, greater[0])

        local_timestamp = self.timestamp[slcb:slce + 1].copy()
        # linear fit to timestamp to smooth jitter
        if local_timestamp.size >= 2:
            p = np.polyfit(np.arange(1, local_timestamp.size + 1), local_timestamp, 1)
            local_timestamp = np.polyval(p, np.arange(1, local_timestamp.size + 1))

        local_lat = self.lat[slcb:slce + 1]
        local_lng = self.lng[slcb:slce + 1]
        local_alt = self.alt[slcb:slce + 1]

        # convert radians->degrees for lla conversion (MATLAB did rad2deg before lla2ned)
        local_lat_deg = np.rad2deg(local_lat)
        local_lon_deg = np.rad2deg(local_lng)
        # reference point: middle index
        mid = int(math.floor(local_lat_deg.size / 2))
        if mid < 0:
            mid = 0
        ref_lat = local_lat_deg[mid]
        ref_lon = local_lon_deg[mid]
        ref_alt = local_alt[mid]

        ecef = lla_to_ecef(local_lat_deg, local_lon_deg, local_alt)
        ecef_ref = lla_to_ecef(np.array([ref_lat]), np.array([ref_lon]), np.array([ref_alt]))[0]
        ned = ecef_to_ned(ecef, ecef_ref, ref_lat, ref_lon)  # shape (3, N)
        local_north = ned[0, :]
        local_east = ned[1, :]
        # replicate MATLAB's local_height = ref_alt - down
        local_height = ref_alt - ned[2, :]

        # plane coords: [east; north]
        plane_east = local_east
        plane_north = local_north

        # heading: fit north vs east and derive direction vector [1; slope]
        if plane_east.size >= 2:
            slope = np.polyfit(plane_east, plane_north, 1)[0]
        else:
            slope = 0.0
        sign = np.sign(plane_east[-1] - plane_east[0]) if plane_east.size >= 2 else 1.0
        heading = sign * np.array([1.0, slope])
        heading = heading / np.sqrt(np.sum(heading ** 2))

        # calculate coordinates
        forward = heading[0] * plane_east + heading[1] * plane_north
        right = heading[1] * plane_east - heading[0] * plane_north
        down = -local_height

        # centering
        forward = forward - np.mean(forward)
        right = right - np.mean(right)

        # interpolation onto requested time values
        if local_timestamp.size == 1:
            f_forward = np.full_like(tsec, forward[0])
            f_right = np.full_like(tsec, right[0])
            f_down = np.full_like(tsec, down[0])
        else:
            # interp_kind = 'cubic' if max(1, local_timestamp.size - 1) >= 3 else 'linear'
            # fi = interp1d(local_timestamp, forward, kind=interp_kind, fill_value='extrapolate', assume_sorted=True)
            # ri = interp1d(local_timestamp, right, kind=interp_kind, fill_value='extrapolate', assume_sorted=True)
            # di = interp1d(local_timestamp, down, kind=interp_kind, fill_value='extrapolate', assume_sorted=True)
            fi = CubicSpline(local_timestamp, forward, extrapolate=True)
            ri = CubicSpline(local_timestamp, right, extrapolate=True)
            di = CubicSpline(local_timestamp, down, extrapolate=True)
            f_forward = fi(tsec)
            f_right = ri(tsec)
            f_down = di(tsec)

        return f_forward, f_right, f_down


# small helpers (kept for parity with original)
def gps2xyz(gpslong, gpslat, gpsalti):
    gpslong = np.asarray(gpslong)
    gpslat = np.asarray(gpslat)
    gpsalti = np.asarray(gpsalti)
    platform_lon = gpslong[0]
    platform_lat = gpslat[0]
    platform_alt = gpsalti[0]
    Re = 6378137.0
    e = 0.003352810929627
    Rs = math.sin(platform_lat)
    Rm = Re * (1 - 2 * e + 3 * e * Rs * Rs)
    Rn = Re * (1 + e * Rs * Rs)
    deast = (gpslong - platform_lon) * (Rn + platform_alt) * math.cos(platform_lat)
    dnorth = (gpslat - platform_lat) * (Rm + platform_alt)
    dalti = gpsalti
    return deast, dnorth, dalti


def smooth_coord(coord):
    coord = np.asarray(coord)
    mean_coord = np.mean(coord)
    if coord.size <= 1:
        return coord
    diff_coord = np.diff(coord)
    # simple moving average smoothing window 65 (pad to keep same length)
    window = 65
    if diff_coord.size < window:
        sm = np.convolve(diff_coord, np.ones(diff_coord.size) / diff_coord.size, mode='same')
    else:
        sm = np.convolve(diff_coord, np.ones(window) / window, mode='same')
    coord_new = np.concatenate(([0.0], np.cumsum(sm)))
    coord_new = coord_new - np.mean(coord_new) + mean_coord
    return coord_new