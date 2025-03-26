import pyproj
import skyfield
import skyfield.api
import numpy
import sgp4.api
import datetime
from pymap3d.aer import eci2aer
from pymap3d.eci import eci2ecef
from pymap3d.ecef import ecef2eci,geodetic2ecef,geodetic2eci
from pymap3d.sidereal import datetime2sidereal
import numpy as np

# 多普勒频移计算思路
# 1. 给定 地面站位置 (LLA)
# 2. 给定 卫星轨道数据 (TLE)
# 3. 确定时间
# 4. 计算给定时间下 卫星运动状态 (ECI)
# 5. 计算地面站 ECI 


# satelliteTLE = {
#     "name": "StarLink-1030",
#     "TLE_line1": "1 44735U 19074Y   24151.67073227  .00005623  00000+0  39580-3 0  9994",
#     "TLE_line2": "2 44735  53.0540 235.6876 0001395  85.6354 274.4795 15.06429209250797",
# }

# groundStation = {
#     "name": "Station",
#     "latitude_deg": 20.7634433315784,
#     "longitude_deg": 116.560494091634,
#     "altitude_m": 0,
# }


class GroundStation:
    def __init__(self,name,latitude_deg,longitude_deg,altitude_m):
        self.name = name
        self.latitude_deg = latitude_deg
        self.longitude_deg = longitude_deg
        self.altitude_m = altitude_m

    def get_ground_station_position_eci(self,timestamp: datetime.datetime):
        eci_pos = geodetic2eci(self.latitude_deg,self.longitude_deg,self.altitude_m,timestamp)
        return eci_pos

    def get_ground_station_position_ecef(self,timestamp: datetime.datetime):
        ecef_pos = geodetic2ecef(self.latitude_deg,self.longitude_deg,self.altitude_m)
        return ecef_pos

class Channel:
    def __init__(self,groundStation,satelliteTLE):
        self.groundStation = GroundStation(groundStation["name"],groundStation["latitude_deg"],groundStation["longitude_deg"],groundStation["altitude_m"])
        self.satelliteTLE = satelliteTLE
        self.satellite = sgp4.api.Satrec.twoline2rv(satelliteTLE["TLE_line1"],satelliteTLE["TLE_line2"])

    def calculate_normalized_doppler_frequency_shift_by_ecef(self,timestamp: datetime.datetime):
        c = 299792458
        eci_pos = self.groundStation.get_ground_station_position_eci(timestamp)
        gSta_pos_ecef = self.groundStation.get_ground_station_position_ecef(timestamp)
        jd, fr = sgp4.api.jday(timestamp.year,timestamp.month,timestamp.day,timestamp.hour,timestamp.minute,timestamp.second)
        e,r,v = self.satellite.sgp4(jd, fr)
        
        r = numpy.array(r) * 1e3
        v = numpy.array(v) * 1e3

        sat_pos_ecef = np.array(eci2ecef(r[0], r[1], r[2], timestamp))
        sat_vel_ecef = eci2ecef_velocity(v, timestamp)

        vector_pos_groundStation_to_satellite = sat_pos_ecef - np.array(gSta_pos_ecef)
        vector_pos_groundStation_to_satellite_unit = vector_pos_groundStation_to_satellite / numpy.linalg.norm(vector_pos_groundStation_to_satellite)
        vector_vel_groundStation_to_satellite = numpy.dot(vector_pos_groundStation_to_satellite_unit, sat_vel_ecef)

        
        normalized_doppler_frequency_shift = -1 * vector_vel_groundStation_to_satellite / c

        return normalized_doppler_frequency_shift
    
    def calculate_normalized_doppler_frequency_shift_by_eci(self,timestamp: datetime.datetime):
        c = 299792458
        eci_pos = self.groundStation.get_ground_station_position_eci(timestamp)
        jd, fr = sgp4.api.jday(timestamp.year,timestamp.month,timestamp.day,timestamp.hour,timestamp.minute,timestamp.second)
        e,r,v = self.satellite.sgp4(jd, fr)
        
        r = numpy.array(r) * 1e3
        v = numpy.array(v) * 1e3

        vector_pos_groundStation_to_satellite = np.array(r) - eci_pos
        vector_pos_groundStation_to_satellite_unit = vector_pos_groundStation_to_satellite / numpy.linalg.norm(vector_pos_groundStation_to_satellite)
        vector_vel_groundStation_to_satellite = numpy.dot(vector_pos_groundStation_to_satellite_unit, v)

        normalized_doppler_frequency_shift = -1 * vector_vel_groundStation_to_satellite / c

        return normalized_doppler_frequency_shift
    
    def calculate_elevation_groundStation_to_satellite(self,timestamp: datetime.datetime):
        # eci_pos = self.groundStation.get_ground_station_position_eci(timestamp)
        jd, fr = sgp4.api.jday(timestamp.year,timestamp.month,timestamp.day,timestamp.hour,timestamp.minute,timestamp.second)
        e,r,v = self.satellite.sgp4(jd, fr)

        r = numpy.array(r) * 1e3

        az,el,_range = eci2aer(r[0],r[1],r[2],self.groundStation.latitude_deg,self.groundStation.longitude_deg,self.groundStation.altitude_m,timestamp)
        
        return el


def eci2ecef_velocity(v_eci, time):
    """
    Convert velocity from ECI to ECEF using rotation matrix
    """
    gst = datetime2sidereal(time, 0.0)
    rot = np.array([[np.cos(gst), np.sin(gst), 0],
                   [-np.sin(gst), np.cos(gst), 0],
                   [0, 0, 1]])
    return rot @ v_eci


