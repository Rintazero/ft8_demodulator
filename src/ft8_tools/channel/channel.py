import pyproj
import skyfield
import skyfield.api
import numpy
import sgp4.api
import datetime

satelliteTLE = {
    "name": "StarLink-1030",
    "TLE_line1": "1 44735U 19074Y   24151.67073227  .00005623  00000+0  39580-3 0  9994",
    "TLE_line2": "2 44735  53.0540 235.6876 0001395  85.6354 274.4795 15.06429209250797",
}

jd, fr = sgp4.api.jday(2024, 5, 31, 16, 5, 51)
timestamp = datetime.datetime(2024, 5, 31, 16, 5, 51)

groundStation = {
    "name": "Station",
    "latitude_deg": 0.0002*1e5,
    "longitude_deg": 0.0012*1e5,
    "altitude_m": 0,
}

class GroundStation:
    def __init__(self,name,latitude_deg,longitude_deg,altitude_m):
        self.name = name
        self.latitude_deg = latitude_deg
        self.longitude_deg = longitude_deg
        self.altitude_m = altitude_m

class Channel:
    def __init__(self,groundStation,satelliteTLE):
        self.groundStation = groundStation
        self.satelliteTLE = satelliteTLE
        self.satellite = sgp4.api.Satrec.twoline2rv(satelliteTLE["TLE_line1"],satelliteTLE["TLE_line2"])

    def get_ground_station_position_eci(self,timestamp):
        station = skyfield.api.Topos(latitude_degrees=self.groundStation["latitude_deg"],longitude_degrees=self.groundStation["longitude_deg"],elevation_m=self.groundStation["altitude_m"])
        ts = skyfield.api.load.timescale()
        # time = ts.utc(timestamp.year,timestamp.month,timestamp.day,timestamp.hour,timestamp.minute,timestamp.second)
        eci_pos = station.at(timestamp).position.m
        return eci_pos
    
channel = Channel(groundStation,satelliteTLE)
eci_pos = channel.get_ground_station_position_eci(timestamp)
print(eci_pos)

