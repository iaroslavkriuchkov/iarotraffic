import numpy
from pandas import Series
from . import traffic

class TrafficMeasurmentStationFinland:
    """
    A class to represent a Traffic Measurement Station.

    ...

    Attributes
    ----------
    id : int
        id in the Finnish Road Network
    day_number : int
        ordinal date (1-366, taking into account the leap years, January 1 = 1), when the data was collected
    date : datetime.date
        day, when the data was collected. Converted from ``day_number``.
    year : int
        year, when the data was collected
    yearshort : int
        last two digits of the ``year``
    hour_from : int
        hour, from which the data is collected
    hour_to : int
        hour, till which the data is collected (including the hour). Should be >= ``hour_to``
    """

    def __init__(self, id: int, day: int, year: int, hour_from: int = 6, hour_to: int = 20):
        
        # Traffic Measurement Station id
        self.id = id

        # Date and time information
        self.year = year
        self.yearshort = repr(year)[-2:]
        self.day_number = day
        self.date = traffic.day_to_date(self.year, self.day_number)
        self.hour_from = hour_from
        self.hour_to = hour_to

    def get_data(self, density: Series, flow: Series):
        self.density = numpy.array(density)
        self.flow = numpy.array(flow)
