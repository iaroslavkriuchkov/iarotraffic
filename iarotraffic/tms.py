from pyexpat.errors import XML_ERROR_XML_DECL
import numpy as np
from pandas import Series
from . import traffic
from pystoned import wCQER, CQER
from pystoned.plot import plot2d
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

class TrafficMeasurmentStationFinland:
    """
    A class to represent a Traffic Measurement Station.

    ...

    Attributes
    ----------
    id : int
        Traffic measurement station id in the Finnish Road Network.
    day_number : int
        Ordinal date (1-366, taking into account the leap years, January 1 = 1), when the data was collected.
    date : datetime.date
        Day, when the data was collected. Converted from ``day_number``.
    year : int
        Year, when the data was collected.
    yearshort : int
        Last two digits of the ``year``.
    hour_from : int
        Hour, from which the data is collected.
    hour_to : int
        Hour, till which the data is collected (including the hour). Should be >= ``hour_to``.
    """

    def __init__(self, tms_id: int, year: int, day: int, direction: int, hour_from: int = 6, hour_to: int = 20):
        
        #if hour_to > hour_from: 
        #    raise ValueError("Hour, till which the data is collected, is smaller, then hour, from which data is collected. Check hour_to and hour_from parameters.")
        # Traffic Measurement Station id
        self.id = tms_id

        # Date and time information
        self.year = year
        self.yearshort = repr(year)[-2:]
        self.day_number = day
        self.date = traffic.day_to_date(self.year, self.day_number)
        self.hour_from = hour_from
        self.hour_to = hour_to

        # Vehicle positionging
        self.direction = direction

    def load_raw_data(self):
        self.raw_data = traffic.download_lam_day_report(str(self.id), self.year, self.day_number, self.direction, self.hour_from, self.hour_to)

    def aggregate(self, aggregation_time_period: int = 1):
        self.aggregated_data = traffic.fscalc(self.raw_data, aggregation_time_period)
        self.x_aggregated = self.aggregated_data.density
        self.y_aggregated = self.aggregated_data.flow

    def bag(self, gridsize_x: int = 70, gridsize_y: int = 400):
        self.bagged_data = traffic.bagging(self.aggregated_data, gridsize_x, gridsize_y)
        self.x_bagged = self.bagged_data.centroid_density
        self.y_bagged = self.bagged_data.centroid_flow
        self.weight = self.bagged_data.weight

    def weighted_model(self, tau_list: list = [0.2, 0.5, 0.75, 0.95]):
        self.bagged_model = []
        for tau in tau_list:
            model = wCQER.wCQR(y=self.y_bagged, x=self.x_bagged, w=self.weight, tau=tau)
            model.__model__.beta.setlb(None)
            model.optimize()
            self.bagged_model.append(model)

    def non_weighted_model(self, tau_list: list = [0.2, 0.5, 0.75, 0.95]):
        self.aggregation_model = []
        for tau in tau_list:
            model = CQER.CQR(y=self.y_aggregated, x=self.x_aggregated, tau=tau)
            model.__model__.beta.setlb(None)
            model.optimize("OPT_LOCAL")
            self.aggregated_model.append(model)

    def plot_aggregated(self, save: bool = False):
        plt.clf()
        figure(figsize=(10, 10), dpi=400)
        plt.scatter(self.x_aggregated,self.y_aggregated, marker='x', c='black', label="Aggregated data")
        plt.xlabel("Density [veh/km]")
        plt.ylabel("Flow [veh/h]")
        plt.legend()
        if save is True: 
            plt.savefig("aggregated.png")

    def plot_bagged(self, save: bool = False):
        plt.clf()
        figure(figsize=(10, 10), dpi=400)
        plt.scatter(self.x_bagged,self.y_bagged, marker='x', c='black', label="Bagged data" )
        plt.xlabel("Density [veh/km]")
        plt.ylabel("Flow [veh/h]")
        plt.legend()
        if save is True: 
            plt.savefig("bagged.png")

    def plot_weighted_bagged(self, save: bool = False):
        plt.clf()
        figure(figsize=(10, 10), dpi=400)
        plt.scatter(self.x_bagged,self.y_bagged, marker='o', c='black', s=self.weight*10000, label="Bagged data with weighted representation" )
        plt.xlabel("Density [veh/km]")
        plt.ylabel("Flow [veh/h]")
        plt.legend()
        if save is True: 
            plt.savefig("bagged_weighted_rep.png")

    def plot_weighted_model(self, weighted_rep: bool = True, save: bool = False):
        plt.clf()
        figure(figsize=(10, 10), dpi=400)

        if weighted_rep is True:
            plt.scatter(self.x_bagged,self.y_bagged, marker='o', c='black', s=self.weight*10000, label="Bagged data with weighted representation" )
        else:
            plt.scatter(self.x_bagged,self.y_bagged, marker='x', c='black', label="Bagged data" )

        for model in self.bagged_model:
            x = np.array(model.x).T
            y = np.array(model.y).T
            yhat = np.array(model.get_frontier()).T
            data = (np.stack([x, y, yhat], axis=0)).T

            # sort
            data = data[np.argsort(data[:, 0])].T
            x, y, f = data[0], data[1], data[2]
            plt.plot(x, f, c='g')

        if save is True: 
            plt.savefig("weighted_model.png")
    
    def plot_non_weighted_model(self, save: bool = False):
        if save is True: 
            plt.savefig("non_weighted_model.png")