# -*- coding: utf-8 -*-


from analyze_sets import *

"""
Created on Mon May 30 13:06:36 2022

@author: johny
"""
###########################################
#LIBRARIES AND CODE INIT
###########################################
import time
from datetime import timedelta
import numpy as np
import pandas as pd
import itertools as iters
import random
import os
from openpyxl.styles import Font
##############-PLOTTING-###################
import matplotlib.pyplot as plt
########-ANOMALY DETECTION-################
from adtk.detector import ThresholdAD
from adtk.visualization import plot


from adtk.detector import PersistAD as persist
from adtk.detector import GeneralizedESDTestAD as genESD
from adtk.detector import AutoregressionAD as autoreg
from adtk.detector import PcaAD as PCA
from adtk.detector import RegressionAD as Regression
from adtk.detector import OutlierDetector as Outlier
from adtk.detector import MinClusterDetector as minCluster


from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest
from adtk.data import validate_series

from statsmodels.tsa.tsatools import add_trend

plt.rcParams['figure.dpi'] = 400
plt.rcParams['text.usetex'] = True
plt.rc('text.latex', preamble=r'\usepackage{amsmath}')

###########################################
###########################################
def timegaps(dframe, gap=timedelta(hours=1)):
    """
    input dframe with timedate index
    find gaps larger than var: gap - default 1 hour
    gap is of type(): timedelta
    """
    t=pd.Series(dframe.index).diff()
    t=np.array(t[t>gap].index)
    gaps=np.sort(np.append(t,t-1))
    return dframe.iloc[gaps]

def makeDF(path):
    data=pd.read_csv(path,sep=",")
    data["Timestamp"]=pd.to_datetime(data["Timestamp"], format="%Y-%m-%d %H:%M:%S")    
    data=data.set_index("Timestamp")
    data=data[~data.index.duplicated(keep="last")] #remove duplicate 
    return data
###########################################

locations = {"GRC":["aktio","preveza","aigio","patra"]}
locdict={
    "aigio":"Aigio",
    "aktio":"Actium",
    "patra":"Patras",
    "preveza":"Preveza"}
locdict=pd.Series(list(locdict.values()), index=list(locdict.keys()))


sense = ["Timestamp",
         "TMP",
         "DEP",
         "CND",
         "DO",
         "DOP",
         "PH",
         "ORP",
         "TDS",
         "NTU",
         "SAL",
         "V"]
readings=["TMP",
         "CND",
         "DO",
         "PH",
         "ORP",
         "NTU"]


senseLONG = ["Date & Time",
             "Temperature",
             "Depth",
             "Conductivity",
             "Dissolved Oxygen",
             "Dissolved Oxygen Percentage",
             "pH",
             "Oxidation/Reduction Potential",
             "Total Dissolved Solids",
             "Nephelometric Turbidity Unit",
             "Salinity",
             "Battery Voltage"]
dataraw={}
#####################################################
#########INITIALIZE DATABASES, READ CSV##############
for i in locations.keys():
    for j in locations[i]:
        path="data/"+"REVIEWED"+j+".csv"
        dataraw[j]=makeDF(path)
#####################################################

patra=dataraw["patra"]
aigio=dataraw["aigio"]
aktio=dataraw["aktio"]
preveza=dataraw["preveza"]

def predictionPLOT(y, y_pred, y_actual, fwd):
    fig, ax=plt.subplots(dpi=400, figsize=(20,5))
    yplot=y.loc[y.index[-1]-timedelta(hours=fwd):].resample("1h").mean()
    ax.plot(yplot, label="y")
    ax.plot(y_pred, label="y predicted")
    ax.plot(y_actual, label="y actual")
    ax.legend()
    
    fig.savefig(str(random.random())[2:]+".png")


from sktime.forecasting.ets import AutoETS
from sktime.forecasting.base import ForecastingHorizon 
from sktime.forecasting.theta import ThetaForecaster
from sktime.utils.plotting import plot_series #Visualization
def ETS_PREDICT(back_horizon, fwd_hours, forecaster=AutoETS(auto=True, n_jobs=-1, sp=24)):
    
    idx=pd.date_range(back_horizon.index[-1].round("1h"), periods=fwd_hours, freq="1h")
    fh=ForecastingHorizon(idx, is_relative=False) 
    y=back_horizon.resample("1h").mean()
    
    forecaster.fit(y)  
    y_pred = forecaster.predict(fh=fh)
    return y_pred



for param in params:
    predictionPLOT(back_horizon[param], y_pred[param], y_actual[param], 48)

