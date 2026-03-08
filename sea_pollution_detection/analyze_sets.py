# -*- coding: utf-8 -*-
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
        path="data/"+i+"/"+j+"/"+"2122"+j+".csv"
        dataraw[j]=makeDF(path)
#####################################################

patra=dataraw["patra"]
aigio=dataraw["aigio"]
aktio=dataraw["aktio"]
preveza=dataraw["preveza"]


def fullplot(place, delta=timedelta(days=1), freq="1d",value=readings,save=False, resample=False, showgaps=False):
    fig, ax=plt.subplots(6,1,dpi=400, figsize=(20,20))
    gaps=timegaps(place, delta)
    for i in range(6):
        ax[i].plot(place[value[i]])
        if resample:
            ax[i].plot(place[value[i]].resample(freq).mean(),label="resample")
        if showgaps:
            ax[i].scatter(gaps[value[i]].index,gaps[value[i]],color="r")
        ax[i].grid(False)
        ax[i].set_xlabel(sensedict[value[i]])
        ax[i].set_facecolor('white')
        plt.setp(ax[i].spines.values(), color='grey')
        ax[i].legend()
    if save:
        fig.savefig("fullplot"+str(time.time())[5:-8])

def outlierUPDATE(series, anomalies):
    cleanseries=series.copy()
    anomalies=anomalies.fillna(False)
    for i in range(len(anomalies)):
        if anomalies.iloc[i]:
            for d in range(i,len(anomalies)):
                if not anomalies.iloc[d]:
                    cleanseries.iloc[i]=cleanseries.iloc[d]
                    break
    return cleanseries
def quickplot(series, save=False):
    fig,ax=plt.subplots(dpi=400,figsize=(20,5))
    ax.grid(False)
    ax.plot(series.index, series)
    
    ax.set_facecolor('white')
    plt.setp(ax.spines.values(), color='k')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    if save:
        fig.savefig("quickplot"+str(time.time())[-5:])
def anomalyplot(series, anomalies, save=False):
    anomalies=anomalies.fillna(False)
    fig,ax=plt.subplots(dpi=400,figsize=(20,5))
    ax.grid(False)
    ax.plot(series.index, series, zorder=0)
    ax.scatter(series.index[anomalies],series[anomalies], color="red", s=4, zorder=10)
    
    ax.set_facecolor('white')
    plt.setp(ax.spines.values(), color='k')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    if save:
        fig.savefig("anomalyplot"+str(time.time())[-5:])
def ISOForest(series, decision_function):
    outlier=pd.Series(decision_function)
    val=2*outlier.describe().loc["std"]+outlier.describe().loc["min"] #Value under which = outlier
    #change indicators to Boolean Mask
    outlier[outlier>val]=False 
    outlier[outlier<=val]=True
    #DateTime index for Boolean Mask
    outlier=outlier.set_axis(series.index)
    return outlier

def quickunivariate(series, alpha=2, c1=4, c2=4):
    GENESD=genESD(alpha=alpha)
    PERSIST=persist(c=c1)
    AUTOREG=autoreg(n_steps=7*2, step_size=24, c=c2)
    anomalies1=GENESD.fit_predict(series)
    anomalies2=PERSIST.fit_predict(series)
    anomalies3=AUTOREG.fit_predict(series)
    anomalyplot(series, anomalies1)
    anomalyplot(series, anomalies2)
    anomalyplot(series, anomalies3)


Detectors={"patra":{"TMP":[],
                    "CND":[],
                    "DO":[],
                    "PH":[],
                    "ORP":[],
                   "NTU":[]},
            "aigio":{"TMP":[],
                    "CND":[],
                    "DO":[],
                    "PH":[],
                    "ORP":[],
                    "NTU":[]},
            "aktio":{"TMP":[],
                    "CND":[],
                    "DO":[],
                    "PH":[],
                    "ORP":[],
                    "NTU":[]},
            "preveza":{"TMP":[],
                    "CND":[],
                    "DO":[],
                    "PH":[],
                    "ORP":[],
                    "NTU":[]}}


#PATRA:
#########################
patraNEW=patra.copy()

######################################
#############Temperature##############
TMPpersist=persist(c=20, side="both")
TMPpersist.fit(patra.TMP) #Very rough outlier estimate
#Add to detectors
Detectors["patra"]["TMP"].append(TMPpersist) 

######################################
#########Conductivity#################
CNDgenesd = genESD(alpha=0.1)

CNDanomaly=CNDgenesd.fit_predict(patra.CND) #Data correction
CNDnew=outlierUPDATE(patra.CND,CNDanomaly) #Data Update

X = np.array(CNDnew).reshape(-1, 1)
isf = IsolationForest(n_estimators=200,random_state=0, bootstrap=True, contamination=0.1,warm_start=True).fit(X)
idfDECISION=isf.decision_function(X)
outliers=ISOForest(CNDnew,idfDECISION)
CNDnew=outlierUPDATE(CNDnew, outliers) #Data Update

CNDautoreg = autoreg(n_steps=7*2, step_size=24, c=5.0)
CNDanomaly = CNDautoreg.fit_detect(CNDnew) 
CNDnew=outlierUPDATE(CNDnew, CNDanomaly) #Final Data Update

#Add to detectors
Detectors["patra"]["CND"].append(CNDgenesd)
Detectors["patra"]["CND"].append(isf)
"""
Use above as decfunc=CNDdetectors["patra"][1].decision_function(X) where 
X = np.array(CND).reshape(-1,1)
ISOForest(CND,decfunc) returns outliers
"""
Detectors["patra"]["CND"].append(CNDautoreg)

######################################
##############PH######################
PHpersist = persist(c=2.0, side='both')
PHanomalies = PHpersist.fit_detect(patra.PH)
PHnew=outlierUPDATE(patra.PH, PHanomalies) #Data Update

PHautoreg=autoreg(n_steps=7*2, step_size=24, c=4.0)
PHanomalies=PHautoreg.fit_detect(PHnew)
PHnew=outlierUPDATE(PHnew, PHanomalies) #Data Update

Detectors["patra"]["PH"].append(PHpersist)
Detectors["patra"]["PH"].append(PHautoreg)

######################################
#########Dissolved Oxygen#############
DOanomalies=genESD(alpha=3).fit_detect(patra.DO)
DOnew=outlierUPDATE(patra.DO, DOanomalies) #Data Update


#MultiVariate - DO&TMP with LOF
df=pd.DataFrame({"DO":DOnew,"TMP":patra.TMP})
DOoutlier = Outlier(LocalOutlierFactor(contamination=0.01))
DOanomalies = DOoutlier.fit_detect(df)
DOnew=outlierUPDATE(df.DO, DOanomalies)
#MultiVariate - DO&TMP with PCA
df.DO=DOnew
DOpca = PCA(k=1)
DOanomalies = DOpca.fit_detect(df)
DOnew=outlierUPDATE(df.DO, DOanomalies)

#MultiVariate - DO&PH with LOF
df=pd.DataFrame({"PH":PHnew,"DO":DOnew})
DOoutlierPH = Outlier(LocalOutlierFactor(contamination=0.01))
DOoutlierPH.fit(df)


Detectors["patra"]["DO"].append(DOoutlier)
Detectors["patra"]["DO"].append(DOoutlierPH)
Detectors["patra"]["DO"].append(DOpca)

######################################
##########Redox Potential#############

ORPpersist=persist(c=15, side="positive")
ORPanomalies=ORPpersist.fit_predict(patra.ORP)
ORPnew=outlierUPDATE(patra.ORP, ORPanomalies)

#MultiVariate - ORP&DO - Too many anomalies
df=pd.DataFrame({"DO":DOnew, "ORP":patra.ORP})
ORPpca = PCA(k=2)
ORPpca.fit(df)

Detectors["patra"]["ORP"].append(ORPpersist)
Detectors["patra"]["ORP"].append(ORPpca)

######################################
###############Turbidity##############
NTUpersist=persist(c=20,side="positive")
NTUanomaly=NTUpersist.fit_detect(patra.NTU)
NTUnew=outlierUPDATE(patra.NTU, NTUanomaly)

Detectors["patra"]["NTU"].append(NTUpersist)


patraNEW[["CND","PH","DO","ORP","NTU"]]=pd.DataFrame({"CND":CNDnew,"PH":PHnew, "DO":DOnew, "ORP":ORPnew, "NTU":NTUnew})



#Aigio:
#########################
aigioNEW=aigio.copy()

######################################
#############Temperature##############
TMPpersist=persist(c=20, side="both")
TMPpersist.fit(aigio.TMP) #Very rough outlier estimate
#Add to detectors

Detectors["aigio"]["TMP"].append(TMPpersist) 


######################################
#########Conductivity#################
CNDgenesd = genESD(alpha=0.1)

CNDanomaly=CNDgenesd.fit_predict(aigio.CND) #Data correction
CNDnew=outlierUPDATE(aigio.CND,CNDanomaly) #Data Update

X = np.array(CNDnew).reshape(-1, 1)
isf = IsolationForest(n_estimators=200,random_state=0, bootstrap=True, contamination=0.1,warm_start=True).fit(X)
idfDECISION=isf.decision_function(X)
outliers=ISOForest(CNDnew,idfDECISION)
CNDnew=outlierUPDATE(CNDnew, outliers) #Data Update

CNDautoreg = autoreg(n_steps=7*2, step_size=24, c=5.0)
CNDanomaly = CNDautoreg.fit_detect(CNDnew) 
CNDnew=outlierUPDATE(CNDnew, CNDanomaly) #Final Data Update

#Add to detectors
Detectors["aigio"]["CND"].append(CNDgenesd)
Detectors["aigio"]["CND"].append(isf)
"""
Use above as decfunc=CNDdetectors["patra"][1].decision_function(X) where 
X = np.array(CND).reshape(-1,1)
ISOForest(CND,decfunc) returns outliers
"""
Detectors["aigio"]["CND"].append(CNDautoreg)

######################################
##############PH######################
PHpersist = persist(c=2.0, side='both')
PHanomalies = PHpersist.fit_detect(aigio.PH)
PHnew=outlierUPDATE(aigio.PH, PHanomalies) #Data Update

PHautoreg=autoreg(n_steps=7*2, step_size=24, c=4.0)
PHanomalies=PHautoreg.fit_detect(PHnew)
PHnew=outlierUPDATE(PHnew, PHanomalies) #Data Update

Detectors["aigio"]["PH"].append(PHpersist)
Detectors["aigio"]["PH"].append(PHautoreg)

######################################
#########Dissolved Oxygen#############
DOanomalies=genESD(alpha=10).fit_detect(aigio.DO)
DOnew=outlierUPDATE(aigio.DO, DOanomalies) 

df=pd.DataFrame({"DO":DOnew,"TMP":aigio.TMP})
DOoutlier = Outlier(LocalOutlierFactor(contamination=0.02))
DOanomalies = DOoutlier.fit_detect(df)
DOnew=outlierUPDATE(df.DO, DOanomalies)
df.DO=DOnew

DOpca = PCA(k=1)
DOanomalies = DOpca.fit_detect(df)
DOnew=outlierUPDATE(df.DO, DOanomalies)

df=pd.DataFrame({"PH":PHnew,"DO":DOnew})
DOoutlierPH = Outlier(LocalOutlierFactor(contamination=0.01))
DOoutlierPH.fit(df)

Detectors["aigio"]["DO"].append(DOoutlier)
Detectors["aigio"]["DO"].append(DOoutlierPH)
Detectors["aigio"]["DO"].append(DOpca)

######################################
##########Redox Potential#############
df=pd.DataFrame({"DO":DOnew, "ORP":aigio.ORP})
ORPoutlier = Outlier(LocalOutlierFactor(contamination=0.02))
ORPanomalies=ORPoutlier.fit_predict(df)
ORPnew=outlierUPDATE(aigio.ORP, ORPanomalies)

Detectors["aigio"]["ORP"].append(ORPoutlier)

######################################
###############Turbidity##############
NTUpersist=persist(c=20,side="positive")
NTUanomaly=NTUpersist.fit_detect(aigio.NTU)
NTUnew=outlierUPDATE(aigio.NTU, NTUanomaly)

Detectors["aigio"]["NTU"].append(NTUpersist)


aigioNEW[["CND","PH","DO","ORP","NTU"]]=pd.DataFrame({"CND":CNDnew,"PH":PHnew, "DO":DOnew, "ORP":ORPnew, "NTU":NTUnew})


#aktio

aktioNEW=aktio.copy()

TMPautoreg=autoreg(n_steps=7*2, step_size=24, c=5)
TMPanomalies=TMPautoreg.fit_detect(aktio.TMP)
TMPnew=outlierUPDATE(aktio.TMP, TMPanomalies)

Detectors["aktio"]["TMP"].append(TMPautoreg)

######################################
#########Conductivity#################
CNDautoreg=autoreg(n_steps=7*2, step_size=24, c=5)
CNDanomalies=CNDautoreg.fit_detect(aktio.CND)
CNDnew=outlierUPDATE(aktio.CND, CNDanomalies)

CNDpersist=persist(c=20)
CNDpersist.fit(CNDnew.loc["2022-04":])

Detectors["aktio"]["CND"].append(CNDautoreg)
Detectors["aktio"]["CND"].append(CNDpersist)

######################################
##############PH######################
PHgenesd=genESD(alpha=10)
PHnew=outlierUPDATE(aktio.PH, PHgenesd.fit_predict(aktio.PH))

PHautoreg=autoreg(n_steps=7*2, step_size=24, c=4)
PHanomalies=PHautoreg.fit_predict(PHnew)
PHnew=outlierUPDATE(PHnew, PHanomalies)

Detectors["aktio"]["PH"].append(PHautoreg)

######################################
#########Dissolved Oxygen#############
df=pd.DataFrame({"DO":aktio.DO,"TMP":aktio.TMP})
DOoutlier = Outlier(LocalOutlierFactor(contamination=0.01))
DOanomalies = DOoutlier.fit_detect(df)

DOnew=outlierUPDATE(aktio.DO, DOanomalies)

Detectors["aktio"]["DO"].append(DOoutlier)

######################################
##########Redox Potential#############
ORPpersist=persist(c=10)
ORPautoreg=autoreg(n_steps=7*2, step_size=24, c=10.0)
ORPan1=ORPpersist.fit_predict(aktio.ORP)
ORPan2=ORPautoreg.fit_predict(aktio.ORP)

ORPnew = outlierUPDATE(aktio.ORP, ORPan1 | ORPan2)

df=pd.DataFrame({"DO":aktio.DO,"ORP":ORPnew})
ORPoutlier = Outlier(LocalOutlierFactor(contamination=0.01))
ORPoutlier.fit(df)

Detectors["aktio"]["ORP"].append(ORPpersist)
Detectors["aktio"]["ORP"].append(ORPautoreg)
Detectors["aktio"]["ORP"].append(ORPoutlier)

######################################
###############Turbidity##############
anomalies=Detectors["patra"]["NTU"][0].detect(aktio.NTU)
NTUnew= outlierUPDATE(aktio.NTU, anomalies)

Detectors["aktio"]["NTU"].append(Detectors["patra"]["NTU"][0])

aktioNEW[["TMP","CND","PH","DO","ORP","NTU"]]=pd.DataFrame({"TMP":TMPnew,"CND":CNDnew,"PH":PHnew, "DO":DOnew, "ORP":ORPnew, "NTU":NTUnew})



prevezaNEW=preveza.loc[:"2022-06-12 00:00:00"].copy()

NTUoutliers=Detectors["aktio"]["NTU"][0].detect(prevezaNEW.NTU)
NTUnew=outlierUPDATE(prevezaNEW.NTU, NTUoutliers)

prevezaNEW["NTU"]=NTUnew
Detectors=pd.DataFrame(Detectors)
Detectors.preveza=Detectors.aktio

