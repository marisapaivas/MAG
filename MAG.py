import math as m
import numpy as np
import pandas as pd
from MEP_model import MEP_class
import matplotlib.pyplot as plt


## import the MEP model 
mep_instance= MEP_class()

mep_instance.importRegressionCoef()
mep_instance.importBaseline()
mep_instance.importXP()
mep_instance.importDelay()
mep_instance.importGain()

#Data for each muscle and velocity
muscles= ['TA', 'Sol', 'GasMed', 'GasLat']
muscles_ceinms= ['tib_ant_r','soleus_r','med_gas_r','lat_gas_r' ]
speed= 3.6 #from imu
elevation=0 #from imu????

signal= mep_instance.calc_MEP(elevation, speed) ## signals for all 4 muscles

TA= signal['TA']
Sol= signal['Sol']
GasMed= signal['GasMed']
GasLat= signal['GasLat']




