#include <fstream>
#include <sstream>
#include <string>
#include <stdlib.h>
#include <iostream>
#include <math.h>
#include <locale.h>
#include "include/MEP_class.h"

import locale
import pandas as pd
import io 
import numpy as np
import math as m
from scipy.interpolate import interp1d



class MEP_class:
    reg_coef=[]
    #gaussXP=[[0 for _ in range(200)] for _ in range(15)] 
    weightings_baseline=[] 
    weightings = [[0 for _ in range(4)] for _ in range(15)]    
    mep=[[0 for _ in range(15)] for _ in range(200)] 
    #print(len(mep))
    def __init__(self):
        self.importRegressionCoef()
        self.importXP()
        self.importBaseline()
        locale.setlocale(locale.LC_NUMERIC, "C")
    
    #import and store data    
    @staticmethod
    def importRegressionCoef():
        try:
            regression_path=r"C:\Users\marisa\OneDrive - University of Twente\Master Thesis\Code\regression.txt"
            regression_coef=pd.read_table(regression_path, sep=',', header=None)

            MEP_class.reg_coef = regression_coef.astype(float).values.tolist()  #converting the data in the file to float type 
        except FileNotFoundError:
            print('Error: File Not found COEF')
        except pd.errors.ParserError:
            print('Error: Incorrect file type')
            
    def importXP(self):
        try:
            gaussXP_path=r"C:\Users\marisa\OneDrive - University of Twente\Master Thesis\Code\gaussXP.txt"
            gauss_XP=pd.read_table(gaussXP_path, sep=',', header=None)

            MEP_class.gaussXP = gauss_XP.astype(float).values.tolist()  #converting the data in the file to float type 
            num_rows, num_cols = gauss_XP.shape
        except FileNotFoundError:
            print('Error: File Not found XP')
        except pd.errors.ParserError:
            print('Error: Incorrect file type')
            
    def importBaseline(self):
        try:
            baseline_path=r"C:\Users\marisa\OneDrive - University of Twente\Master Thesis\Code\baseline.txt"
            wBaseline= pd.read_table(baseline_path, sep=',', header=None)

            MEP_class.weightings_baseline = wBaseline.astype(float).values.tolist()  #converting the data in the file to float type 
            #print(MEP_class.weightings_baseline)
        except FileNotFoundError:
            print('Error: File Not found BASELINE')
        except pd.errors.ParserError:
            print('Error: Incorrect file type')
    
    def importDelay(self):
        try:
            delay_path = r"C:\Users\marisa\OneDrive - University of Twente\Master Thesis\Code\MAG\per_muscle_velocity_delays.txt"
            df = pd.read_csv(delay_path)  # expects columns: muscle,velocity,lag_samples

            # Basic cleaning / types
            required = {'muscle', 'velocity', 'lag_samples'}
            if not required.issubset(df.columns):
                raise ValueError(f"Delay file missing columns. Found {df.columns.tolist()}")

            df = df.dropna(subset=['muscle', 'velocity', 'lag_samples']).copy()
            df['velocity'] = df['velocity'].astype(float)
            df['lag_samples'] = df['lag_samples'].astype(int)

            # Nested dict: {muscle: {velocity: lag}}
            self.per_muscle_velocity_delays = {
                m: dict(zip(g['velocity'], g['lag_samples']))
                for m, g in df.groupby('muscle')
            }

            # Also build arrays per muscle for fast interpolation
            df = df.sort_values(['muscle', 'velocity'])
            self._delay_lookup = {
                m: (g['velocity'].to_numpy(dtype=float),
                    g['lag_samples'].to_numpy(dtype=float))
                for m, g in df.groupby('muscle')
            }


        except FileNotFoundError:
            print('Error: File Not Found DELAY')
        except pd.errors.ParserError:
            print('Error: Incorrect file type for DELAY')
        except Exception as e:
            print(f"Error reading DELAY: {e}")
            
    def importGain(self):
        try:
            path = r"C:\Users\marisa\OneDrive - University of Twente\Master Thesis\Code\ga\Results\Condition5\optimization_best_solutions_trial25.csv"
            df = pd.read_csv(path)

            cols = ['Gain_1', 'Gain_2', 'Gain_3', 'Gain_4']
            # basic sanity check
            for c in cols:
                if c not in df.columns:
                    raise KeyError(f"Missing column: {c}")

            # take the first row (best solution) as floats
            gains = df.loc[df.index[0], cols].astype(float).tolist()

            # store as a simple list: [g1, g2, g3, g4]
            MEP_class.gains = gains
            # print(MEP_class.gains)  # optional

        except FileNotFoundError:
            print('Error: File not found (GAIN)')
        except pd.errors.ParserError:
            print('Error: Incorrect file type (GAIN)')
        except (KeyError, IndexError) as e:
            print(f'Error: Problem reading gain columns/rows -> {e}')

    # Optional: tiny helper to get an interpolated shift (samples)
    def get_shift(self, muscle: str, velocity: float, method: str = 'linear', clamp: bool = True) -> int:
        vs, lags = self._delay_lookup[muscle]

        vq = float(np.clip(velocity, vs[0], vs[-1])) if clamp else float(velocity)
        return int(round(np.interp(vq, vs, lags)))
        
    #auxiliary functions 
    def w_aux(cond: str, comp: int, val: int): #gets the position of the each regression coefficient
        if cond== "elevation":
            return (comp-1)*3+val-1
        if cond=='speed':
            return (comp-1)*3+12+val-1
        else:
            return -1
            
    #calculate the weightings

            
    def calcWeightings(self, elevation, speed):
        elevation_temp=0.0
        speed_temp=0.0
        correction=0.0
        constant=0.0
        
        for comp in range(4): #4 components: 1 to 4
            for musc in range(15): #15 muscles: 1 to 15 
                #correction value at baseline: elevation=0 speed=3- Generic
                constant=MEP_class.reg_coef[musc-1][MEP_class.w_aux('elevation', comp, 1)] #a * e^2
                elevation_temp = constant*(0.0*0.0) #correction value for elevation: baseline value is 0
                
                constant=MEP_class.reg_coef[musc-1][MEP_class.w_aux('elevation', comp, 2)]#b*v
                elevation_temp+=(constant)*0.0
                
                constant=MEP_class.reg_coef[musc-1][MEP_class.w_aux('elevation', comp, 3)] #c
                elevation_temp+=constant
                
                constant=MEP_class.reg_coef[musc-1][MEP_class.w_aux('speed', comp, 1)]#A*v^2
                speed_temp = constant*(3.0*3.0) #correction value for speed- baseline value is 3 km/h
                
                constant=MEP_class.reg_coef[musc-1][MEP_class.w_aux('speed', comp, 2)]#B*v
                speed_temp+=(constant)*3.0
                
                constant=MEP_class.reg_coef[musc-1][MEP_class.w_aux('speed', comp, 3)] #C
                speed_temp+=constant
                
                correction=elevation_temp+speed_temp
                
                #weighting value -Subject specific
                constant=MEP_class.reg_coef[musc-1][MEP_class.w_aux('elevation', comp, 1)]
                elevation_temp = constant*(elevation*elevation) #correction value for elevation: baseline value is 0
                
                constant=MEP_class.reg_coef[musc-1][MEP_class.w_aux('elevation', comp, 2)]
                elevation_temp+=(constant)*elevation
                
                constant=MEP_class.reg_coef[musc-1][MEP_class.w_aux('elevation', comp, 3)]
                elevation_temp+=constant
                
                constant=MEP_class.reg_coef[musc-1][MEP_class.w_aux('speed', comp, 1)]
                speed_temp = constant*(speed*speed) #correction value for speed- baseline value is 3 km/h
                
                constant=MEP_class.reg_coef[musc-1][MEP_class.w_aux('speed', comp, 2)]
                speed_temp+=(constant)*speed
                
                constant=MEP_class.reg_coef[musc-1][MEP_class.w_aux('speed', comp, 3)]
                speed_temp+=constant
                
                #W=elevation+speed+baseline_weighting-correction
                MEP_class.weightings[musc-1][comp-1] = MEP_class.weightings_baseline[musc-1][comp-1]+elevation_temp+speed_temp-correction
                if(MEP_class.weightings[musc-1][comp-1]<0):
                    MEP_class.weightings[musc-1][comp-1]=0
        return MEP_class.weightings
                
                    
                
    def calcMEP(self, elevation: int, speed:float ):
        MEP_class.calcWeightings(self, elevation, speed)
        for i in range(200):
            for musc in range(15):
                MEP_class.mep[i][musc]=0
                for comp in range(4):
                    MEP_class.mep[i][musc] += MEP_class.gaussXP[comp][i]*MEP_class.weightings[musc][comp] #for the same timesample calculates the XP for all the muscles and then proceeds to the next time sample
                    #accumulates the MEP for one muscle for all components 
    
    def calc_MEP(self, elevation: int, speed: float):
        self.calcWeightings(elevation, speed)
        muscles=['TA', 'Sol', 'GasLat', 'GasMed']
        muscle_indices = [self.getMuscle_indx(m) for m in muscles]

        MEP=[[0 for _ in range(4)] for _ in range(200)] 
        for t in range(200):
            for m, m_idx in enumerate(muscle_indices): #nº of muscles
                MEP[t][m]=0
                for prim in range(4):
                    w_scaled=MEP_class.gains[prim]*MEP_class.weightings[m_idx][prim] 
                    MEP[t][m] += MEP_class.gaussXP[prim][t]*w_scaled #for the same timesample calculates the XP for all the muscles and then proceeds to the next time sample
        
        muscle_data = {m: [MEP[t][j] for t in range(200)] for j, m in enumerate(muscles)}
    
        MEP_normalized={muscle: [] for muscle in muscles}

        for musc in muscles: 
            normalized_time= np.linspace(0,100,100)    
            interpolation= interp1d(np.linspace(0,100, len(muscle_data[musc])), muscle_data[musc], kind='cubic')
            normalized_data=interpolation(normalized_time)
            
            try:
                lag = self.get_shift(musc, speed, method='linear', clamp=True)  # lag for 200-sample cycle
            except Exception:
                lag = 0
            
            data_shifted = np.roll(normalized_data, lag)

            MEP_normalized[musc]=data_shifted

        return MEP_normalized
                      
        
    
        
    @staticmethod

    def getMuscle_indx( muscle: str):
        if muscle=='TA':
            return 0
        elif muscle=='Sol':
            return 1
        elif muscle=='Per':
            return 2
        elif muscle=='VasLat':
            return 3
        elif muscle=='VastMed':
            return 4
        elif muscle=='RFem':
            return 5
        elif muscle=='Sar':
            return 6
        elif muscle=='Add':
            return 7
        elif muscle=='GlutMed':
            return 8
        elif muscle=='TFL':
            return 9
        elif muscle=='GasLat':
            return 10
        elif muscle=='GasMed':
            return 11
        elif muscle=='BFem':
            return 12
        elif muscle=='Semi':
            return 13
        elif muscle=='GlutMax':
            return 14

    def getMEP(self,  gaitCycle: int, muscle: str):
        i=MEP_class.getMuscle_indx(muscle)
        return MEP_class.mep[gaitCycle][i]
    
    def saveXP():
        #XP=open("XPsaved.txt", 'w')
        with open("XPsaved.txt", "w") as saveFile:
            for i in range(4):  # Loop over columns
                row_data = ",".join(str(MEP_class.gaussXP[j][i]) for j in range(200))
                saveFile.write(row_data + "\n")
                
    def saveWeightings():
        with open("Weightingssaved.txt", "w") as saveFile:
            for i in range(5):  # Loop over columns
                row_data = ",".join(str(MEP_class.gaussXP[j][i]) for j in range(15))
                saveFile.write(row_data + "\n")
    
    def saveMEP():
        with open("MEPsaved.txt", "w") as saveFile:
            for i in range(200):  # Loop over columns
                row_data = ",".join(str(MEP_class.gaussXP[j][i]) for j in range(16))
                saveFile.write(row_data + "\n")
    
    def saveData(data):
        print("saving data")
        with open("data_saved.txt", "w") as saveFile:
            saveFile.write(",".join(map(str, data)) + "\n")
        
                
        
                    
"""

void MEP_class::calcMEP(int elevation, int speed)
{
	MEP_class::calcWeightings(elevation, speed);
	for (int i=0;i<200; i++)
	{
		for (int musc=0;musc<15; musc++)
		{
			MEP_class::mep[i][musc] = 0; 
			for (int comp=0;comp<4;comp++)
			{
				MEP_class::mep[i][musc] += MEP_class::gaussXP[i][comp]*MEP_class::weightings[comp][musc];
			}
		}
	}		
}

int indexMuscle(string muscle)
{
	if (muscle == "TA") 
		return 1;
	else
		return 0;	
}

int MEP_class::getMuscle_Index(string muscle)
{
    if (muscle == "TA")
    {
        return 0;
    } else if (muscle == "Sol")
    {
        return 1;
    } else if (muscle == "Per")
    {
        return 2;
    } else if (muscle == "VastLat")
    {
        return 3;
    } else if (muscle == "VastMed")
    {
        return 4;
    } else if (muscle == "RFem")
    {
        return 5;
    } else if (muscle == "Sar")
    {
        return 6;
    } else if (muscle == "Add")
    {
        return 7;
    } else if (muscle == "GlutMed")
    {
        return 8;
    } else if (muscle == "TFL")
    {
        return 9;
    } else if (muscle == "GastLat")
    {
        return 10;
    } else if (muscle == "GastMed")
    {
        return 11;
    } else if (muscle == "BFem")
    {
        return 12;
    } else if (muscle == "Semi")
    {
        return 13;
    } else if (muscle == "GlutMax")
    {
        return 14;
    }

}

double MEP_class::getMEP(int gaitCycle, string muscle)
{
    return MEP_class::mep[gaitCycle][getMuscle_Index(muscle)];
}

void getMEP_all(int gaitCycle, double* data)
{
	//memcpy(data, MEP_class::mep[gaitCycle][1], 15);
}

void MEP_class::saveXP()
{
	ostringstream sstream;
	string saveText = "";
		
	cout << "saving XP" << endl;
	for (int i=0; i<4; i++)
	{	
		for (int j=0;j<200; j++)
		{
			sstream << MEP_class::gaussXP[j][i] << ",";
		}
		sstream << endl;
	}
	saveText = sstream.str();
	ofstream saveFile ("save/XP.txt");
	saveFile << saveText; 
	
	saveFile.close();
}

void MEP_class::saveWeightings()
{
	ostringstream sstream;
	string saveText = "";
	
	cout << "saving weigthings" << endl;
	for (int i=0; i<4; i++)
	{	
		for (int j=0;j<15; j++)
		{
			sstream << MEP_class::weightings[j][i] << ",";
		}
		sstream << endl;
	}
	saveText = sstream.str();
	ofstream saveFile ("save/weightings.txt");
	saveFile << saveText; 
	
	saveFile.close();
}

void MEP_class::saveMEP()
{
	ostringstream sstream;
	string saveText = "";
	
	cout << "saving MEP" << endl;
	for (int j=0; j<200; j++)
	{	
		for (int i=0;i<15; i++)
		{
			sstream << MEP_class::mep[j][i] << ",";
		}
		sstream << endl;
	}
	saveText = sstream.str();
	ofstream saveFile ("save/mep.txt");
	saveFile << saveText; 
	
	saveFile.close();
}

void MEP_class::saveData(double data[])
{
	ostringstream sstream;
	string saveText = "";
		
	cout << "saving data" << endl;
	for (int i=0; i<400; i++)
	{	
			sstream << data[i] << ",";
	}

	sstream << endl;
	saveText = sstream.str();
	ofstream saveFile ("save/data_out.txt");
	saveFile << saveText; 
	
	saveFile.close();
}
"""