import math as m
import numpy as np
import pandas as pd
from MEP_model import MEP_class
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy import signal, stats


#Path of text files
#Subjects 06, 08, 09, 12
#velocity=0.9 km/h
EMG_path_sub06_09=r"C:\Users\marisa\OneDrive - University of Twente\Master Thesis\Code\Sub06\EMG_data_sub06_09_old.txt"
STD_path_sub06_09=r"C:\Users\marisa\OneDrive - University of Twente\Master Thesis\Code\Sub06\STD_data_sub06_09_old.txt"

EMG_path_sub08_09=r"C:\Users\marisa\OneDrive - University of Twente\Master Thesis\Code\Sub08\EMG_data_sub08_09_old.txt"
STD_path_sub08_09=r"C:\Users\marisa\OneDrive - University of Twente\Master Thesis\Code\Sub08\STD_data_sub08_09_old.txt"

EMG_path_sub09_09=r"C:\Users\marisa\OneDrive - University of Twente\Master Thesis\Code\Sub09\EMG_data_sub09_09_old.txt"
STD_path_sub09_09=r"C:\Users\marisa\OneDrive - University of Twente\Master Thesis\Code\Sub09\STD_data_sub09_09_old.txt"

EMG_path_sub12_09=r"C:\Users\marisa\OneDrive - University of Twente\Master Thesis\Code\Sub12\EMG_data_sub12_09_old.txt"
STD_path_sub12_09=r"C:\Users\marisa\OneDrive - University of Twente\Master Thesis\Code\Sub12\STD_data_sub12_09_old.txt"


#velocity=1.8 km/h
EMG_path_sub06_18=r"C:\Users\marisa\OneDrive - University of Twente\Master Thesis\Code\Sub06\EMG_data_sub06_18_old.txt"
STD_path_sub06_18=r"C:\Users\marisa\OneDrive - University of Twente\Master Thesis\Code\Sub06\STD_data_sub06_18_old.txt"

EMG_path_sub08_18=r"C:\Users\marisa\OneDrive - University of Twente\Master Thesis\Code\Sub08\EMG_data_sub08_18_old.txt"
STD_path_sub08_18=r"C:\Users\marisa\OneDrive - University of Twente\Master Thesis\Code\Sub08\STD_data_sub08_18_old.txt"

EMG_path_sub09_18=r"C:\Users\marisa\OneDrive - University of Twente\Master Thesis\Code\Sub09\EMG_data_sub09_18_old.txt"
STD_path_sub09_18=r"C:\Users\marisa\OneDrive - University of Twente\Master Thesis\Code\Sub09\STD_data_sub09_18_old.txt"

EMG_path_sub12_18=r"C:\Users\marisa\OneDrive - University of Twente\Master Thesis\Code\Sub12\EMG_data_sub12_18_old.txt"
STD_path_sub12_18=r"C:\Users\marisa\OneDrive - University of Twente\Master Thesis\Code\Sub12\STD_data_sub12_18_old.txt"

#velocity=2.7
EMG_path_sub06_27=r"C:\Users\marisa\OneDrive - University of Twente\Master Thesis\Code\Sub06\EMG_data_sub06_27_old.txt"
STD_path_sub06_27=r"C:\Users\marisa\OneDrive - University of Twente\Master Thesis\Code\Sub06\STD_data_sub06_27_old.txt"

EMG_path_sub08_27=r"C:\Users\marisa\OneDrive - University of Twente\Master Thesis\Code\Sub08\EMG_data_sub08_27_old.txt"
STD_path_sub08_27=r"C:\Users\marisa\OneDrive - University of Twente\Master Thesis\Code\Sub08\STD_data_sub08_27_old.txt"

EMG_path_sub09_27=r"C:\Users\marisa\OneDrive - University of Twente\Master Thesis\Code\Sub09\EMG_data_sub09_27_old.txt"
STD_path_sub09_27=r"C:\Users\marisa\OneDrive - University of Twente\Master Thesis\Code\Sub09\STD_data_sub09_27_old.txt"

EMG_path_sub12_27=r"C:\Users\marisa\OneDrive - University of Twente\Master Thesis\Code\Sub12\EMG_data_sub12_27_old.txt"
STD_path_sub12_27=r"C:\Users\marisa\OneDrive - University of Twente\Master Thesis\Code\Sub12\STD_data_sub12_27_old.txt"

#velocity=3.6 km/h
EMG_path_sub06_36=r"C:\Users\marisa\OneDrive - University of Twente\Master Thesis\Code\Sub06\EMG_data_sub06_36_old.txt"
STD_path_sub06_36=r"C:\Users\marisa\OneDrive - University of Twente\Master Thesis\Code\Sub06\STD_data_sub06_36_old.txt"

EMG_path_sub08_36=r"C:\Users\marisa\OneDrive - University of Twente\Master Thesis\Code\Sub08\EMG_data_sub08_36_old.txt"
STD_path_sub08_36=r"C:\Users\marisa\OneDrive - University of Twente\Master Thesis\Code\Sub08\STD_data_sub08_36_old.txt"

EMG_path_sub09_36=r"C:\Users\marisa\OneDrive - University of Twente\Master Thesis\Code\Sub09\EMG_data_sub09_36_old.txt"
STD_path_sub09_36=r"C:\Users\marisa\OneDrive - University of Twente\Master Thesis\Code\Sub09\STD_data_sub09_36_old.txt"

EMG_path_sub12_36=r"C:\Users\marisa\OneDrive - University of Twente\Master Thesis\Code\Sub12\EMG_data_sub12_36_old.txt"
STD_path_sub12_36=r"C:\Users\marisa\OneDrive - University of Twente\Master Thesis\Code\Sub12\STD_data_sub12_36_old.txt"


#velocity=4.5
EMG_path_sub06_45=r"C:\Users\marisa\OneDrive - University of Twente\Master Thesis\Code\Sub06\EMG_data_sub06_45_old.txt"
STD_path_sub06_45=r"C:\Users\marisa\OneDrive - University of Twente\Master Thesis\Code\Sub06\STD_data_sub06_45_old.txt"

EMG_path_sub08_45=r"C:\Users\marisa\OneDrive - University of Twente\Master Thesis\Code\Sub08\EMG_data_sub08_45_old.txt"
STD_path_sub08_45=r"C:\Users\marisa\OneDrive - University of Twente\Master Thesis\Code\Sub08\STD_data_sub08_45_old.txt"

EMG_path_sub09_45=r"C:\Users\marisa\OneDrive - University of Twente\Master Thesis\Code\Sub09\EMG_data_sub09_45_old.txt"
STD_path_sub09_45=r"C:\Users\marisa\OneDrive - University of Twente\Master Thesis\Code\Sub09\STD_data_sub09_45_old.txt"

EMG_path_sub12_45=r"C:\Users\marisa\OneDrive - University of Twente\Master Thesis\Code\Sub12\EMG_data_sub12_45_old.txt"
STD_path_sub12_45=r"C:\Users\marisa\OneDrive - University of Twente\Master Thesis\Code\Sub12\STD_data_sub12_45_old.txt"

#velocity=5.4 km/h

EMG_path_sub06_54=r"C:\Users\marisa\OneDrive - University of Twente\Master Thesis\Code\Sub06\EMG_data_sub06_54_old.txt"
STD_path_sub06_54=r"C:\Users\marisa\OneDrive - University of Twente\Master Thesis\Code\Sub06\STD_data_sub06_54_old.txt"

EMG_path_sub08_54=r"C:\Users\marisa\OneDrive - University of Twente\Master Thesis\Code\Sub08\EMG_data_sub08_54_old.txt"
STD_path_sub08_54=r"C:\Users\marisa\OneDrive - University of Twente\Master Thesis\Code\Sub08\STD_data_sub08_54_old.txt"

EMG_path_sub09_54=r"C:\Users\marisa\OneDrive - University of Twente\Master Thesis\Code\Sub09\EMG_data_sub09_54_old.txt"
STD_path_sub09_54=r"C:\Users\marisa\OneDrive - University of Twente\Master Thesis\Code\Sub09\STD_data_sub09_54_old.txt"

EMG_path_sub12_54=r"C:\Users\marisa\OneDrive - University of Twente\Master Thesis\Code\Sub12\EMG_data_sub12_54_old.txt"
STD_path_sub12_54=r"C:\Users\marisa\OneDrive - University of Twente\Master Thesis\Code\Sub12\STD_data_sub12_54_old.txt"


#velocity=6.3 km/h
EMG_path_sub06_63=r"C:\Users\marisa\OneDrive - University of Twente\Master Thesis\Code\Sub06\EMG_data_sub06_63_old.txt"
STD_path_sub06_63=r"C:\Users\marisa\OneDrive - University of Twente\Master Thesis\Code\Sub06\STD_data_sub06_63_old.txt"

EMG_path_sub08_63=r"C:\Users\marisa\OneDrive - University of Twente\Master Thesis\Code\Sub08\EMG_data_sub08_63_old.txt"
STD_path_sub08_63=r"C:\Users\marisa\OneDrive - University of Twente\Master Thesis\Code\Sub08\STD_data_sub08_63_old.txt"

EMG_path_sub09_63=r"C:\Users\marisa\OneDrive - University of Twente\Master Thesis\Code\Sub09\EMG_data_sub09_63_old.txt"
STD_path_sub09_63=r"C:\Users\marisa\OneDrive - University of Twente\Master Thesis\Code\Sub09\STD_data_sub09_63_old.txt"

EMG_path_sub12_63=r"C:\Users\marisa\OneDrive - University of Twente\Master Thesis\Code\Sub12\EMG_data_sub12_63_old.txt"
STD_path_sub12_63=r"C:\Users\marisa\OneDrive - University of Twente\Master Thesis\Code\Sub12\STD_data_sub12_63_old.txt"


#velocity=8.1 km/h
EMG_path_sub06_81=r"C:\Users\marisa\OneDrive - University of Twente\Master Thesis\Code\Sub06\EMG_data_sub06_81_old.txt"
STD_path_sub06_81=r"C:\Users\marisa\OneDrive - University of Twente\Master Thesis\Code\Sub06\STD_data_sub06_81_old.txt"

EMG_path_sub08_81=r"C:\Users\marisa\OneDrive - University of Twente\Master Thesis\Code\Sub08\EMG_data_sub08_81_old.txt"
STD_path_sub08_81=r"C:\Users\marisa\OneDrive - University of Twente\Master Thesis\Code\Sub08\STD_data_sub08_81_old.txt"

EMG_path_sub09_81=r"C:\Users\marisa\OneDrive - University of Twente\Master Thesis\Code\Sub09\EMG_data_sub09_81_old.txt"
STD_path_sub09_81=r"C:\Users\marisa\OneDrive - University of Twente\Master Thesis\Code\Sub09\STD_data_sub09_81_old.txt"

EMG_path_sub12_81=r"C:\Users\marisa\OneDrive - University of Twente\Master Thesis\Code\Sub12\EMG_data_sub12_81_old.txt"
STD_path_sub12_81=r"C:\Users\marisa\OneDrive - University of Twente\Master Thesis\Code\Sub12\STD_data_sub12_81_old.txt"


#velocity=9.9 km/h

EMG_path_sub06_99=r"C:\Users\marisa\OneDrive - University of Twente\Master Thesis\Code\Sub06\EMG_data_sub06_99_old.txt"
STD_path_sub06_99=r"C:\Users\marisa\OneDrive - University of Twente\Master Thesis\Code\Sub06\STD_data_sub06_99_old.txt"

EMG_path_sub08_99=r"C:\Users\marisa\OneDrive - University of Twente\Master Thesis\Code\Sub08\EMG_data_sub08_99_old.txt"
STD_path_sub08_99=r"C:\Users\marisa\OneDrive - University of Twente\Master Thesis\Code\Sub08\STD_data_sub08_99_old.txt"

EMG_path_sub09_99=r"C:\Users\marisa\OneDrive - University of Twente\Master Thesis\Code\Sub09\EMG_data_sub09_99_old.txt"
STD_path_sub09_99=r"C:\Users\marisa\OneDrive - University of Twente\Master Thesis\Code\Sub09\STD_data_sub09_99_old.txt"

EMG_path_sub12_99=r"C:\Users\marisa\OneDrive - University of Twente\Master Thesis\Code\Sub12\EMG_data_sub12_99_old.txt"
STD_path_sub12_99=r"C:\Users\marisa\OneDrive - University of Twente\Master Thesis\Code\Sub12\STD_data_sub12_99_old.txt"


#Data for each muscle and velocity
muscles= ['Sol', 'TA', 'GasLat', 'GasMed']
velocities=[0.9, 1.8, 2.7, 3.6, 4.5, 5.4, 6.3, 8.1, 9.9]
subjects=['sub06', 'sub08', 'sub09', 'sub12']


velocity_map = { 
'09': 0.9, '18': 1.8, '27': 2.7, '36': 3.6, '45': 4.5,
'54': 5.4, '63': 6.3, '81': 8.1, '99': 9.9
}

EMG_data = {v: {} for v in velocities}
STD_data = {v: {} for v in velocities}

for v_str, v_float in velocity_map.items():
    for s in subjects:
        emg_var_name = f'EMG_sub{s[-2:]}_{v_str}'
        std_var_name = f'STD_sub{s[-2:]}_{v_str}'

        EMG_path = globals().get(f'EMG_path_sub{s[-2:]}_{v_str}')
        STD_path = globals().get(f'STD_path_sub{s[-2:]}_{v_str}')

        if EMG_path and STD_path:
            EMG_data[v_float][s] = pd.read_table(EMG_path, sep=',')
            STD_data[v_float][s] = pd.read_table(STD_path, sep=',')
emg_06=EMG_data[0.9]['sub06']
emg_09=EMG_data[0.9]['sub09']


#Define the MEP class
mep_instance= MEP_class()

mep_instance.importRegressionCoef()
mep_instance.importBaseline()
mep_instance.importXP()

MP=[]

MEP={muscle: {v: [] for v in velocities} for muscle in muscles} #each row has 200 samples 
vel_test= np.concatenate([
    np.linspace(6.0, 8.0, 100),
    np.linspace(5.0, 0.0, 100)
])

#MEP_test={muscle: [] for muscle in muscles}
for v in velocities:
#for v,indx in zip(vel_test, range(len(vel_test))):
    mep_instance.calcWeightings(elevation=0, speed=v)#
    mep_instance.calcMEP(elevation=0, speed=v)
    #mep_instance.calc_MEP_rt(elevation=0, speed=v, i=indx)
    for m in muscles:
        mep_row=[]
        for i in range(200):
            mep=mep_instance.getMEP(i, m)
            mep_row.append(mep)
        MEP[m][v]=mep_row
        #MEP_test[m]=mep_row

#Normalization of the data
MEP_normalized={muscle: {v: [] for v in velocities} for muscle in muscles}
for m in muscles:
    for v in velocities:
        normalized_time= np.linspace(0,100,100)
        interpolation= interp1d(np.linspace(0,100, len(MEP[m][v])), MEP[m][v], kind='cubic')
        normalized_data=interpolation(normalized_time)
        MEP_normalized[m][v]=normalized_data
    #interpolation_test= interp1d(np.linspace(0,100, len(MEP_test[m])), MEP_test[m], kind='cubic')
    #normalized_data_test=interpolation_test(normalized_time)
    #MEP_normalized[m]=normalized_data_test



#MA_09=EMG_data[0.9]['sub06']['TA']
#STD_09=STD_data[0.9]['sub06']['TA']

'''
## Evaluation of the performance of the CPG: r^2, pearson correlation, cross-correlation (time shift) 
metrics=['PearsonCorr', 'R2', 'CrossCorr']
Performance={met: {vel: {muscle: {sub: [] for sub in subjects} for muscle in muscles} for vel in velocities} for met in metrics }

for v in velocities: 
    for musc in muscles:
        estimate=MEP_normalized[musc][v]

        for s in subjects:
            measurement=EMG_data[v][s][musc]
            syn=0.5*measurement+0.5*estimate
            Performance['CrossCorr'][v][musc][s]=np.corrcoef(syn, estimate)[0, 1]
            lag=np.argmax(signal.correlate(syn, estimate))-(len(syn)-1)
            aligned_estimate=np.roll(estimate, lag)
            Performance['PearsonCorr'][v][musc][s]=stats.pearsonr(syn, aligned_estimate)
            Performance['R2'][v][musc][s]=sklearn.metrics.r2_score(syn, aligned_estimate)

#Average across subjects 
for muscle in muscles: 
    rows=[]
    for v in velocities:
        row={'velocity': v}
        for metric in metrics:
            values=list(Performance[metric][v][muscle].values())
            mean=np.mean(values)
            std=np.std(values)
            row[metric]= f'{mean:.2f} \u00B1 {std:.2f}'
        rows.append(row)
        
    df_muscle=pd.DataFrame(rows)
    print(f"\n=== {muscle.upper()} ===")
    print(df_muscle.to_string(index=False))
    filename = f'{muscle}_synergy_performance.csv'
    df_muscle.to_csv(filename, index=False)

'''

fig, axs = plt.subplots(2, 2, figsize=(12, 8))

# Setup
subs = ['sub06', 'sub08', 'sub09', 'sub12']
positions = [(0, 0), (0, 1), (1, 0), (1, 1)]
MP_estimate_09=MEP_normalized['TA'][2.7]

for i, sub in enumerate(subs):
    r, c = positions[i]
    ax1 = axs[r, c]
    #ax2 = ax1.twinx()

    # Select EMG and STD data
    emg = EMG_data[2.7][sub]['GasLat']
    std = STD_data[2.7][sub]['GasLat']
    #blended_signal=MP_estimate_09*0.5+ emg*0.5
    
    
    
    plt.xlabel('Gait Cycle (%)')
    plt.ylabel('Muscle Primitive')
    #plt.title("Muscle Activation: Soleus")
    #plt.legend()
    #plt.show()
    
    # Left Y-axis (EMG)
    
    ax1.plot(normalized_time, emg, color='m', label='Mean EMG+ STD')
    ax1.fill_between(normalized_time, emg - std, emg + std, color='m', alpha=0.2)
    ax1.set_ylabel('EMG')
    ax1.tick_params(axis='y')
    ax1.grid(True)

    # Right Y-axis (MP)
    ax1.plot(normalized_time, MP_estimate_09, color='b', label='MP')

   

    # Title & labels
    ax1.set_title(sub)
    ax1.set_xlabel('Gait Cycle (%)')
    ax1.legend()
    #lines1, labels1 = ax1.get_legend_handles_labels()
    #lines2, labels2 = ax2.get_legend_handles_labels()
    #ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    
# Adjust layout and prevent overlap
fig.tight_layout()
fig.suptitle('Synergy: Gastrocnemius Lateralis v=2.7 km/h')   
#plt.title('CPG+EMG: Tibialis Anterior v=9.9 km/h')
#plt.grid()

plt.show()






"""_summary_
    
#Calculate the RMSE between MA and MEP
error={muscle: [] for muscle in muscles}
for m in muscles:
    for e in range(len(MA[m])):
        er=sklearn.metrics.mean_absolute_error(MA[m][e], MEP_normalized[m][e])
        error[m].append(er)
        #print(error)

plt.plot(velocities, error['Sol'],label='Soleus', color='blue')
plt.plot(velocities, error['TA'], label='Tibialis Anterior', color='m')
plt.plot(velocities, error['GasLat'], label='Gastrocnemius Lateralis', color='green' )
plt.plot(velocities, error['GasMed'], label='Gastrocenmius Medialis', color='r')
"""
# X-axis and Title





#muscle_index=mep_instance.getMuscle_indx(muscle= 'TA')
#muscles=['TA', 'Sol', 'GastLat', 'GastMed', 'Per']
#for i in range(200):
#    mep=mep_instance.getMEP(i, 'Sol')
#    MEP.append(mep)
        
#w=mep_instance.w_aux(cond= 1 , comp=1, val= 1)
#normalized_time= np.linspace(0,100,100)
#interpolation= interp1d(np.linspace(0,100, len(MEP)), MEP, kind='cubic')
#normalized_data=interpolation(normalized_time)


# First Y-axis (left) for Muscle Primitive
#plt.plot(normalized_time, normalized_data, label="Muscle Primitive", color='blue')
#plt.plot(normalized_time, EMG_SOL, label='Mean EMG Envelope', color='magenta')
#plt.fill_between(normalized_time, EMG_SOL - STD_SOL, EMG_SOL + STD_SOL, 
#                 color='m', alpha=0.2, label='Standard Deviation')

# X-axis and Title
#plt.xlabel('Gait Cycle (%)')
#plt.ylabel('Muscle Primitive')
#plt.title("Muscle Activation: Soleus")
#plt.legend()
#plt.grid()
#plt.show()
#save the output 
#mep_instance.saveData()
#mep_instance.saveMEP()
#mep_instance.saveWeightings()
#mep_instance.saveXP()

#### Save Muscle Activations to sto file -EMGfilt.sto  
fs=100
nRows=100
nCols= 5
time= np.linspace(0, (nRows-1)/fs, nRows)
velocity=5.4
data_per_muscle = [MEP_normalized[m][velocity] for m in muscles]
header = (f"version=1\n"
    f"nRows={nRows}\n"
    f"nColumns={nCols}\n"
    f"data_start_time=0.0\n"
    f"end_time={(nRows-1)/fs:.2f}\n"
    f"end header\n"
    f"time {' '.join(muscles)}\n" )
         
with open('EMGfilt.sto', 'w') as f:
    f.write(header)
    for ri, t in enumerate(time):
        # get each muscle’s value at this row
        vals = [data_per_muscle[i][ri] for i in range(len(muscles))]
        # format time and muscle values (6 decimal places)
        row = [f"{t:.6f}"] + [f"{v:.6f}" for v in vals]
        f.write(' '.join(row) + '\n')
        

      
