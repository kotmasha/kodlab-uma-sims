from __future__ import division
import sys
import json
import numpy as np
from numpy.random import randint as rnd
import matplotlib as mpl
import os
mpl.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.animation as animation


#--------------------------------------------------------------------------------------
# This plotter assumes the presence of the following data file structure:
#
# - All output files are located inside a subdir of the working dir named <name>
#
# - The subdir <name> contains a preamble file named "name.pre"
#
# - Each run is recorded in a data file named "name_i.dat", where $i$ is the run index
#
# - The preamble format is a Python DICT with the following standard keys:
#   'Nruns'             -   the number of runs recorded in this directory
#   'name'              -   the string identifier <name>
#   'ex_dataQ'          -   Boolean indicating whether or not experiment update cycle
#                           data was recorded in addition to the experiment state
#   'agent_dataQ'       -   Boolean indicating whether or not per-agent update cycle
#                           data was recorded in addition to the experiment state
#   'mids_to_record'    -   List of mids (belonging to the experiment) whose values
#                           were recorded (for each cycle of each run)
#
# - Additional preamble values are experiment-dependent.
#   For SNIFFY, we have:
#   'env_length'        -   The size of SNIFFY's environment
#   'total_cycles'      -   Length (in cycles) of the whole run   
#--------------------------------------------------------------------------------------

def get_pickles(infile):
    for item in infile:
        yield json.loads(item)


#
# Read the command line input and preamble (GENERIC)
#

NAME=sys.argv[1] #first argument to this script is an experiment name

preamble_file_name = os.path.join(NAME, NAME+".pre")
preamblef=open(preamble_file_name,'rb')
preamble=json.load(preamblef)
preamblef.close()


#
# Open the data files from a signle run (GENERIC)
#

RUN_NAME=lambda i: NAME+"_"+str(i)
input_file_name=lambda i: os.path.join(NAME, RUN_NAME(i) + ".dat")
supp_file_name=lambda i: os.path.join(NAME, RUN_NAME(i) + ".sup")

try:
    NUM=int(sys.argv[2]) #second argument to this script (if any) is a run number
except:
    NUM=rnd(preamble['Nruns'])

input_file=open(input_file_name(NUM),'rb')
supp_file=open(supp_file_name(NUM),'rb')


#
# Prepare data entries from a single run (GENERIC)
#

DATA={}

#- prepare entries for experiment measurables
if preamble['mids_recorded'] is []:
    pass
else:
    for mid in preamble['mids_recorded']:
        DATA[mid]=[]

#- prepare entries for update cycle reports
if bool(preamble['ex_dataQ']):
    for mid in preamble['ex_data_recorded']:
        DATA[mid]=[]

#- prepare entries for per-agent update cycle reports
if bool(preamble['agent_dataQ']):
    for mid in preamble['agent_data_recorded']:
        for agent_id in preamble['agents']:
            DATA[(agent_id,mid)]=[]


#
# Read the data from the .dat and .sup files (GENERIC)
#

#load data from the supplementary file:
SUPP=json.loads(supp_file.readline())  

#load data from the data file:
for record in get_pickles(input_file):
    #- read entries for experiment measurables        
    if preamble['mids_recorded'] is []:
        pass
    else:
        for mid,item in zip(preamble['mids_recorded'],record['mids_recorded']):
            DATA[mid].append(item)
    
    #- read entries for experiment update cycle data
    if bool(preamble['ex_dataQ']):    
        for tag,item in zip(preamble['ex_data_recorded'],record['ex_data_recorded']):
            DATA[tag].append(item)

    #- read entries for experiment update cycle data        
    if bool(preamble['agent_dataQ']):
        for agent_id in preamble['agents']:
            for tag,item in zip(preamble['agent_data_recorded'],record['agent_data_recorded'][agent_id]):
                DATA[(agent_id,tag)].append(item)

# close the data & supplementary files:
input_file.close()
supp_file.close()


#------------------------------------------------------------------------------------
# At this point, each DATA[tag] item is a 1-dim Python list object,
# with the tags taking the form of:
# - an experiment measurable id;
# - a measurement tag from the update cycle (time stamp, decision, etc.);
# - a double tag of the form (agent_id,tag) indicating an agent-specific measurement
#   from that agent's update cycle.
#
# From this point on, all instructions are specific to the experiment at hand
#------------------------------------------------------------------------------------

#
# Prepare the plots (EXPERIMENT-SPECIFIC)
#

AGENT_TYPES={
    '_Q':['qualitative','xkcd:red'],
    '_Eu':['empirical uniform','xkcd:sky blue'],
    '_Ev':['empirical value-based','xkcd:blue'],
    '_Du':['discounted uniform','xkcd:forest green'],
    '_Dv':['discounted value-based','xkcd:green'],
    }
#ORDERED_TYPES=['_Q','_Eu','_Ev','_Du','_Dv']
#ORDERED_TYPES=['_Eu','_Du']
<<<<<<< HEAD
ORDERED_TYPES=['_Q','_Ev','_Dv']
=======
ORDERED_TYPES=['_Q','_Eu']
>>>>>>> 25850924b7ce01f3c9bfb0969867a997c104c27d
NTYPES=len(ORDERED_TYPES)

# length of the environment (due to differences between circle and interval):
ENV_LENGTH=SUPP['env_length']
# number of sensors:
Nsensors=SUPP['Nsensors']

# duration of the experiment
DURATION=len(DATA['counter'])

#
#Initialize the plots
#

t=xrange(DURATION)
WEIGHT_DIFFS={typ:np.array(DATA['wdiff'+typ]) for typ in ORDERED_TYPES}
RAW_DIFFS={typ:np.array(DATA['rdiff'+typ]) for typ in ORDERED_TYPES}
FULL_DIFFS={typ:np.array(DATA['fdiff'+typ]) for typ in ORDERED_TYPES}

#Create target and agent representation:
TARGET_TRAJECTORY={typ:np.array(DATA['targ'+typ],dtype=int).T for typ in ORDERED_TYPES}
TARGET_GROUND={typ:np.array([SUPP['ground_targ'][typ] for t in xrange(DURATION)],dtype=int).T for typ in ORDERED_TYPES}
POS=np.array(DATA['pos'])

#- initialize figure
fig,axes=plt.subplots(nrows=NTYPES,ncols=1,sharex=True,sharey=False)
fig.suptitle('# of incorrect implications over time',fontsize=10)
plt.subplots_adjust(left=0.05,right=0.95,bottom=0.05,top=0.95)
plt.xlabel('time elapsed (cycles)',fontsize=10)

#- initialize implications plot
#ax_imps=axes[0]
#ax_imps.set_ylabel('# incorrect implications',fontsize=10)
#ax_imps.set_ylim(bottom=0,top=Nsensors*Nsensors)

#- associate axes to types
AX={}
#for typ,ax in zip(ORDERED_TYPES,axes[1:]):
for typ,ax in zip(ORDERED_TYPES,axes):
    AX[typ]=ax
    AX[typ].set_title('Target state representation over time, '+AGENT_TYPES[typ][0],fontsize=10)
    AX[typ].set_ylabel('position in env',fontsize=10)
    AX[typ].yaxis.set_ticks(-0.5+np.arange(ENV_LENGTH))
    AX[typ].tick_params(labelbottom=True,labelleft=False)

#- form the implications plot
t=np.array(DATA['counter'])
#for typ in ORDERED_TYPES:
#    diff_raw=np.array(RAW_DIFFS[typ])
#    diff_full=np.array(FULL_DIFFS[typ])
#    ALPH=0.7
#    ax_imps.plot(t,diff_raw,linestyle='solid',linewidth=3,color=AGENT_TYPES[typ][1],alpha=ALPH,label='Learned PCR vs. Ground PCR, '+AGENT_TYPES[typ][0])
#    ax_imps.plot(t,diff_full,linestyle='dashed',linewidth=3,color=AGENT_TYPES[typ][1],alpha=ALPH,label='Full implications vs. Ground PCR, '+AGENT_TYPES[typ][0])
#ax_imps.legend()

#- form the trajectories plots
for typ in ORDERED_TYPES:
    EXTENT=(1,DURATION+1,0,ENV_LENGTH-1)
    AX[typ].imshow(TARGET_TRAJECTORY[typ], cmap = plt.cm.Blues, vmin = 0, vmax = 1, aspect='auto',interpolation='none',alpha=0.5,extent=EXTENT)
    AX[typ].imshow(TARGET_GROUND[typ], cmap = plt.cm.Reds, vmin = 0, vmax = 1, aspect='auto',interpolation='none',alpha=0.5,extent=EXTENT)
    AX[typ].plot(t,np.array(POS),'.r',alpha=1,label='Observer\'s position')
    AX[typ].legend()

#Show the plots
plt.show()