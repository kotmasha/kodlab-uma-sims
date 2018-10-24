from __future__ import division
import sys
import json
import numpy as np
from functools import partial
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
# Read the preamble (GENERIC)
#

NAME=sys.argv[1]

preamble_file_name = os.path.join(NAME, NAME+".pre")
preamblef=open(preamble_file_name,'rb')
preamble=json.load(preamblef)
preamblef.close()

RUN_NAME=lambda i: NAME+"_"+str(i)
input_file_name=lambda i: os.path.join(NAME, RUN_NAME(i) + ".dat")
supp_file_name=lambda i: os.path.join(NAME, RUN_NAME(i) + ".sup")
NRUNS=preamble['Nruns']

#
# Open the data files (GENERIC)
#

input_file={}
for ind in xrange(NRUNS):
    input_file[ind]=open(input_file_name(ind),'rb')

supp_file={}
for ind in xrange(NRUNS):
    supp_file[ind]=open(supp_file_name(ind),'rb')


#
# Prepare data entries (GENERIC)
#

DATA={}

#- prepare entries for experiment measurables
if preamble['mids_recorded'] is []:
    pass
else:
    for mid in preamble['mids_recorded']:
        DATA[mid]=[[] for ind in xrange(NRUNS)]

#- prepare entries for update cycle reports
if bool(preamble['ex_dataQ']):
    for mid in preamble['ex_data_recorded']:
        DATA[mid]=[[] for ind in xrange(NRUNS)]

#- prepare entries for per-agent update cycle reports
if bool(preamble['agent_dataQ']):
    for mid in preamble['agent_data_recorded']:
        for agent_id in preamble['agents']:
            DATA[(agent_id,mid)]=[[] for ind in xrange(NRUNS)]


#
# Read the data from the .dat files (GENERIC)
#
SUPP={}
for ind in xrange(NRUNS):
    #load data from the supplementary files:
    SUPP[ind]=json.loads(supp_file[ind].readline())  

    #load data from the data files:
    for record in get_pickles(input_file[ind]):
        #- read entries for experiment measurables        
        if preamble['mids_recorded'] is []:
            pass
        else:
            for mid,item in zip(preamble['mids_recorded'],record['mids_recorded']):
                DATA[mid][ind].append(item)
        #- read entries for experiment update cycle data
        if bool(preamble['ex_dataQ']):    
            for tag,item in zip(preamble['ex_data_recorded'],record['ex_data_recorded']):
                DATA[tag][ind].append(item)
        #- read entries for experiment update cycle data        
        if bool(preamble['agent_dataQ']):
            for agent_id in preamble['agents']:
                for tag,item in zip(preamble['agent_data_recorded'],record['agent_data_recorded'][agent_id]):
                    DATA[(agent_id,tag)][ind].append(item)

# close the data & supplementary files:
for ind in xrange(NRUNS):
    input_file[ind].close()
    supp_file[ind].close()


#------------------------------------------------------------------------------------
# At this point, each DATA[tag] item is a 2-dim Python list object,
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
<<<<<<< HEAD
#ORDERED_TYPES=['_Q','_Eu','_Ev','_Du','_Dv']
ORDERED_TYPES=['_Eu','_Ev']
=======
ORDERED_TYPES=['_Q','_Eu','_Ev','_Du','_Dv']
#ORDERED_TYPES=['_Q','_Eu']
>>>>>>> 25850924b7ce01f3c9bfb0969867a997c104c27d
NTYPES=len(ORDERED_TYPES)

# length of the environment (due to differences between circle and interval):
ENV_LENGTH=SUPP[0]['env_length']
# number of sensors:
NSENSORS=SUPP[0]['Nsensors']

# duration of the experiment for each run
duration=lambda ind: len(DATA['counter'][ind])
def duration_gen():
    for ind in xrange(NRUNS):
        yield duration(ind)
DURATION=max(duration_gen())


#
#Initialize the plots
#

t=xrange(DURATION)
WEIGHT_DIFFS={typ:np.array(DATA['wdiff'+typ]) for typ in ORDERED_TYPES}
RAW_DIFFS={typ:np.array(DATA['rdiff'+typ]) for typ in ORDERED_TYPES}
FULL_DIFFS={typ:np.array(DATA['fdiff'+typ]) for typ in ORDERED_TYPES}

#- initialize figure
fig,ax_imps=plt.subplots(nrows=len(ORDERED_TYPES),ncols=1,sharex=True,sharey=True)
fig.suptitle('# of incorrect implications (out of '+str(4*pow(NSENSORS,2))+') over time',fontsize=10)
plt.subplots_adjust(left=0.05,right=0.95,bottom=0.05,top=0.95)
plt.xlabel('time elapsed (cycles)',fontsize=10)

#- form the implications plots
AX={}
for typ,ax in zip(ORDERED_TYPES,ax_imps):
    AX[typ]=ax
    ax.set_ylabel('# incorrect implications',fontsize=10)
    
    #weight_diffs=WEIGHT_DIFFS[typ]
    raw_diffs=RAW_DIFFS[typ]
    full_diffs=FULL_DIFFS[typ]
    YMAX=max(np.max(raw_diffs),np.max(full_diffs))
    ax.set_ylim(bottom=0.,top=YMAX)

    #means over runs
    #diff_weight_mean=np.mean(weight_diffs,axis=0)
    diff_raw_mean=np.mean(raw_diffs,axis=0)
    diff_full_mean=np.mean(full_diffs,axis=0)

    #standard deviations over runs
    #diff_weight_sdv=np.std(weight_diffs,axis=0)
    diff_raw_sdv=np.std(raw_diffs,axis=0)
    diff_full_sdv=np.std(full_diffs,axis=0)

    ALPH=0.7 #foreground transparency coefficients
    BETA=0.2 #background transparency coefficients

    #ax.fill_between(t,diff_weight_mean-diff_weight_sdv,diff_weight_mean+diff_weight_sdv,alpha=BETA,color=AGENT_TYPES[typ][1])
    ax.fill_between(t,diff_raw_mean-diff_raw_sdv,diff_raw_mean+diff_raw_sdv,alpha=BETA,color=AGENT_TYPES[typ][1])
    ax.fill_between(t,diff_full_mean-diff_full_sdv,diff_full_mean+diff_full_sdv,alpha=BETA,color=AGENT_TYPES[typ][1])
    
    #ax.plot(t,diff_weight_mean,linestyle=':',linewidth=3,color=AGENT_TYPES[typ][1],alpha=ALPH,label='Learned weights vs. Expected weights, '+AGENT_TYPES[typ][0])
    ax.plot(t,diff_raw_mean,linestyle='solid',linewidth=3,color=AGENT_TYPES[typ][1],alpha=ALPH,label='Learned PCR vs. Ground PCR, '+AGENT_TYPES[typ][0])
    ax.plot(t,diff_full_mean,linestyle='dashed',linewidth=3,color=AGENT_TYPES[typ][1],alpha=ALPH,label='Full implications vs. actual implications, '+AGENT_TYPES[typ][0])

    ax.legend()
#ax_imps.legend()

#Show the plots
plt.show()
