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

BURN_IN=preamble['burn_in_cycles']
ENV_LENGTH=preamble['circumference']
RADIUS=ENV_LENGTH/4+3

#Prepare the axes
fig,ax=plt.subplots()
plt.subplots_adjust(left=0.05,right=0.95,bottom=0.1,top=0.9)
#fig.suptitle('Sniffy: mean and std. deviations of distance to the target as a function of time\n during and after training',fontsize=22)
plt.xlabel('time elapsed (cycles)',fontsize=16)
plt.ylabel('distance to target',fontsize=16)

#Form the plots
t=np.array(DATA['counter'][0][BURN_IN+1:])

# total mean
#dmean=np.mean(np.array(DATA['dist']),axis=0)
#ddev=np.std(np.array(DATA['dist']),axis=0)
#
#plt.fill_between(t,dmean-ddev,dmean+ddev,color='k',alpha=0.2,label='std. deviation over '+str(NRUNS)+' runs')
#plt.plot(t,dmean,'-k',alpha=1,label='mean over '+str(NRUNS)+' runs')

# mean over training walks ending closer to target
DATA_BOT=filter(lambda item: item[BURN_IN+2]<=RADIUS,DATA['dist'])
bmean=np.mean(DATA_BOT,axis=0)[BURN_IN+1:]
bdev=np.std(DATA_BOT,axis=0)[BURN_IN+1:]

plt.fill_between(t,bmean-bdev,bmean+bdev,alpha=0.2,color='r',label='std. deviation over '+str(len(DATA_BOT))+' runs')
for item in DATA_BOT:
    plt.plot(t,item[BURN_IN+1:],'-r',alpha=1,color='r',label='mean over population A ('+str(len(DATA_BOT))+') runs')

# mean over training walks ending closer to antipode
DATA_TOP=filter(lambda item: item[BURN_IN+2]>RADIUS,DATA['dist'])
tmean=np.mean(DATA_TOP,axis=0)[BURN_IN+1:]
tdev=np.std(DATA_TOP,axis=0)[BURN_IN+1:]

plt.fill_between(t,tmean-tdev,tmean+tdev,alpha=0.2,color='b',label='std. deviation over '+str(len(DATA_TOP))+' runs')
for item in DATA_TOP:
    plt.plot(t,item[BURN_IN+1:],'-',alpha=1,color='b',label='mean over population B ('+str(len(DATA_TOP))+') runs')

#training period separator
ymin,ymax=plt.ylim()
plt.plot([preamble['burn_in_cycles']+2,preamble['burn_in_cycles']+2],[ymin,ymax],'--b.',label='training period ends (t='+str(preamble['burn_in_cycles'])+')')
#ax.legend()

#Show the plots
plt.show()
