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

np.set_printoptions(threshold=np.nan)
#
### CONSTANTS
#

BURN_IN=preamble['burn_in_cycles']
ENV_LENGTH=SUPP['env_length']
DIAM=SUPP['diam']
NSENSORS=SUPP['Nsensors']

ACCESS_KEYS=['_rt+','_rt-','_lt+','_lt-']
nc=lambda x: x[:-1]+'+' if x[-1]=='-' else x[:-1]+'-'

#distQ=True
#targetQ=True
#predQ=True
#divsQ=True

#
### Create data representation
#

#time counter
t=np.array(DATA['counter'])
DURATION=len(t)

#distance to target
dist=np.array(DATA['dist'])

#position
POS=np.array(DATA['pos'])

#target
TARGET_TRAJECTORY={key:np.array(DATA['targ'+key],dtype=int).T for key in ACCESS_KEYS}
TARGET_GROUND=np.array([[pos==DATA['tpos'][ind] for pos in xrange(ENV_LENGTH)] for ind in xrange(DURATION)]).T

#prediction
PREDICTION_TRAJECTORY={key:np.array(DATA['pred'+key],dtype=int).T for key in ACCESS_KEYS}

#divergences
DIVERGENCES={key:np.array(DATA['div'+key],dtype=int) for key in ACCESS_KEYS}

#print np.matrix(np.array(PREDICTION_TRAJECTORY['_rt+']).T[:-20])
#exit(0)

#
### Prepare the axes
#

fig,axes=plt.subplots(nrows=5,ncols=1,sharex=True,sharey=False)
plt.subplots_adjust(left=0.05,right=0.95,bottom=0.1,top=0.9)

ax_dist=axes[0]
ax_dist.set_title('distance to target')
#ax_dist.set_ylabel('# steps to target',fontsize=10)
#ax_dist.set_ylim(bottom=-1,top=ENV_LENGTH/2)
#main plot
ax_dist.plot(t,dist,'-k',alpha=1,label='distance to target')
#end of training
ax_dist.plot([BURN_IN,BURN_IN],[-1,DIAM+1],'-b.',label='training period ends')
ax_dist.legend()

AX={}
for key,ax in zip(ACCESS_KEYS,axes[1:]):
    AX[key]=ax
    #ax.invert_yaxis()
    ax.set_title('internal state in snapshot'+key,fontsize=10)
    ax.set_ylabel('agent\'s position',fontsize=10)
    ax.yaxis.set_ticks(-1+np.arange(ENV_LENGTH+1))
    if key==ACCESS_KEYS[-1]:
        ax.tick_params(labelbottom=True,labelleft=True,labelsize=6)
    else:
	ax.tick_params(labelbottom=False,labelleft=True,labelsize=6)
    #main plot
    EXTENT=(0.5,DURATION+0.5,-0.5,ENV_LENGTH-0.5)
    ax.imshow(PREDICTION_TRAJECTORY[key], origin='lower', cmap = plt.cm.Greens, vmin = 0, vmax = 1, aspect='auto',interpolation='none',alpha=1,extent=EXTENT,label='predicted position')
    ax.imshow(TARGET_TRAJECTORY[key], origin='lower', cmap = plt.cm.Blues, vmin = 0, vmax = 1, aspect='auto',interpolation='none',alpha=0.7,extent=EXTENT,label='computed target')
    ax.imshow(TARGET_GROUND, origin='lower', cmap = plt.cm.Reds, vmin = 0, vmax = 1, aspect='auto',interpolation='none',alpha=0.5,extent=EXTENT,label='true target')
    ax.plot(t,np.array(POS),'.r',alpha=.5,label='agent\'s position')
    #end of training
    ax.plot([BURN_IN,BURN_IN],[-1,ENV_LENGTH],'-b.',label='training period ends')
    ax.plot(t,-2+np.sign(-DIVERGENCES[key]+DIVERGENCES[nc(key)]),'-g',alpha=1,label='divergence difference')
    if key==ACCESS_KEYS[-1]:
        ax.legend(loc='upper right')


#Show the plots
plt.show()
