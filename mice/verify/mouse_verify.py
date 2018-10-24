from __future__ import division
import sys
import json
import numpy as np
import matplotlib as mpl
import os
mpl.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib import cm

#--------------------------------------------------------------------------------------
# This plotter assumes the presence of the following output file structure:
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
#   'burn_in_cycles'    -   Length (in cycles) of initial period of randomized behavior
#   'total_cycles'      -   Length (in cycles) of the whole run   
#--------------------------------------------------------------------------------------

def get_pickles(infile):
    for item in infile:
        yield json.loads(item)
#    try:
#        while True:
#            yield cPickle.load(infile)
#    except EOFError:
#        pass


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
NRUNS=preamble['Nruns']


#
# Open the data files (GENERIC)
#

input_file={}
for ind in xrange(NRUNS):
    input_file[ind]=open(input_file_name(ind),'rb')


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
if (not preamble['ex_dataQ']) or (preamble['ex_data_recorded'] is []):
    pass
else:
    for mid in preamble['ex_data_recorded']:
        DATA[mid]=[[] for ind in xrange(NRUNS)]

#- prepare entries for per-agent update cycle reports
if (not preamble['agent_dataQ']) or (preamble['agent_data_recorded'] is []):
    pass
else:
    for mid in preamble['agent_data_recorded']:
        for agent_id in preamble['agents']:
            DATA[(agent_id,mid)]=[[] for ind in xrange(NRUNS)]


#
# Read the data from the files (GENERIC)
#

for ind in xrange(NRUNS):
    for record in get_pickles(input_file[ind]):
        #- read entries for experiment measurables        
        if preamble['mids_recorded'] is []:
            pass
        else:
            for mid,item in zip(preamble['mids_recorded'],record['mids_recorded']):
                DATA[mid][ind].append(item)
        #- read entries for experiment update cycle data        
        if (not preamble['ex_dataQ']) or (preamble['ex_data_recorded'] is []):
            pass
        else:
            for tag,item in zip(preamble['ex_data_recorded'],record['ex_data_recorded']):
                DATA[tag][ind].append(item)
        #- read entries for experiment update cycle data        
        if (not preamble['agent_dataQ']) or (preamble['agent_data_recorded'] is []):
            pass
        else:
            for agent_id in preamble['agents']:
                for tag,item in zip(preamble['agent_data_recorded'],record['agent_data_recorded'][agent_id]):
                    DATA[(agent_id,tag)][ind].append(item)

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

#Prepare the axes
fig,ax=plt.subplots()
plt.subplots_adjust(left=0.05,right=0.95,bottom=0.1,top=0.9)
fig.suptitle('Mice: mean and std. deviations of distance to nearest cheese as a function of time\n after '+str(preamble['burn_in_cycles'])+'cycles of training by random exploration in a '+str(2*preamble['viewportSize'])+'*'+str(2*preamble['viewportSize'])+' environment.',fontsize=22)
plt.xlabel('time elapsed (run cycles)',fontsize=16)
plt.ylabel('min distance to target',fontsize=16)

#Form the plots
t=np.array(DATA['counter'][0])
#print len(DATA['che_out'][0][0])
#print len(DATA['che_out'][1][0])
#print len(DATA['che_out'][0])
#exit(0)

#cheeseNums=np.array([[len(item) for item in DATA['che_out'][ind]] for ind in xrange(NRUNS)])
#dmean=np.mean(cheeseNums,axis=0)
#dstd=np.std(cheeseNums,axis=0)

minDist=np.array([DATA['mdist'][ind] for ind in xrange(NRUNS)])
dmean=np.mean(minDist,axis=0)
dstd=np.std(minDist,axis=0)


plt.plot(t,dmean,'-r',alpha=1,label='Mean over '+str(NRUNS)+' runs')
plt.fill_between(t,dmean-dstd,dmean+dstd,alpha=0.2,color='r',label='std. deviation over '+str(NRUNS)+' runs')
ymin,ymax=plt.ylim()
plt.plot([preamble['burn_in_cycles'],preamble['burn_in_cycles']],[ymin,ymax],'-bo',label='training period ends')
plt.plot(preamble['burn_in_cycles'],ymin,preamble['burn_in_cycles'],ymax,'bo',)
ax.legend()

#Show the plots
plt.show()