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



def get_data(path,name,data,supp):
    data.clear()
    supp.clear()
    #
    # Read the preamble (GENERIC)
    #
    preamble_file_name = os.path.join(path, name, name+".pre")
    preamblef=open(preamble_file_name,'rb')
    preamble=json.load(preamblef)
    preamblef.close()

    run_name=lambda i: name+"_"+str(i)
    input_file_name=lambda i: os.path.join(path, name, run_name(i) + ".dat")
    supp_file_name=lambda i: os.path.join(path, name, run_name(i) + ".sup")
    nruns=preamble['Nruns']
    
    #
    # Open the data files (GENERIC)
    #
    input_file={}
    for ind in xrange(nruns):
        input_file[ind]=open(input_file_name(ind),'rb')

    supp_file={}
    for ind in xrange(nruns):
        supp_file[ind]=open(supp_file_name(ind),'rb')
            
            
    #
    # Prepare data entries (GENERIC)
    #

    #- prepare entries for experiment measurables
    if preamble['mids_recorded'] is []:
        pass
    else:
        for mid in preamble['mids_recorded']:
            data[mid]=[[] for ind in xrange(nruns)]

    #- prepare entries for update cycle reports
    if bool(preamble['ex_dataQ']):
        for mid in preamble['ex_data_recorded']:
            data[mid]=[[] for ind in xrange(nruns)]

    #- prepare entries for per-agent update cycle reports
    if bool(preamble['agent_dataQ']):
        for mid in preamble['agent_data_recorded']:
            for agent_id in preamble['agents']:
                data[(agent_id,mid)]=[[] for ind in xrange(NRUNS)]


    #
    # Read the data from the .dat files (GENERIC)
    #
    for ind in xrange(nruns):
        #load data from the supplementary files:
        supp[ind]=json.loads(supp_file[ind].readline())
        
        #load data from the data files:
        for record in get_pickles(input_file[ind]):
            #- read entries for experiment measurables
            if preamble['mids_recorded'] is []:
                pass
            else:
                for mid,item in zip(preamble['mids_recorded'],record['mids_recorded']):
                    data[mid][ind].append(item)
            #- read entries for experiment update cycle data
            if bool(preamble['ex_dataQ']):
                for tag,item in zip(preamble['ex_data_recorded'],record['ex_data_recorded']):
                    data[tag][ind].append(item)
            #- read entries for experiment update cycle data
            if bool(preamble['agent_dataQ']):
                for agent_id in preamble['agents']:
                    for tag,item in zip(preamble['agent_data_recorded'],record['agent_data_recorded'][agent_id]):
                        data[(agent_id,tag)][ind].append(item)

    # close the data & supplementary files:
    for ind in xrange(nruns):
        input_file[ind].close()
        supp_file[ind].close()
    return preamble

#
# Load the data
#

name=sys.argv[1]
path=os.getcwd()
DATA={}
SUPP={}
# Read the data and supplementary data from ./name/
preamble=get_data(path,name,DATA,SUPP)
NRUNS=preamble['Nruns']
BURN_IN=preamble['burn_in_cycles']

#
# Prepare the plots
#

DURATION=len(DATA['counter'][0])
DIAM=SUPP[0]['diam']
goodQ=lambda ind: DATA['dist'][ind][-1]<DIAM/2
badQ=lambda ind: not goodQ(ind)
good_indices=filter(goodQ,xrange(NRUNS))
bad_indices=filter(badQ,xrange(NRUNS))
Ngood=len(good_indices)
Nbad=len(bad_indices)
last_good=good_indices[-1]
last_bad=bad_indices[-1]

#
#Initialize the plots
#

#- initialize figure
fig,ax=plt.subplots()
#fig.suptitle('Error rates in learned PCRs over time',fontsize=10)
#plt.subplots_adjust(left=0.05,right=0.95,bottom=0.05,top=0.95)
ax.set_ylabel('distance from target',fontsize=10)
ax.set_xlabel('time elapsed (cycles)',fontsize=10)
t=np.array(range(DURATION)[BURN_IN-1:])

for ind in xrange(NRUNS):
    run=np.array(DATA['dist'][ind][BURN_IN-1:],dtype=int)
    if goodQ(ind):
        if ind==last_good:
            ax.plot(t,run,'-r',label=str(Ngood)+' runs satisfying condition')
        else:
            ax.plot(t,run,'-r')
    else:
        if ind==last_bad:
            ax.plot(t,run,'-b',label=str(Nbad)+' runs violating condition')
        else:
            ax.plot(t,run,'-b')

ax.plot([BURN_IN-1,BURN_IN-1],[-1,DIAM+1],'--b.',label='training period ends (t='+str(BURN_IN)+')')

ax.legend(loc='center right')

#Show the plots
plt.show()
