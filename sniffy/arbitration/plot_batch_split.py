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
    #input_file={}
    #for ind in xrange(nruns):
    #    input_file[ind]=open(input_file_name(ind),'rb')
    #
    #supp_file={}
    #for ind in xrange(nruns):
    #    supp_file[ind]=open(supp_file_name(ind),'rb')
            
            
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
	input_file=open(input_file_name(ind),'rb')
	supp_file=open(supp_file_name(ind),'rb')

        supp[ind]=json.loads(supp_file.readline())
        
        #load data from the data files:
        for record in get_pickles(input_file):
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
	#close the data & suppolementary files
	input_file.close()
	supp_file.close()

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
TYPE=preamble['SnapType']
VARN=preamble['PeakType']
ENV_TYPE=preamble['env_type']
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
try:
	last_good=good_indices[-1]
except:
	pass
try:
	last_bad=bad_indices[-1]
except:
	pass
#
#Initialize the plots
#

#- initialize figure
fig,ax=plt.subplots()
fig.suptitle('Sniffy w/arbiter on a '+ENV_TYPE+' of diameter '+str(DIAM)+':\ndistance to the target as a function of time during and after training',fontsize=18)
plt.subplots_adjust(left=0.1,right=0.98,bottom=0.13,top=0.92)
ALPHA=0.05
THICK=3
#ax.set_title(TYPE+', '+VARN,fontsize=36)
ax.set_ylabel('distance from target',fontsize=18)
ax.set_xlabel('time elapsed (cycles)',fontsize=18)
ax.tick_params(labelsize=18)
ax.set_yticks(range(0,DIAM+1,2))
SHOW_START=0#BURN_IN-1
t=np.array(range(DURATION)[SHOW_START:])

for ind in xrange(NRUNS):
    run=np.array(DATA['dist'][ind][SHOW_START:],dtype=int)
    if goodQ(ind):
        if ind==last_good:
            ax.plot(t,run,'-r',linewidth=THICK,alpha=ALPHA,label=str(Ngood)+' runs satisfying condition')
        else:
            ax.plot(t,run,'-r',linewidth=THICK,alpha=ALPHA)
    else:
        if ind==last_bad:
            ax.plot(t,run,'-b',linewidth=THICK,alpha=ALPHA,label=str(Nbad)+' runs violating condition')
        else:
            ax.plot(t,run,'-b',linewidth=THICK,alpha=ALPHA)

ax.plot([BURN_IN-1,BURN_IN-1],[-1,DIAM+1],'--b.',linewidth=THICK,label='training period ends (t='+str(BURN_IN)+')')

ax.legend(loc='best',fontsize=14)

#Show the plots
plt.show()
