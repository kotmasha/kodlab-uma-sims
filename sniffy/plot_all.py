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


#
# Load the data
#

# Paramaters

# axes are separated by environment and type
# - rows:
ENVS=['interval','circle']
# - columns:
TYPES={
    'qualitative':'QAT',
    'empirical':'EAT',
    'discounted':'DAT',
    }
# each axis has two plots:
VARIATIONS={
    'dull':'xkcd:blue',
    'sharp':'xkcd:red',
    }
# generate a set of keys to access data
KEYS=[(env,typ,varn) for env in ENVS for typ in TYPES.keys() for varn in VARIATIONS.keys()]

# Read the data and supplementary data from ./[env]/comp_[MODE]/
DATA={key:{} for key in KEYS}
SUPP={key:{} for key in KEYS}
for key in KEYS:
    env,typ,varn=key
    tmp_path=os.getcwd()
    tmp_name=env+'_'+TYPES[typ]+'_'+varn
    get_data(tmp_path,tmp_name,DATA[key],SUPP[key])

# length of the environment (due to differences between circle and interval):
ENV_LENGTH=lambda key: SUPP[key][0]['env_length']
# number of sensors:
NSENSORS=lambda key: SUPP[key][0]['Nsensors']
NIMPS=lambda key: 4.*pow(NSENSORS(key),2)
# duration of a run (assumed the same for all runs)
DURATION=lambda key: len(DATA[key]['counter'][0])

#
#Initialize the plots
#



#- initialize figure
fig,axes=plt.subplots(nrows=len(ENVS),ncols=len(TYPES),sharex=True,sharey=True)
#fig.suptitle('Error rates in learned PCRs over time',fontsize=10)
plt.subplots_adjust(left=0.05,right=0.95,bottom=0.05,top=0.95)

#- form the implications plots
for env,ax_row in zip(ENVS,axes):
    for typ,ax in zip (TYPES.keys(),ax_row):
        # prepare axis parameters
        ax.set_ylabel('distance from target',fontsize=10)
        ax.set_xlabel('time elapsed (cycles)',fontsize=10)
        ax.set_ylim(bottom=0.)
        
        # iterate over variations
        for varn in VARIATIONS.keys():
            key=(env,typ,varn)
            t=xrange(DURATION(key))
            data=np.array(DATA[key]['dist'])
            mean_dist=np.mean(data)
            sdiv_dist=np.std(data)
            
            ALPH=0.7 #foreground transparency coefficients
            BETA=0.2 #background transparency coefficients
        
            ax.fill_between(t,mean_dist-sdiv_dist,mean_dist+sdiv_dist,alpha=BETA,color=VARIATIONS[varn])
        
            ax.plot(t,mean_dist,linestyle='solid',linewidth=3,color=VARIATIONS[varn],alpha=ALPH)
        
            #ax.legend()


#Show the plots
plt.show()
