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

# Paramaters

# axes are separated by environment and type
# - rows:
ENVS=['interval','circle']
# - columns:
ORDERED_TYPES=['empirical','discounted','qualitative']
TYPES={
    'empirical':'EAT',
    'discounted':'DAT',
    'qualitative':'QAT',
    }
# each axis has two plots:
VARIATIONS={
    'dull':'xkcd:blue',
    'sharp':'xkcd:red',
    }
# generate a set of keys to access data
KEYS=[(env,typ,varn) for env in ENVS for typ in ORDERED_TYPES for varn in VARIATIONS.keys()]

# Read the data and supplementary data from ./[env]/comp_[MODE]/
DATA={key:{} for key in KEYS}
SUPP={key:{} for key in KEYS}
BURN={}
for key in KEYS:
    env,typ,varn=key
    tmp_path=os.getcwd()
    tmp_name=env+'_'+TYPES[typ]+'_'+varn
    preamble=get_data(tmp_path,tmp_name,DATA[key],SUPP[key])
    BURN[key]=preamble['burn_in_cycles']

# length of the environment (due to differences between circle and interval):
ENV_LENGTH=lambda key: SUPP[key][0]['env_length']
# number of sensors:
NSENSORS=lambda key: SUPP[key][0]['Nsensors']
# duration of a run (assumed the same for all runs)
DURATION=lambda key: len(DATA[key]['counter'][0])
# diameter of environment
DIAM=lambda key: SUPP[key][0]['diam']

#
#Initialize the plots
#

#- initialize figure
fig,axes=plt.subplots(nrows=len(ENVS),ncols=len(TYPES),sharex=True)
#fig.suptitle('Error rates in learned PCRs over time',fontsize=10)
plt.subplots_adjust(left=0.05,right=0.95,bottom=0.1,top=0.95)

#- form the implications plots
for env,ax_row in zip(ENVS,axes):
    for typ,ax in zip (ORDERED_TYPES,ax_row):
	burn=BURN[(env,typ,'dull')]
	diam=int(DIAM((env,typ,'dull')))
        # prepare axis parameters
        ax.set_title(env+', '+typ,fontsize=20)
        ax.set_ylabel('distance from target',fontsize=20)
        ax.set_xlabel('time elapsed (cycles)',fontsize=20)
	ax.tick_params(labelsize=20)
	ax.set_ylim(bottom=-1,top=diam+1)
	ax.set_yticks(range(0,diam+1,diam/5))
        #ax.set_ylim(bottom=0.)
        # iterate over variations
        for varn in VARIATIONS.keys():
            key=(env,typ,varn)
            start=2*burn-DURATION(key)
            finish=DURATION(key)
            t=xrange(start,finish)
            data=np.array(DATA[key]['dist'])
            #print data.shape
            mean_dist=np.mean(data,axis=0)[start:]
            sdiv_dist=np.std(data,axis=0)[start:]
            #print mean_dist.shape
            #print sdiv_dist.shape
            #exit(0)

            ALPHA=0.7 #foreground transparency coefficients
            BETA=0.1 #background transparency coefficients

            ax.fill_between(t,mean_dist-sdiv_dist,mean_dist+sdiv_dist,alpha=BETA,color=VARIATIONS[varn])

            ax.plot(t,mean_dist,linestyle='solid',linewidth=3,color=VARIATIONS[varn],alpha=ALPHA,label=varn+' peak value signal')

 	ax.plot([burn-1,burn-1],[-1,diam+1],'--b.',label='training period ends (t='+str(burn)+')')
	if env==ENVS[0] and typ==ORDERED_TYPES[0]:
	    ax.legend(fontsize=16)


#Show the plots
plt.show()
