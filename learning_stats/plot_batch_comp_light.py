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
ENVS=['interval','circle','rnd_interval']
MODE=sys.argv[1] # 'teleport','lazy','walk','simple',...
VARIATIONS=['dull','sharp']
AGENT_TYPES={
    '_Q':['qualitative','xkcd:red'],
    '_Ev':['empirical value-based','xkcd:blue'],
    '_Dv':['discounted value-based','xkcd:forest green'],
    }

# Read the data and supplementary data from ./[env]/comp_[MODE]/
DATA={env:{} for env in ENVS}
SUPP={env:{} for env in ENVS}
for env in ENVS:
    tmp_path=os.path.join(os.getcwd(),env)
    get_data(tmp_path,'comp_'+MODE,DATA[env],SUPP[env])



# length of the environment (due to differences between circle and interval):
ENV_LENGTH=lambda env: SUPP[env][0]['env_length']
# number of sensors:
NSENSORS=lambda env: SUPP[env][0]['Nsensors']
NIMPS=lambda env: 4.*pow(NSENSORS(env),2)
# duration of a run (assumed the same for all runs)
DURATION=lambda env: len(DATA[env]['counter'][0])

#
#Initialize the plots
#



#- initialize figure
fig,axes=plt.subplots(nrows=len(VARIATIONS),ncols=len(ENVS),sharex=True,sharey=True)
#fig.suptitle('Error rates in learned PCRs over time',fontsize=10)
plt.subplots_adjust(left=0.08,right=0.99,bottom=0.09,top=0.92)

#- form the implications plots
for varn,ax_row in zip(VARIATIONS,axes):
    for env,ax in zip (ENVS,ax_row):
        # prepare axis parameters
	ax.set_title(env+', '+varn,fontsize=18)
        ax.set_xlabel('time elapsed (cycles)',fontsize=18)
	ax.set_ylabel('error rate',fontsize=18)
	ax.tick_params(labelsize=18)
        #ax.set_yscale('log')
        t=xrange(DURATION(env))
        # iterate over types
        for styp in AGENT_TYPES:
            typ=styp+('1' if varn=='dull' else '2') #form specific recorded type

            #weight_diffs=np.array(DATA[env]['wdiff'+typ])
            raw_diffs=np.array(DATA[env]['rdiff'+typ])/NIMPS(env)
            full_diffs=np.array(DATA[env]['fdiff'+typ])/NIMPS(env)

            #YMAX=1. #max(np.max(raw_diffs),np.max(full_diffs),np.max(weight_diffs))
            #ax.set_ylim(bottom=0.)#,top=YMAX)
        
            #means over runs
            #diff_weight_mean=np.mean(weight_diffs,axis=0)
            diff_raw_mean=np.mean(raw_diffs,axis=0)
            diff_full_mean=np.mean(full_diffs,axis=0)
        
            #standard deviations over runs
            #diff_weight_sdv=np.std(weight_diffs,axis=0)
            diff_raw_sdv=np.std(raw_diffs,axis=0)
            diff_full_sdv=np.std(full_diffs,axis=0)
        
            ALPH=0.7 #foreground transparency coefficients
            BETA=0.1 #background transparency coefficients
        
            #ax.fill_between(t,diff_weight_mean-diff_weight_sdv,diff_weight_mean+diff_weight_sdv,alpha=BETA,color=AGENT_TYPES[styp][1])
            ax.fill_between(t,diff_raw_mean-diff_raw_sdv,diff_raw_mean+diff_raw_sdv,alpha=BETA,color=AGENT_TYPES[styp][1])
            ax.fill_between(t,diff_full_mean-diff_full_sdv,diff_full_mean+diff_full_sdv,alpha=BETA,color=AGENT_TYPES[styp][1])
        
            #ax.plot(t,diff_weight_mean,linestyle='-',linewidth=3,color=AGENT_TYPES[styp][1],alpha=ALPH,label='Learned weights vs. Expected weights, '+AGENT_TYPES[styp][0])
            ax.plot(t,diff_raw_mean,linestyle='solid',linewidth=3,color=AGENT_TYPES[styp][1],alpha=ALPH,label='Learned PCR vs. Ground PCR, '+AGENT_TYPES[styp][0])
            ax.plot(t,diff_full_mean,linestyle='dashed',linewidth=3,color=AGENT_TYPES[styp][1],alpha=ALPH,label='Full implications vs. actual implications, '+AGENT_TYPES[styp][0])
        
            #ax.legend()


#Show the plots
plt.show()
