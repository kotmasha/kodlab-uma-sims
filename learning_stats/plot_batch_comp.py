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

def compi(x):
    if type(x)==type(0):
        return x+1 if x%2==0 else x-1
    else:
        raise Exception('Input to \"compi\" must be an integer! \n')

def fcomp(x):
    #assuming x is a footprint and an np.array:
    return 1-x

# inequality check for qualitative weights: "IS x strictly less than y?"
def qless(x,y):
    if x<0: #infinity is never less than anything
        return False
    elif y<0: #anything finite is less than infinity
        return True
    else: #finite things compared as usual
        return x<y

# max function for qualitative weights
def qmax(*args):
    if min(args)<0:
        return -1
    else:
        return max(args)

def qmin(*args):
    if max(args)<0:
        return -1
    else:
        return min(filter(lambda x: x>=0, args))

# convert npdirs data into a matrix
def convert_full_implications(matr):
    L=len(matr)
    for i in range(L):
        for j in range(L):
            if j >= len(matr[i]):
                matr[i].append(matr[compi(j)][compi(i)])
    return np.matrix(matr,dtype=int)

# convert dirs data into a matrix
def convert_raw_implications(matr):
    L=len(matr)
    for i in range(L):
        for j in range(L):
            if j >= len(matr[i]):
                try:
                    matr[i].append(matr[compi(j)][compi(i)])
                except IndexError:
                    matr[i].append(False)
    return np.matrix(matr,dtype=int)

# convert weights data into a matrix
def convert_weights(matr):
    L=len(matr)
    newmatr=[[] for ind in xrange(L)]
    for i in xrange(L):
        for j in xrange(L):
            if j>=len(matr[i]):
                newmatr[i].append(matr[j][i])
            else:
                newmatr[i].append(matr[i][j])
    return np.matrix(newmatr)

def ellone(x,y):
    #assuming x,y are np arrays of the same shape, 
    #return the ell-1 distance between them:
    return np.sum(np.abs(x-y))


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
#ORDERED_TYPES=['_Q','_Eu','_Ev','_Du','_Dv']
ORDERED_TYPES=['_Eu','_Du']
#ORDERED_TYPES=['_Q','_Ev','_Dv']
NTYPES=len(ORDERED_TYPES)

# length of the environment (due to differences between circle and interval)
env_length=lambda ind,typ: len(SUPP[ind]['values'][typ])
# duration of the experiment for each run
duration=lambda ind: len(DATA['counter'][ind])
def duration_gen():
    for ind in xrange(NRUNS):
        yield duration(ind)
DURATION=max(duration_gen())

# value of each position in the environment, needed as np.array, for each run and type
vm=lambda ind,typ: np.array(SUPP[ind]['values'][typ])
# extreme (=target) value of signal for each run and type
v_extreme=lambda ind,typ: vm(ind,typ).min() if typ=='_Q' else vm(ind,typ).max()

# sensor footprints for each run
fp=lambda ind,sensor_ind: np.array(SUPP[ind]['footprints'][sensor_ind]) #compute a footprint vector
# number of sensor for each run
Nsensors=lambda ind: len(SUPP[ind]['footprints'])   #the number of footprint vectors
def Nsensors_gen():
    for ind in xrange(NRUNS):
        yield Nsensors(ind)
NSENSORS=max(Nsensors_gen())

# [initial] implication threshold for each run
threshold=lambda ind: SUPP[ind]['threshold']

#Form the implications matrices
#
#- check for inclusions among footprints:
std_imp_check=lambda x,y: all(x<=y)
#- check for ground truth thresholded inclusions
def lookup_val(x,y,ind,typ):
    if typ=='_Q':
        return -1 if not sum(x*y) else np.extract(x*y,vm(ind,typ)).min()
    else:
        return (0.+sum(x*y*vm(ind,typ)))/env_length(ind,typ)

def imp_check(x,y,ind,typ):
    XY=lookup_val(x,y,ind,typ)
    X_Y=lookup_val(fcomp(x),y,ind,typ)
    X_Y_=lookup_val(fcomp(x),fcomp(y),ind,typ)
    XY_=lookup_val(x,fcomp(y),ind,typ)
    total=sum(vm(ind,typ))
    if typ=='_Q': #qualitative implication (zero threshold)
        return qless(qmax(XY,X_Y_),XY_)
    else: #real-valued (statistical) implication
        return XY_<min(total*threshold(ind),XY,X_Y_,X_Y) or (XY_==0 and X_Y==0)# and XY>0 and X_Y_>0)

#- construct the ground truth PCR matrices for each run and type
GROUND_WEIGHTS={typ:[] for typ in ORDERED_TYPES}
GROUND_RAW_IMPS={typ:[] for typ in ORDERED_TYPES}
ABS_GROUND_RAW_IMPS=[]
for typ in ORDERED_TYPES:
    for ind in xrange(NRUNS):
        lookup=lambda x,y: lookup_val(x,y,ind,typ)
        check=lambda x,y: imp_check(x,y,ind,typ)
        # Weight matrix computed from known values of states
        GROUND_WEIGHTS[typ].append(np.matrix([[lookup(fp(ind,yind),fp(ind,xind)) for xind in xrange(Nsensors(ind))] for yind in xrange(Nsensors(ind))]))
        # PCR matrix computed from known values of states; note the transpose!!
        GROUND_RAW_IMPS[typ].append(np.matrix([[check(fp(ind,yind),fp(ind,xind)) for xind in xrange(Nsensors(ind))] for yind in xrange(Nsensors(ind))],dtype=int))
for ind in xrange(NRUNS):
    ABS_GROUND_RAW_IMPS.append(np.matrix([[std_imp_check(fp(ind,yind),fp(ind,xind)) for xind in xrange(Nsensors(ind))] for yind in xrange(Nsensors(ind))],dtype=int))


#- construct matrices from data
WEIGHTS={typ:[[] for ind in xrange(NRUNS)] for typ in ORDERED_TYPES}
RAW_IMPS={typ:[[] for ind in xrange(NRUNS)] for typ in ORDERED_TYPES}
FULL_IMPS={typ:[[] for ind in xrange(NRUNS)] for typ in ORDERED_TYPES}
WEIGHT_DIFFS={typ:[[] for ind in xrange(NRUNS)] for typ in ORDERED_TYPES}
RAW_DIFFS={typ:[[] for ind in xrange(NRUNS)] for typ in ORDERED_TYPES}
FULL_DIFFS={typ:[[] for ind in xrange(NRUNS)] for typ in ORDERED_TYPES}
STD_DIFFS={typ:[[] for ind in xrange(NRUNS)] for typ in ORDERED_TYPES}
for typ in ORDERED_TYPES:
    for ind in xrange(NRUNS):
        for t in xrange(DURATION):
            #- learned weight matrix at time t:
            tmp_raw_weights=convert_weights(DATA[('obs'+typ,'weights')][ind][t]['minus'])
            WEIGHTS[typ][ind].append(tmp_raw_weights)

            #- learned PCR structure (learned raw implications) at time t:
            tmp_raw_imps=convert_raw_implications(DATA[('obs'+typ,'raw_implications')][ind][t]['minus'])
            RAW_IMPS[typ][ind].append(tmp_raw_imps)

            #- transitive closure of learned PCR at time t:
            tmp_full_imps=convert_full_implications(DATA[('obs'+typ,'full_implications')][ind][t]['minus'])
            FULL_IMPS[typ][ind].append(tmp_full_imps)

            #- ell-1 distance of learned weights to ground truth weights
            WEIGHT_DIFFS[typ][ind].append((np.abs(tmp_raw_weights-GROUND_WEIGHTS[typ][ind])).max())
            #- ell-1 distance of learned PCR to ground truth PCR
            RAW_DIFFS[typ][ind].append(ellone(tmp_raw_imps,GROUND_RAW_IMPS[typ][ind]))
            #- ell-1 distance of transitive closure to ground truth PCR (could be quite bigger)
            FULL_DIFFS[typ][ind].append(ellone(tmp_full_imps,GROUND_RAW_IMPS[typ][ind]))
            STD_DIFFS[typ][ind].append(ellone(tmp_full_imps,ABS_GROUND_RAW_IMPS[ind]))

#
#Initialize the plots
#

#- initialize figure
fig,ax_imps=plt.subplots(nrows=len(ORDERED_TYPES),ncols=1,sharex=True,sharey=True)
fig.suptitle('# of incorrect implications over time',fontsize=10)
plt.subplots_adjust(left=0.05,right=0.95,bottom=0.05,top=0.95)
plt.xlabel('time elapsed (cycles)',fontsize=10)

#- form the implications plots
AX={}
t=xrange(DURATION)
for typ,ax in zip(ORDERED_TYPES,ax_imps):
    AX[typ]=ax
    ax.set_ylabel('% incorrect implications',fontsize=10)
    
    SKIP=DURATION/100
    OFFSET=1

    weight_diffs=np.array(WEIGHT_DIFFS[typ])
    raw_diffs=np.array(RAW_DIFFS[typ])/pow(NSENSORS,2)
    full_diffs=np.array(FULL_DIFFS[typ])/pow(NSENSORS,2)
    std_diffs=np.array(STD_DIFFS[typ])/pow(NSENSORS,2)
    #YMAX=max(np.max(raw_diffs),np.max(full_diffs),np.max(std_diffs))
    ax.set_ylim(bottom=0.,top=1.)

    #means over runs
    diff_weight_mean=np.mean(weight_diffs,axis=0)
    diff_raw_mean=np.mean(raw_diffs,axis=0)
    diff_full_mean=np.mean(full_diffs,axis=0)
    diff_std_mean=np.mean(std_diffs,axis=0)
    #standard deviations over runs
    diff_weight_sdv=np.std(weight_diffs,axis=0)
    diff_raw_sdv=np.std(raw_diffs,axis=0)
    diff_full_sdv=np.std(full_diffs,axis=0)
    diff_std_sdv=np.std(std_diffs,axis=0)

    ALPH=0.7 #foreground transparency coefficients
    BETA=0.2 #background transparency coefficients

    ax.fill_between(t,diff_weight_mean-diff_weight_sdv,diff_weight_mean+diff_weight_sdv,alpha=BETA,color=AGENT_TYPES[typ][1])
    ax.fill_between(t,diff_raw_mean-diff_raw_sdv,diff_raw_mean+diff_raw_sdv,alpha=BETA,color=AGENT_TYPES[typ][1])
    #ax.fill_between(t,diff_full_mean-diff_full_sdv,diff_full_mean+diff_full_sdv,alpha=BETA,color=AGENT_TYPES[typ][1])
    #ax.fill_between(t,diff_std_mean-diff_std_sdv,diff_std_mean+diff_std_sdv,alpha=BETA,color=AGENT_TYPES[typ][1])
    
    ax.plot(t,diff_weight_mean,linestyle='--',linewidth=3,color=AGENT_TYPES[typ][1],alpha=ALPH,label='Learned weights vs. Ground weights, '+AGENT_TYPES[typ][0])
    ax.plot(t,diff_raw_mean,linestyle='solid',linewidth=3,color=AGENT_TYPES[typ][1],alpha=ALPH,label='Learned PCR vs. Ground PCR, '+AGENT_TYPES[typ][0])
    #ax.plot(t,diff_full_mean,linestyle='dashed',linewidth=3,color=AGENT_TYPES[typ][1],alpha=ALPH,label='Full implications vs. Ground PCR, '+AGENT_TYPES[typ][0])
    #ax.plot(t,diff_std_mean,linestyle='dotted',linewidth=2,color=AGENT_TYPES[typ][1],alpha=ALPH,label='Full implications vs. Set implications, '+AGENT_TYPES[typ][0])

    ax.legend()
#ax_imps.legend()

#Show the plots
plt.show()
