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
    return np.matrix(newmatr,dtype=int)

def ellone(x,y):
    #assuming x,y are np arrays of the same shape, 
    #return the ell-1 distance between them:
    return np.sum(np.abs(x-y))


#
# Read the preamble (GENERIC)
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

#
# Prepare the plots (EXPERIMENT-SPECIFIC)
#

# length of the environment (due to differences between circle and interval)
ENV_LENGTH=len(SUPP['values'])
# duration of the experiment
DURATION=len(DATA['counter'])
# value of each position in the environment, needed as np.array
VM=np.array(SUPP['values'])
# extreme (=target) value
V_EXTREME=VM.min() if preamble['SnapType']=='qualitative' else VM.max()

# footprint for each sensor
FP=[np.array(item) for item in SUPP['footprints']]
L=len(FP) #the number of footprint vectors

# [initial] implication threshold for this run
THRESHOLD=SUPP['threshold']

#Form the implications matrices
#
#- define standard implications (inclusions) among footprints:
std_imp_check=lambda x,y: all(x<=y)
#- define signal- and type-dependent implications
if preamble['SnapType']=='qualitative':
    #Qualitative implications among the footprints:
    #- compute minimum value in the intersection of two footprints:
    lookup_val=lambda x,y: -1 if not sum(x*y) else np.extract(x*y,VM).min()
    #- check for implication x->y (x and y are footprints):
    imp_check=lambda x,y: qless(qmax(lookup_val(x,y),lookup_val(fcomp(x),fcomp(y))),lookup_val(x,fcomp(y)))
else:
    #Thresholded value-based implications
    #- compute the ground-truth weight of the intersection of two footprints:
    lookup_val=lambda x,y: sum(x*y*VM)
    #- check for thresholded implication based on ground truth weight
    imp_check=lambda x,y: lookup_val(x,fcomp(y))<min(sum(VM)*THRESHOLD,lookup_val(x,y),lookup_val(fcomp(x),fcomp(y)),lookup_val(fcomp(x),y)) or (lookup_val(x,fcomp(y))==0 and lookup_val(y,fcomp(x))==0)

#- construct the ground truth matrices
GROUND_WEIGHTS=convert_weights([[lookup_val(FP[yind],FP[xind]) for xind in xrange(yind+1)] for yind in xrange(L)])
GROUND_RAW_IMPS=np.matrix([[imp_check(FP[yind],FP[xind]) for xind in xrange(L)] for yind in xrange(L)],dtype=int)

#- construct matrices from data
WEIGHTS=[]
RAW_IMPS=[]
FULL_IMPS=[]
RAW_DIFFS=[]
FULL_DIFFS=[]
for t in xrange(DURATION):
    #- learned weight matrix at time t:
    #WEIGHTS.append(convert_weights(DATA[('obs','weights')][t]['minus']))
    
    #- learned PCR structure (learned raw implications) at time t:
    tmp_raw_imps=convert_raw_implications(DATA[('obs','raw_implications')][t]['minus'])
    RAW_IMPS.append(tmp_raw_imps)

    #- transitive closure of learned PCR at time t:
    tmp_full_imps=convert_full_implications(DATA[('obs','full_implications')][t]['minus'])
    FULL_IMPS.append(tmp_full_imps)

    #- ell-1 distance of learned PCR to ground truth PCR
    RAW_DIFFS.append(ellone(tmp_raw_imps,GROUND_RAW_IMPS))
    #- ell-1 distance of transitive closure to ground truth PCR (could be quite bigger)
    FULL_DIFFS.append(ellone(tmp_full_imps,GROUND_RAW_IMPS))

#Form state representation
#
#- prepare target ground truth
TARGET_GROUND=np.array([[abs(val-V_EXTREME)<pow(10,-10) for t in DATA['counter']] for val in SUPP['values']],dtype=int)
#- prepare position
POS=DATA['pos']
#- prepare upper and lower bounds for target according to observer
TARGET_TRAJECTORY=np.array(
    [[DATA['targ_foot'][t][pos] for t in xrange(DURATION)] for pos in xrange(ENV_LENGTH)])

#
#Initialize the plots
#
fig,(ax_imps,ax_env)=plt.subplots(nrows=2,ncols=1,sharex=True,sharey=False)
plt.subplots_adjust(left=0.05,right=0.95,bottom=0.1,top=0.9)
fig.suptitle('Sniffy observer: \# of incorrect implications in PCR over time',fontsize=10)
plt.xlabel('time elapsed (cycles)',fontsize=10)
ax_imps.set_ylabel('# incorrect implications',fontsize=10)
ax_env.set_ylabel('position',fontsize=10)
ax_env.set_title('Agent position and target state representation over time',fontsize=10)
#Form the plots
t=np.array(DATA['counter'])
diff_raw=np.array(RAW_DIFFS)
diff_full=np.array(FULL_DIFFS)
ax_imps.set_ylim(bottom=0,top=L*L)
ax_imps.plot(t,diff_raw,'-r',alpha=1,label='Implications in PCR')
ax_imps.plot(t,diff_full,'-b',alpha=1,label='Implications in transitive closure')
ax_imps.legend()

ax_env.yaxis.set_ticks(-0.5+np.arange(len(VM)))
#ax_env.xaxis.set_ticks(-0.5+np.arange(0,len(t),10))
ax_env.tick_params(labelbottom=True,labelleft=False)

#print TARG_TRAJECTORY
ax_env.imshow(TARGET_TRAJECTORY, cmap = plt.cm.Blues, vmin = 0, vmax = 1, aspect='auto',interpolation='none',alpha=0.5,extent=(1,len(t)+1,0,len(VM)-1))
ax_env.imshow(TARGET_GROUND, cmap = plt.cm.Reds, vmin = 0, vmax = 1, aspect='auto',interpolation='none',alpha=0.5,extent=(1,len(t)+1,0,len(VM)-1))
ax_env.plot(t,np.array(POS),'.r',alpha=1,label='Observer\'s position')
ax_env.legend()

#Show the plots
plt.show()
