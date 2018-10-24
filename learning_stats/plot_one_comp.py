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
    return np.matrix(newmatr)

def ellone(x,y):
    #assuming x,y are np arrays of the same shape, 
    #return the ell-1 distance between them:
    return np.sum(np.abs(x-y))


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

#
# Prepare the plots (EXPERIMENT-SPECIFIC)
#
    
AGENT_TYPES={
    '_Q':['qualitative','xkcd:red'],
    '_Eu':['empirical uniform','xkcd:sky blue'],
    '_Ev':['empirical value-based','xkcd:blue'],
    '_Du':['discounted uniform','xkcd:lavender'],
    '_Dv':['discounted value-based','xkcd:purple'],
    }
#ORDERED_TYPES=['_Q','_Eu','_Ev','_Du','_Dv']
ORDERED_TYPES=['_Q','_Eu','_Ev','_Du','_Dv']
NTYPES=len(ORDERED_TYPES)

# length of the environment (due to differences between circle and interval)
ENV_LENGTH={typ:len(SUPP['values'][typ]) for typ in ORDERED_TYPES}
# duration of the experiment
DURATION=len(DATA['counter'])

# value of each position in the environment, needed as np.array
vm=lambda typ: np.array(SUPP['values'][typ])
# extreme (=target) value
v_extreme=lambda typ: vm(typ).min() if typ=='_Q' else vm(typ).max()

# footprint for each sensor
FP=[np.array(item) for item in SUPP['footprints']]
Nsensors=len(FP) #the number of footprint vectors

# [initial] implication threshold for this run
THRESHOLD=SUPP['threshold']

#Form the implications matrices
#
#- check for inclusions among footprints:
std_imp_check=lambda x,y: all(x<=y)
#- check for ground truth thresholded inclusions
def lookup_val(x,y,typ):
    if typ=='_Q':
        return -1 if not sum(x*y) else np.extract(x*y,vm(typ)).min()
    else:
        return sum(x*y*vm(typ))

def imp_check(x,y,typ):
    XY=lookup_val(x,y,typ)
    X_Y=lookup_val(fcomp(x),y,typ)
    X_Y_=lookup_val(fcomp(x),fcomp(y),typ)
    XY_=lookup_val(x,fcomp(y),typ)
    total=sum(vm(typ))
    if typ=='_Q': #qualitative implication (zero threshold)
        return qless(qmax(XY,X_Y_),XY_)
    else: #real-valued (statistical) implication
        return XY_<min(total*THRESHOLD,XY,X_Y_,X_Y) or (XY_==0 and X_Y==0)

#- construct the ground truth PCR matrices
GROUND_WEIGHTS={}
GROUND_RAW_IMPS={}
ABS_GROUND_RAW_IMPS={}
for typ in ORDERED_TYPES:
    lookup=lambda x,y: lookup_val(x,y,typ)
    check=lambda x,y: imp_check(x,y,typ)
    # Weight matrix computed from known values of states
    GROUND_WEIGHTS[typ]=np.matrix([[lookup(FP[xind],FP[yind]) for xind in xrange(Nsensors)] for yind in xrange(Nsensors)])
    # PCR matrix computed from known values of states; note the transpose!!
    GROUND_RAW_IMPS[typ]=np.matrix([[check(FP[yind],FP[xind]) for xind in xrange(Nsensors)] for yind in xrange(Nsensors)],dtype=int)
ABS_GROUND_RAW_IMPS=np.matrix([[std_imp_check(FP[yind],FP[xind]) for xind in xrange(Nsensors)] for yind in xrange(Nsensors)])

#- construct matrices from data
WEIGHTS={typ:[] for typ in ORDERED_TYPES}
RAW_IMPS={typ:[] for typ in ORDERED_TYPES}
FULL_IMPS={typ:[] for typ in ORDERED_TYPES}
RAW_DIFFS={typ:[] for typ in ORDERED_TYPES}
FULL_DIFFS={typ:[] for typ in ORDERED_TYPES}
STD_DIFFS={typ:[] for typ in ORDERED_TYPES}
for typ in ORDERED_TYPES:
    for t in xrange(DURATION):
        #- learned weight matrix at time t:
        try:
            WEIGHTS[typ].append(convert_weights(DATA[('obs'+typ,'weights')][t]['minus']))
        except:
            print DATA[('obs'+typ,'weights')][t]['minus']
            raise Exception('')
        #- learned PCR structure (learned raw implications) at time t:
        tmp_raw_imps=convert_raw_implications(DATA[('obs'+typ,'raw_implications')][t]['minus'])
        RAW_IMPS[typ].append(tmp_raw_imps)

        #- transitive closure of learned PCR at time t:
        tmp_full_imps=convert_full_implications(DATA[('obs'+typ,'full_implications')][t]['minus'])
        FULL_IMPS[typ].append(tmp_full_imps)

        #- ell-1 distance of learned PCR to ground truth PCR
        RAW_DIFFS[typ].append(ellone(tmp_raw_imps,GROUND_RAW_IMPS[typ]))
        #- ell-1 distance of transitive closure to ground truth PCR (could be quite bigger)
        FULL_DIFFS[typ].append(ellone(tmp_full_imps,GROUND_RAW_IMPS[typ]))
        STD_DIFFS[typ].append(ellone(tmp_full_imps,ABS_GROUND_RAW_IMPS))

#Form state representation
#
#- prepare Sniffy's position
POS=np.array(DATA['pos'])
#- prepare target representation
TARGET_GROUND={}
TARGET_TRAJECTORY={}
for typ in ORDERED_TYPES:
    #- prepare target ground truth
    TARGET_GROUND[typ]=np.array([[abs(val-v_extreme(typ))<pow(10,-10) for t in xrange(DURATION)] for val in vm(typ)],dtype=int)
    #- prepare footpritns for target according to observer
    TARGET_TRAJECTORY[typ]=np.array([[all([FP[ind][pos] for ind,val in enumerate(DATA['targ'+typ][t]) if val]) for t in xrange(DURATION)] for pos in xrange(ENV_LENGTH[typ])])

#
#Initialize the plots
#

#- initialize figure
fig,axes=plt.subplots(nrows=1+NTYPES,ncols=1,sharex=True,sharey=False)
fig.suptitle('# of incorrect implications over time',fontsize=10)
plt.subplots_adjust(left=0.05,right=0.95,bottom=0.05,top=0.95)
plt.xlabel('time elapsed (cycles)',fontsize=10)

#- initialize implications plot
ax_imps=axes[0]
ax_imps.set_ylabel('# incorrect implications',fontsize=10)
ax_imps.set_ylim(bottom=0,top=Nsensors*Nsensors)

#- associate axes to types
AX={}
for typ,ax in zip(ORDERED_TYPES,axes[1:]):
    AX[typ]=ax
    AX[typ].set_title('Target state representation over time, '+AGENT_TYPES[typ][0],fontsize=10)
    AX[typ].set_ylabel('position in env',fontsize=10)
    AX[typ].yaxis.set_ticks(-0.5+np.arange(ENV_LENGTH[typ]))
    AX[typ].tick_params(labelbottom=True,labelleft=False)

#- form the implications plot
t=np.array(DATA['counter'])
for typ in ORDERED_TYPES:
    diff_raw=np.array(RAW_DIFFS[typ])
    diff_full=np.array(FULL_DIFFS[typ])
    diff_std=np.array(STD_DIFFS[typ])
    ALPH=0.7
    ax_imps.plot(t,diff_raw,linestyle='solid',linewidth=3,color=AGENT_TYPES[typ][1],alpha=ALPH,label='Learned PCR vs. Ground PCR, '+AGENT_TYPES[typ][0])
    ax_imps.plot(t,diff_full,linestyle='dashed',linewidth=3,color=AGENT_TYPES[typ][1],alpha=ALPH,label='Full implications vs. Ground PCR, '+AGENT_TYPES[typ][0])
    ax_imps.plot(t,diff_std,linestyle='dotted',linewidth=2,color=AGENT_TYPES[typ][1],alpha=ALPH,label='Full implications vs. Set implications, '+AGENT_TYPES[typ][0])
ax_imps.legend()

#- form the trajectories plots
for typ in ORDERED_TYPES:
    EXTENT=(1,DURATION+1,0,ENV_LENGTH[typ]-1)
    AX[typ].imshow(TARGET_TRAJECTORY[typ], cmap = plt.cm.Blues, vmin = 0, vmax = 1, aspect='auto',interpolation='none',alpha=0.5,extent=EXTENT)
    AX[typ].imshow(TARGET_GROUND[typ], cmap = plt.cm.Reds, vmin = 0, vmax = 1, aspect='auto',interpolation='none',alpha=0.5,extent=EXTENT)
    AX[typ].plot(t-1,np.array(POS),'.r',alpha=1,label='Observer\'s position')
    AX[typ].legend()

#Show the plots
plt.show()