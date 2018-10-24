from __future__ import division
import sys
import json
import numpy as np
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

#Construct implications matrices for each run
#print DATA[('obs','raw_implications')][0][0]['minus']
#print convert_raw_implications(DATA[('obs','raw_implications')][0][0]['minus'])
#exit(0)
GROUND_WEIGHTS=[]
GROUND_IMP=[]
WEIGHTS=[]
IMPS=[]
IMPS_full=[]
DIFFS=[]
DIFFS_full=[]
VM=[]
for ind in xrange(NRUNS):
    FP=[np.array(item) for item in SUPP[ind]['footprints']] #for each run, load its sensor footprints
    vm=np.array(SUPP[ind]['values']) #for each run, load the values of each position
    VM.append(vm)
    #print VM
    THRESHOLD=SUPP[ind]['threshold'] #for each run, load its implication threshold
    L=len(FP) #the number of footprint vectors

    #Standard implications (inclusions) among footprints:
    std_imp_check=lambda x,y: all(x<=y)
    #Signal- and type-dependent implications
    if preamble['SnapType']=='qualitative':
        #Qualitative implications among the footprints:
        #- compute minimum value in the intersection of two footprints:
        lookup_val=lambda x,y: -1 if not sum(x*y) else np.extract(x*y,vm).min()
        #- check for implication x->y (x and y are footprints):
        imp_check=lambda x,y: qless(qmax(lookup_val(x,y),lookup_val(fcomp(x),fcomp(y))),lookup_val(x,fcomp(y)))
    else:
        #Thresholded value-based implications
        #- compute the ground-truth weight of the intersection of two footprints:
        lookup_val=lambda x,y: sum(x*y*vm)
        #- check for thresholded implication based on ground truth weight
        imp_check=lambda x,y: lookup_val(x,fcomp(y))<min(sum(vm)*THRESHOLD,lookup_val(x,y),lookup_val(fcomp(x),fcomp(y)),lookup_val(fcomp(x),y)) or (lookup_val(x,fcomp(y))==0 and lookup_val(y,fcomp(x))==0)


    #ground truth implications:
    GROUND_WEIGHTS.append(convert_weights([[lookup_val(FP[yind],FP[xind]) for xind in xrange(yind+1)] for yind in xrange(L)]))
    GROUND_IMP.append(np.matrix([[imp_check(FP[yind],FP[xind]) for xind in xrange(L)] for yind in xrange(L)],dtype=int))
    #print GROUND_IMP[-1]
    #construct implications from observer agent's "minus" snapshot:
    run_weights=[]
    run_imps=[]
    run_imps_full=[]
    run_diffs=[]
    run_diffs_full=[]
    for t in xrange(len(DATA['counter'][ind])):
        tmp_weights_matr=convert_weights(DATA[('obs','weights')][ind][t]['minus'])
        tmp_matr=convert_raw_implications(DATA[('obs','raw_implications')][ind][t]['minus'])
        tmp_matr_full=convert_full_implications(DATA[('obs','full_implications')][ind][t]['minus'])
        run_weights.append(tmp_weights_matr)
        run_imps.append(tmp_matr)
        run_imps_full.append(tmp_matr_full)
        run_diffs.append(ellone(tmp_matr,GROUND_IMP[ind]))
        run_diffs_full.append(ellone(tmp_matr_full,GROUND_IMP[ind]))
    WEIGHTS.append(run_weights)
    IMPS.append(run_imps)
    IMPS_full.append(run_imps_full)
    DIFFS.append(run_diffs)
    DIFFS_full.append(run_diffs_full)

#K=0
#print VM[K]
#print '\n'
#print GROUND_WEIGHTS[K]-WEIGHTS[K][100]
#print '\n'
#print WEIGHTS[K][100]
#print '\n'
#print GROUND_IMP[K]
#print '\n'
#print IMPS[K][100]-GROUND_IMP[0]

#exit(0)
#for ind in xrange(NRUNS):
#    print SUPP[ind]['footprints']
#    print GROUND_IMP[ind]
#exit(0)

#Prepare the axes
fig,ax=plt.subplots()
#my_img=ax.imshow(GROUND_IMP[0], cmap = 'Spectral', vmin = 0, vmax = 1)
#
#def plot_implications(matr):
#    my_img.set_data(matr)
#
#anim = animation.FuncAnimation(
#    fig=imps_fig,
#    func=plot_implications,
#    #init_func=animation_init,
#    frames=IMPS[1],
#    repeat=False,
#    save_count=preamble['total_cycles'],
#    interval=50,
#    )
#
#plt.show()
#anim.save('out.gif',writer='imagemagick')

#THE REAL THING
#exit(0)
plt.subplots_adjust(left=0.05,right=0.95,bottom=0.1,top=0.9)
fig.suptitle('Sniffy: l1-distance of the implication record to the ground truth\n as a function of time during training by random exploration',fontsize=22)
plt.xlabel('time elapsed (cycles)',fontsize=16)
plt.ylabel('l1-distance to ground truth implication matrix',fontsize=16)

#Form the plots
t=np.array(DATA['counter'][0])
dmean=np.mean(np.array(DIFFS),axis=0)
dmean_full=np.mean(np.array(DIFFS_full),axis=0)

dstd=np.std(np.array(DIFFS),axis=0)
dstd_full=np.std(np.array(DIFFS_full),axis=0)
plt.plot(t,dmean,'-r',alpha=1,label='Mean raw imps difference over '+str(NRUNS)+' runs')
plt.plot(t,dmean_full,'-b',alpha=1,label='Mean full imps difference over '+str(NRUNS)+' runs')
plt.fill_between(t,dmean-dstd,dmean+dstd,alpha=0.2,color='r',label='std. deviation over '+str(NRUNS)+' runs')
plt.fill_between(t,dmean_full-dstd_full,dmean_full+dstd_full,alpha=0.2,color='b',label='std. deviation over '+str(NRUNS)+' runs')
ax.legend()

#Show the plots
plt.show()