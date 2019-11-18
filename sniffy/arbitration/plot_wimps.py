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
from matplotlib.widgets import Button


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

#np.set_printoptions(threshold=np.nan)
#
### CONSTANTS
#

BURN_IN=preamble['burn_in_cycles']
ENV_LENGTH=SUPP['env_length']
DIAM=SUPP['diam']
NSENSORS=SUPP['Nsensors']

#
### Select the agent:
#
agent='arb'
ANAME={'rt':'RT','lt':'LT','arb':'ARB'}
ALL_LABELS=SUPP[agent+'_LabelsPlus']

#
### Select the sensors for which to display implications:
#

#LABELS=ALL_LABELS
LABELS=['F','F*','toF','toF*','#F','#F*','#toF','#toF*']
#LABELS=['x0','x1','x2','x3','x4','x5','x6','x7','x8','x9',
#       '#x0','#x1','#x2','#x3','#x4','#x5','#x6','#x7','#x8','#x9']


#
### Create data representation
#

MASK=[x in LABELS for x in ALL_LABELS]
DIM=len(LABELS)
TICKS=np.arange(DIM)

#time counter
time_counter=np.array(DATA['counter'])
DURATION=len(time_counter)

#arbiter implications
IMPS={}
tokens=['+','-']
for tok in tokens:
    IMPS[tok]=np.array([np.matrix(item)[np.ix_(MASK,MASK)] for item in DATA['imps_'+agent+tok]])

def summ(tok,s,t):
    return np.sum(IMPS[tok][s:t],axis=0)

class repn():
    def __init__(self,time):
        self.time=time
        fig,axes=plt.subplots(nrows=1,ncols=2)#,sharey=True)
        plt.subplots_adjust(left=0.05,right=0.95,bottom=0.1,top=0.9)
        self.fig=fig
        self.fig.suptitle('Sniffy '+ANAME[agent]+': mean implication matrices for t=1,...,'+str(time))
        self.ax={}
        self.data={}
        self.image={}
        for ind,tok in enumerate(tokens):
            self.ax[tok]=axes[ind]
            self.ax[tok].set_title(('Active ' if tok=='+' else 'Inactive ')+'snapshot')
            self.ax[tok].set_xticks([],minor=False) #remove major y-ticks
            self.ax[tok].set_xticks(TICKS,minor=True) #set minor y-ticks
            self.ax[tok].set_xticklabels(LABELS,minor=True) #set minor y-tick labels
            self.ax[tok].set_yticks([],minor=False) #same for x-ticks...
            self.ax[tok].set_yticks(TICKS,minor=True)
            self.ax[tok].set_yticklabels(LABELS,minor=True)
            self.ax[tok].tick_params(axis='x',which='minor',labelrotation=90) #make x-ticks vertical
            self.data[tok]=summ(tok,0,time)*(1./(0.+time))
            self.image[tok]=self.ax[tok].imshow(self.data[tok],cmap='Reds',vmin=0.,vmax=1)
                                     
        #buttons
        axstart=plt.axes([0.05,0.05,0.1,0.075])
        axbb=plt.axes([0.21,0.05,0.1,0.075])
        axprev=plt.axes([0.37,0.05,0.1,0.075])
        axnext=plt.axes([0.53,0.05,0.1,0.075])
        axff=plt.axes([0.69,0.05,0.1,0.075])
        axend=plt.axes([0.85,0.05,0.1,0.075])
        bstart=Button(axstart, 'start')
        bstart.on_clicked(self.start)
        bbb=Button(axbb,'rew')
        bbb.on_clicked(self.bb)
        bprev=Button(axprev,'prev')
        bprev.on_clicked(self.prev)
        bnext=Button(axnext,'next')
        bnext.on_clicked(self.next)
        bff=Button(axff,'ffwd')
        bff.on_clicked(self.ff)
        bend=Button(axend,'end')
        bend.on_clicked(self.end)
        axstop=plt.axes([.85,.85,.1,0.075])
        bstop=Button(axstop,'exit')
        bstop.on_clicked(self.stop)
        plt.show()

    def update_title(self):
        self.fig.suptitle('Sniffy '+ANAME[agent]+': mean implication matrices for t=1,...,'+str(self.time))

    def move(self,deltat):
        if deltat>0:
            s=self.time
            t=min(self.time+deltat,time_counter[-1])
            for tok in tokens:
                self.data[tok]=summ(tok,0,t)*(1./(0.+t))
                self.image[tok].set_data(self.data[tok])
            self.time=min(self.time+deltat,time_counter[-1])
        else:
            s=max(1,self.time+deltat)
            t=self.time
            for tok in tokens:
                self.data[tok]=summ(tok,0,s)*(1./(0.+s))
                self.image[tok].set_data(self.data[tok])
            self.time=max(1,self.time+deltat)
        self.update_title()
        plt.draw()
        
    def next(self,event):
        self.move(1)
    def prev(self,event):
        self.move(-1)
    def ff(self,event):
        self.move(10)
    def bb(self,event):
        self.move(-10)
    def start(self,event):
        self.move(-self.time)
    def end(self,event):
        self.move(DURATION)
    def stop(self,event):
        exit(0)
        

ARB=repn(1)
#plt.show()
