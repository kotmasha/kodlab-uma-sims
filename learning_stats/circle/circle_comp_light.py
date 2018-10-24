import numpy as np
from operator import sub
from operator import concat
from numpy.random import randint as rnd
from numpy.random import seed
from collections import deque
from functools import partial
#import curses
import time
from UMA.som2_noEP import *
import sys
import os
import json
from client.UMARest import *


############### AUXILIARY FUNCTIONS
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
    new_matr=np.zeros((L,L),dtype=int)
    for i in range(L):
        for j in range(L):
            if j >= len(matr[i]):
                try:
                    new_matr[i][j]+=matr[compi(j)][compi(i)]
                    #matr[i].append(matr[compi(j)][compi(i)])
                except IndexError:
                    pass
            else:
                new_matr[i][j]+=matr[i][j]
                    #matr[i].append(False)
    return new_matr

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

def ldiff(x,y): # pointwise difference between lists
    return map(abs,map(sub,x,y))

def flatten(ls): # concatenation of elements in a list of lists
    return reduce(concat, ls)

###################################


def start_experiment(run_params):
    # set randmization seed
    SEED=seed()
    # System parameters
    test_name=run_params['test_name']
    host = run_params['host']
    port = run_params['port']

    # initialize a new experiment
    EX = Experiment(test_name, UMARestService(host, port))
    id_dec = 'decision'
    id_count= 'counter'

    # Recording options:
    record_mids=run_params['mids_to_record']
    record_global=run_params['ex_dataQ']
    record_agents=run_params['agent_dataQ']
    #A recorder will be initialized later, at the end of the initialization phase,
    #to enable collection of all available data tags

    # Decision cycles:
    TOTAL_CYCLES = run_params['total_cycles']
    # Parameters and definitions
    MODE=run_params['mode'] #mode by which Sniffy moves around: 'teleport'/'walk'/'lazy'
    X_BOUND = run_params['env_length']  # no. of edges in discrete interval = no. of GPS sensors
    BEACON_WIDTH = run_params['beacon_width'] # [max] width of beacon sensor
    ENV_LENGTH=X_BOUND
    NSENSORS=X_BOUND

    try:
        Discount=float(run_params['discount']) #discount coefficient, if any
    except KeyError:
        Discount=0.75
    try:
        Threshold=float(run_params['threshold']) #implication threshold, defaulting to the square of the probability of a single position.
    except KeyError:
        Threshold=1./pow(ENV_LENGTH,2)

    # Environment description
    def in_bounds(pos):
        return (pos >= 0 and pos < X_BOUND)
    def dist(p, q):
        return min(abs(p - q),X_BOUND-abs(p-q)) #distance between two points in environment

    # agent parameters according to .yml file

    empirical_observer={
        'type': 'empirical',
        'AutoTarg': True,
        'threshold': Threshold,
    }

    discounted_observer={
        'type': 'discounted',
        'q': Discount,
        'AutoTarg': True,
        'threshold': Threshold,
    }

    qualitative_observer={
        'type': 'qualitative',
        'AutoTarg': True,
        'threshold': 0,
    }

    AGENT_PARAMS={
        '_Q':qualitative_observer,
        '_Eu':empirical_observer,
        '_Ev':empirical_observer,
        '_Du':discounted_observer,
        '_Dv':discounted_observer,
        }
    ORDERED_TYPES=['_Q','_Eu','_Ev','_Du','_Dv']
 
    #Register "observer" agents:
    #  These agents remain inactive throghout the experiment, in order to record 
    #  all the UNCONDITIONAL implications among the initial sensors (in their 'minus'
    #  snapshots).
    #  For this purpose, each sensor's FOOTPRINT in the state space (positions in
    #  the interval) is recorded, so that implications may be calculated according
    #  to the inclusions among footprints.
    id_obs={}
    cid_obs={}
    for typ in ORDERED_TYPES:
        id_obs[typ],cid_obs[typ]=EX.register_sensor('obs'+typ)

    # register motivation for motion agents
    # - this one is NOT dependent on agents except through the position, so
    #   it carries the default "decdep=False" tag.

    # Value signals for different setups determined as a *function* of distance to target
    id_dist = EX.register('dist')
    id_sig={}
    for typ in ORDERED_TYPES:
        id_sig[typ]=EX.register('sig'+typ)

    # ...which function? THIS function:
    RESCALING={
        #'_Q': lambda r: r,
        '_Q': lambda r: 1 if r>0 else 0,
        '_Eu': lambda r: 1,
        '_Ev': lambda r: pow((X_BOUND/2)-r,1),
        '_Du': lambda r: 1,
        '_Dv': lambda r: pow(1.-Discount,r-(X_BOUND/2)),
        }


    # OBSERVER agents simply collect implications among the assigned sensors, always inactive
    OBSERVERS={}
    OBSACCESS={}
    for typ in ORDERED_TYPES:
        OBSERVERS[typ]=EX.construct_agent(id_obs[typ],id_sig[typ],lambda state: False,AGENT_PARAMS[typ])
        OBSACCESS[typ]=UMAClientData(EX._EXPERIMENT_ID,id_obs[typ],'minus',EX._service)


    #
    ### "mapping" system
    #

    ## introduce agent's position

    # select starting position
    START = rnd(X_BOUND)

    # effect of motion on position
    id_pos = EX.register('pos')

    def walk_through(state):
        newpos = (state[id_pos][0] + 1) % X_BOUND
        return newpos

    def random_walk(state):
        diff = 2*rnd(2)-1
        newpos = (state[id_pos][0] + diff) % X_BOUND
        return newpos

    def lazy_random_walk(state):
        diff = rnd(3)-1
        newpos = (state[id_pos][0] + diff) % X_BOUND
        return newpos

    def teleport(state):
        return rnd(X_BOUND)

    motions={'simple':walk_through,'walk':random_walk,'lazy':lazy_random_walk,'teleport':teleport}
    EX.construct_measurable(id_pos,motions[MODE],[START,START])

    # generate target position
    TARGET=START
    while dist(TARGET, START)==0:
        TARGET = rnd(ENV_LENGTH)

    # set up position sensors
    def xsensor(m,width):  # along x-axis
        return lambda state: dist(state[id_pos][0],m)<=width

    # Construct initial sensors and record their semantics
    FOOTPRINTS=[]
    all_comp=lambda x: [1-t for t in x]
    for ind in xrange(X_BOUND):
        # create new beacon sensor centered at $ind$
        tmp_name = 'x' + str(ind)
        id_tmp, id_tmpc = EX.register_sensor(tmp_name)  # registers the sensor pairs
        EX.construct_sensor(id_tmp, xsensor(ind,BEACON_WIDTH))  # constructs the measurables associated with the sensor
        for typ in ORDERED_TYPES:
            OBSERVERS[typ].add_sensor(id_tmp)
        # record the sensor footprint
        tmp_footprint=[(1 if dist(pos,ind)<=BEACON_WIDTH else 0) for pos in xrange(X_BOUND)]
        FOOTPRINTS.append(tmp_footprint)
        FOOTPRINTS.append(all_comp(tmp_footprint))

    # sensor footprints for this run
    fp=lambda sensor_ind: np.array(FOOTPRINTS[sensor_ind])

    #get internal data for each agent
    id_internal={}
    def get_internal(state,typ):
        return OBSACCESS[typ].get_all()
    INIT={}
    for typ in ORDERED_TYPES:
        id_internal[typ]=EX.register('internal'+typ)
        EX.construct_measurable(id_internal[typ],partial(get_internal,typ=typ),[INIT],depth=0)

    #Construct footprint-type estimate of target position
    id_targ={}
    def look_up_target(state,typ):
        target_fp=np.ones(ENV_LENGTH)
        for ind,val in enumerate(state[id_internal[typ]][0]['target']):
            target_fp=target_fp*fp(ind) if val else target_fp
        return target_fp.tolist()
    #- construct target estimate measurable for each observer
    INIT=np.zeros(ENV_LENGTH).tolist()
    for typ in ORDERED_TYPES:
        id_targ[typ]=EX.register('targ'+typ)
        EX.construct_measurable(id_targ[typ],partial(look_up_target,typ=typ),[INIT],depth=0)

    # distance to target
    # - $id_distM$ has already been registerd
    def dist_to_target(state):
        return dist(state[id_pos][0], TARGET)
    INIT = dist(START, TARGET)
    EX.construct_measurable(id_dist, dist_to_target, [INIT, INIT])

    #
    ### MOTIVATIONS
    #

    #construct the motivational signal for OBS:
    def rescaling(state,typ):
        return RESCALING[typ](state[id_dist][0])
    for typ in ORDERED_TYPES:
        INIT = RESCALING[typ](dist(START,TARGET))
        EX.construct_measurable(id_sig[typ],partial(rescaling,typ=typ),[INIT, INIT])

    #
    ### AUXILIARY COMPUTATIONS
    #
    
    # value of each position in the environment, needed as np.array, for each run and type
    VALUES={typ:[RESCALING[typ](dist(pos,TARGET)) for pos in xrange(ENV_LENGTH)] for typ in ORDERED_TYPES}
    vm=lambda typ: np.array(VALUES[typ])
    # extreme (=target) value of signal for each run and type
    v_extreme=lambda typ: vm(typ).min() if typ=='_Q' else vm(typ).max()
    #Construct value-based ground truth target footprint
    GROUND_TARG={}
    for typ in ORDERED_TYPES:
        GROUND_TARG[typ]=[int(abs(val-v_extreme(typ))<pow(10,-12)) for val in vm(typ)]

    #Ground truth implication matrices
    #
    #- check for inclusions among footprint vectors:
    std_imp_check=lambda x,y: all(fp(x)<=fp(y))

    #- check for ground truth thresholded inclusions (footprint-based)
    def lookup_val(x,y,typ):
        #obtain footprints
        FPx=fp(x)
        FPy=fp(y)
        #compute the expected ground weights
        if typ=='_Q':
            return -1 if not sum(FPx*FPy) else np.extract(FPx*FPy,vm(typ)).min()
        else:
            #return (sum(FPx*FPy*vm(typ))+0.)/ENV_LENGTH
            return (sum(FPx*FPy*vm(typ))+0.)/sum(vm(typ)+0.)

    #- check for [ground truth] implications (index based)
    def imp_check(x,y,typ):
        XY=lookup_val(x,y,typ)
        X_Y=lookup_val(compi(x),y,typ)
        X_Y_=lookup_val(compi(x),compi(y),typ)
        XY_=lookup_val(x,compi(y),typ)
        if typ=='_Q': #qualitative implication (zero threshold)
            return int(qless(qmax(XY,X_Y_),XY_))
        else: #real-valued (statistical) implication
            if y==x:
                #same sensor yields a True entry
                return 1
            elif y==compi(x):
                #complementary sensors yield a False entry
                return 0
            else:
                #otherwise, compare weights:
                #return int(XY_<min(Threshold*sum(vm(typ)),XY,X_Y_,X_Y) or (XY_==0. and X_Y==0.))
                return int(XY_<min(Threshold,XY,X_Y_,X_Y) or (XY_==0. and X_Y==0.))

    #- construct the ground truth matrices (flattened) for each run and type
    GROUND_WEIGHTS={typ:[] for typ in ORDERED_TYPES}
    GROUND_RAW_IMPS={typ:[] for typ in ORDERED_TYPES}
    GROUND_ABS_IMPS=[]
    for yind in xrange(2*NSENSORS):
        for xind in xrange(yind+1):
            #weights and other expected implications, by type:
            for typ in ORDERED_TYPES:
                lookup=partial(lookup_val,typ=typ)
                check=partial(imp_check,typ=typ)
                # Weight matrix computed from known values of states
                GROUND_WEIGHTS[typ].append(lookup(yind,xind))
                # PCR matrix computed from known values of states; note the transpose!!
                GROUND_RAW_IMPS[typ].append(check(yind,xind))
    #absolute ground-truth implications:
    for yind in xrange(2*NSENSORS):
        for xind in xrange(yind+1):
            GROUND_ABS_IMPS.append(std_imp_check(yind,xind))
        if yind%2==0:
            GROUND_ABS_IMPS.append(0)


    #- import weight matrices from core
    id_weights={typ:EX.register('wgt'+typ) for typ in ORDERED_TYPES}
    def weights(state,typ):
        #print typ+' weights:'
        #print convert_weights(state[id_internal[typ]][0]['weights'])
        #print '\n'
        return flatten(state[id_internal[typ]][0]['weights'])

    for typ in ORDERED_TYPES:    
        INIT = (-np.ones(NSENSORS*(2*NSENSORS+1)) if typ=='_Q' else np.zeros(NSENSORS*(2*NSENSORS+1))).tolist()
        EX.construct_measurable(id_weights[typ],partial(weights,typ=typ),[INIT],depth=0)

    #- import raw implications from core
    id_raw_imps={typ:EX.register('raw'+typ) for typ in ORDERED_TYPES}
    def raw_imps(state,typ):
        #print typ+' implications:'
        #print convert_raw_implications(state[id_internal[typ]][0]['dirs'])
        #print '\n'
        return flatten(state[id_internal[typ]][0]['dirs'])

    INIT=[(1 if x==y else 0) for y in xrange(2*NSENSORS) for x in xrange(y+1)] #initialize to (lower triangle of) identity matrix
    for typ in ORDERED_TYPES:
        EX.construct_measurable(id_raw_imps[typ],partial(raw_imps,typ=typ),[INIT],depth=0)
     
    #- import full implications from core
    id_full_imps={typ:EX.register('full'+typ) for typ in ORDERED_TYPES}
    def full_imps(state,typ):
        return flatten(state[id_internal[typ]][0]['npdirs'])

    xr=lambda y: y+2 if y%2==0 else y+1
    INIT=[(1 if x==y else 0) for y in xrange(2*NSENSORS) for x in xrange(xr(y))] #initialize to (lower 2*2 triangle of) identity matrix
    for typ in ORDERED_TYPES:
        EX.construct_measurable(id_full_imps[typ],partial(full_imps,typ=typ),[INIT],depth=0)

    #- ell_infinity distance of current weights to ground truth
    id_wdiffs={typ:EX.register('wdiff'+typ) for typ in ORDERED_TYPES}
    def wdiffs(state,typ):
        #print state[id_weights[typ]][0]
        return max(ldiff(state[id_weights[typ]][0],GROUND_WEIGHTS[typ]))

    for typ in ORDERED_TYPES:
        #print '\n\n'
        #print EX.this_state(id_weights[typ])
        #print len(EX.this_state(id_weights[typ]))
        #print '\n\n'
        #print GROUND_WEIGHTS[typ]
        #print len(GROUND_WEIGHTS[typ])
        #print '\n\n'
        INIT=max(ldiff(EX.this_state(id_weights[typ]),GROUND_WEIGHTS[typ]))
        EX.construct_measurable(id_wdiffs[typ],partial(wdiffs,typ=typ),[INIT],depth=0)
 
    #- ell_one distance of learned PCR to expected ground truth PCR
    id_rawdiffs={typ:EX.register('rdiff'+typ) for typ in ORDERED_TYPES}
    def rawdiffs(state,typ):
        return sum(ldiff(state[id_raw_imps[typ]][0],GROUND_RAW_IMPS[typ]))
    for typ in ORDERED_TYPES:
        #print '\n\n'
        #print EX.this_state(id_raw_imps[typ])
        #print GROUND_RAW_IMPS[typ]
        #print '\n\n'
        INIT=sum(ldiff(EX.this_state(id_raw_imps[typ]),GROUND_RAW_IMPS[typ]))
        EX.construct_measurable(id_rawdiffs[typ],partial(rawdiffs,typ=typ),[INIT],depth=0)

    #- ell_one distance of FULL implications to true ABSOLUTE implications
    id_fulldiffs={typ:EX.register('fdiff'+typ) for typ in ORDERED_TYPES}
    def fulldiffs(state,typ):
        return sum(ldiff(state[id_full_imps[typ]][0],GROUND_ABS_IMPS))
    for typ in ORDERED_TYPES:
        INIT=sum(ldiff(EX.this_state(id_full_imps[typ]),GROUND_ABS_IMPS))
        EX.construct_measurable(id_fulldiffs[typ],partial(fulldiffs,typ=typ),[INIT],depth=0)
    # -------------------------------------init--------------------------------------------

    for agent_name in EX._AGENTS:
        EX._AGENTS[agent_name].init()

    QUERY_IDS={agent_id:{} for agent_id in EX._AGENTS}
    for agent_id in EX._AGENTS:
        for token in ['plus', 'minus']:

            # INTRODUCE DELAYED GPS SENSORS:
            #delay_sigs = [EX._AGENTS[agent_id].generate_signal(['x' + str(ind)], token) for ind in xrange(NSENSORS)]
            #EX._AGENTS[agent_id].delay(delay_sigs, token)

            # MAKE A LIST OF ALL SENSOR LABELS FOR EACH AGENT
            QUERY_IDS[agent_id][token]=EX._AGENTS[agent_id].make_sensor_labels(token)


    # START RECORDING
    default_instruction=[cid_obs[typ] for typ in ORDERED_TYPES]
    EX.update_state(default_instruction)
    recorder=experiment_output(EX,run_params)
    recorder.addendum('query_ids',QUERY_IDS)
    recorder.addendum('threshold',Threshold)
    recorder.addendum('Nsensors',NSENSORS)
    recorder.addendum('env_length',ENV_LENGTH)
    recorder.addendum('ground_targ',GROUND_TARG)
    recorder.record()

    # -------------------------------------RUN--------------------------------------------

    ## Main Loop:
    while EX.this_state(id_count) <= TOTAL_CYCLES:
        # update the state
        EX.update_state(default_instruction)
        recorder.record()

    # Wrap up and collect garbage
    recorder.close()
    EX.remove_experiment()



