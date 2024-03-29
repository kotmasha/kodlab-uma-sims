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
import cPickle
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
    SEED=seed()
    # System parameters
    test_name=run_params['test_name']
    host = run_params['host']
    port = run_params['port']

    # Recording options:
    record_mids=run_params['mids_to_record'] #[id_count,id_dist,id_sig]
    record_global=run_params['ex_dataQ'] #True
    record_agents=run_params['agent_dataQ'] #True

    # Decision cycles:
    TOTAL_CYCLES = run_params['total_cycles']
    BURN_IN_CYCLES = run_params['burn_in_cycles']
    # Parameters and definitions
    SnapType=run_params['SnapType']
    X_BOUND = run_params['circumference']  # length
    BEACON_WIDTH = run_params['beacon_width'] # width of place field centered at each position
    ENV_LENGTH=X_BOUND
    Nsensors=2*X_BOUND

    # distance function
    def dist(p, q):
        return min(abs(p-q),X_BOUND-abs(p-q))

    # agent parameters
    MOTION_PARAMS = {
        'type': SnapType,   #'qualitative',
        'AutoTarg': True,
        'threshold': 0,
    }

    # initialize a new experiment
    EX = Experiment(test_name, UMARestService(host, port))
    id_dec = 'decision'
    id_count= 'counter'

    # register basic motion agents;
    # - $True$ tag means they will be marked as dependent (on other agents)
    id_rt, cid_rt = EX.register_sensor('rt')
    id_lt, cid_lt = EX.register_sensor('lt')

    # register motivation for motion agents
    # - this one is NOT dependent on agents except through the position, so
    #   it carries the default False tag.
    #id_at_targ, cid_at_targ = EX.register_sensor('atT')
    id_dist = EX.register('dist')
    id_sig = EX.register('sig')
    id_nav, cid_nav = EX.register_sensor('nav')

    # register arbiter variable whose purpose is provide a hard-wired response to a conflict
    # between agents 'lt' and 'rt'.
    id_arbiter = EX.register('arb')

    #
    ### Arbitration
    #

    # arbitration state
    def arbiter(state):
        return bool(rnd(2))
    EX.construct_measurable(id_arbiter, arbiter, [bool(rnd(2))], 0, decdep=True)

    # intention sensors
    id_toRT, cid_toRT = EX.register_sensor('toR')

    def intention_RT(state):
        return id_rt in state[id_dec][0]

    EX.construct_sensor(id_toRT, intention_RT, decdep=False)

    id_toLT, cid_toLT = EX.register_sensor('toL')

    def intention_LT(state):
        return id_lt in state[id_dec][0]

    EX.construct_sensor(id_toLT, intention_LT, decdep=False)

    # failure mode for action $lt^rt$
    id_toF, cid_toF = EX.register_sensor('toF')

    def about_to_enter_failure_mode(state):
        return state[id_toLT][0] and state[id_toRT][0]

    EX.construct_sensor(id_toF, about_to_enter_failure_mode, decdep=False)

    # add basic motion agents with arbitration
    def action_RT(state):
        rt_decided = (id_rt in state[id_dec][0])
        if state[id_toF][0]:
            # return not(rt_decided) if state[id_arbiter][0] else rt_decided
            return state[id_arbiter][0]
        else:
            return rt_decided

    RT = EX.construct_agent(id_rt, id_sig, action_RT, MOTION_PARAMS)

    def action_LT(state):
        lt_decided = (id_lt in state[id_dec][0])
        if state[id_toF][0]:
            # return lt_decided if state[id_arbiter][0] else not(lt_decided)
            return not (state[id_arbiter][0])
        else:
            return lt_decided

    LT = EX.construct_agent(id_lt, id_sig, action_LT, MOTION_PARAMS)

    #
    ### Access to agents' internal state
    #

    ACCESS={}
    for agent_id in EX._AGENTS:
        for token in ['plus','minus']:
            ACCESS[(agent_id,token)]=UMAClientData(EX._EXPERIMENT_ID,agent_id,token,EX._service)


    #
    ### "mapping" system
    #

    ## introduce agent's position

    # select starting position
    START = rnd(X_BOUND)

    # target position may be fixed due to rotational symmetry
    TARGET = 0

    # effect of motion on position
    id_pos = EX.register('pos')
    def motion(state):
        triggers = {id_rt: 1, id_lt: -1}
        diff = 0
        for t in triggers:
            diff += triggers[t] * int(state[t][0])
        newpos = (state[id_pos][0] + diff) % X_BOUND
        return newpos
 
    EX.construct_measurable(id_pos, motion, [START, START])

    # set up position sensors
    def xsensor(state,m,w):  # along x-axis
        return dist(state[id_pos][0],m)<=w

    id_x={}
    cid_x={}
    FOOTPRINTS=[]
    all_comp=lambda x: [1-t for t in x]
    for ind in xrange(X_BOUND):
        id_x[ind], cid_x[ind] = EX.register_sensor('x' + str(ind))
        INIT = (dist(START,ind) <= BEACON_WIDTH)
        EX.construct_sensor(id_x[ind], partial(xsensor,m=ind,w=BEACON_WIDTH),[INIT,INIT])
        RT.add_sensor(id_x[ind])
        LT.add_sensor(id_x[ind])
        # record the sensor footprint
        tmp_footprint=[(1 if dist(pos,ind)<=BEACON_WIDTH else 0) for pos in xrange(ENV_LENGTH)]
        FOOTPRINTS.append(tmp_footprint)
        FOOTPRINTS.append(all_comp(tmp_footprint))

    # sensor footprints for this run
    fp=lambda sensor_ind: np.array(FOOTPRINTS[sensor_ind])

    #get internal data for each agent
    id_acc={}
    def get_internal(state,snap_ind):
        return ACCESS[snap_ind].get_all()

    INIT={}
    for key in ACCESS.keys():
        aid,tok=key
        id_acc[key]=EX.register('acc_'+aid+('+' if tok=='plus' else '-'))
        EX.construct_measurable(id_acc[key],partial(get_internal,snap_ind=key),[INIT],depth=0)

    #Construct footprint-type estimate of target position
    id_targ={}
    def look_up_target(state,snap_ind):
        target_fp=np.ones(ENV_LENGTH)
        for ind in xrange(Nsensors):
            target_fp=target_fp*fp(ind) if state[id_acc[snap_ind]][0]['target'][ind] else target_fp
        return target_fp.tolist()
    #- construct target estimate measurable for each observer
    INIT=np.zeros(ENV_LENGTH).tolist()
    for key in ACCESS.keys():
        aid,tok=key
        id_targ[key]=EX.register('targ_'+aid+('+' if tok=='plus' else '-'))
        EX.construct_measurable(id_targ[key],partial(look_up_target,snap_ind=key),[INIT],depth=0)

    #Construct footprint-type estimate of target position
    id_pred={}
    def look_up_prediction(state,snap_ind):
        pred_fp=np.ones(ENV_LENGTH)
        for ind in xrange(Nsensors):
            pred_fp=pred_fp*fp(ind) if state[id_acc[snap_ind]][0]['prediction'][ind] else pred_fp
        return pred_fp.tolist()
    #- construct target estimate measurable for each observer
    INIT=np.zeros(ENV_LENGTH).tolist()
    for key in ACCESS.keys():
        aid,tok=key
        id_pred[key]=EX.register('pred_'+aid+('+' if tok=='plus' else '-'))
        EX.construct_measurable(id_pred[key],partial(look_up_prediction,snap_ind=key),[INIT],depth=0)

    id_div={}
    def compute_divergence(state,snap_ind):
        tmp_targ=Signal(state[id_targ[snap_ind]][0])
        tmp_pred=Signal(state[id_pred[snap_ind]][0])
        return (tmp_targ.subtract(tmp_pred)).weight()
    INIT=0
    for key in ACCESS.keys():
        aid,tok=key
        id_div[key]=EX.register('div_'+aid+('+' if tok=='plus' else '-'))
        EX.construct_measurable(id_div[key],partial(compute_divergence,snap_ind=key),[INIT],depth=0)


    # distance to target
    # - $id_dist$ has already been registered
    def dist_to_target(state):
        return dist(state[id_pos][0], TARGET)
    INIT = dist(START, TARGET)
    EX.construct_measurable(id_dist, dist_to_target, [INIT, INIT])

    #
    ### signal scales with distance to target
    #
    
    #rescaling = lambda r: r
    rescaling = lambda r: 0 if r==0 else 1
    
    def sig(state):
        return rescaling(state[id_dist][0])
    SIG_RANGE=xrange(X_BOUND/2)
    # initial value for signal:
    SIG_INIT = rescaling(dist(START, TARGET))
    # construct the motivational signal
    EX.construct_measurable(id_sig, sig, [SIG_INIT,SIG_INIT])

    #
    ### signal depends on immediate success
    #
    #def sig(state):
    #    return 0 if state[id_dist][0]==0 else 2+(state[id_dist][0]-state[id_dist][1])
    #SIG_RANGE=xrange(4)
    ## initial value for signal:
    #SIG_INIT = 0 if START==TARGET else 2
    ## construct the motivational signal
    #EX.construct_measurable(id_sig, sig, [SIG_INIT,SIG_INIT])

    #
    ### Value sensors for motivational signal
    #

    #id_val={}
    #cid_val={}
    #def val_sensor(state,value):
    #    return state[id_sig][0]<=value
    #for val in SIG_RANGE:
    #    id_val[val],cid_val[val]=EX.register_sensor('v'+str(val))
    #    INIT = (SIG_INIT <= val)
    #    EX.construct_sensor(id_val[val],partial(val_sensor,value=val),[INIT,INIT])
    #    RT.add_sensor(id_val[val])
    #    LT.add_sensor(id_val[val])

    #def nav(state):
    #    return state[id_dist][0] < state[id_dist][1] or state[id_dist][0] == 0
    #EX.construct_sensor(id_nav,nav,[False,False])
    #RT.add_sensor(id_nav)
    #LT.add_sensor(id_nav)
    #
    #LT.add_sensor(id_lt)
    #LT.add_sensor(id_rt)
    #RT.add_sensor(id_lt)
    #RT.add_sensor(id_rt)

    # -------------------------------------init--------------------------------------------

    for agent_id in EX._AGENTS:
        EX._AGENTS[agent_id].init()
    #!!! state update required to fill all data structures *before* delayed sensors are introduced
    EX.update_state([cid_lt,cid_rt])

    # INTRODUCE DELAYED BEACON SENSORS:
    for agent in [RT, LT]:
        for token in ['plus', 'minus']:
            delay_sigs = []
            #delay the beacon sensors:
            delay_sigs.extend([agent.generate_signal([sid],token) for sid in id_x.values()])
            #delay the value sensors:
            #delay_sigs.extend([agent.generate_signal([sid],token) for sid in id_val.values()])
            #construct the delayed sensor in each snapshot:
            agent.delay(delay_sigs, token)

    # -------------------------------------RUN--------------------------------------------
    recorder=experiment_output(EX,run_params)

    recorder.addendum('seed',SEED)
    recorder.addendum('Nsensors',Nsensors)
    recorder.addendum('env_length',ENV_LENGTH)

    # incative step for the agents to "take in" the starting state
    EX.update_state([cid_lt,cid_rt])
    recorder.record()

    ## Random walk period
    while EX.this_state(id_count)-2 <= BURN_IN_CYCLES:
        # update the state
        instruction=[(id_lt if rnd(2) else cid_lt),(id_rt if rnd(2) else cid_rt)]
        #instruction = [id_lt,cid_rt] if ((EX.this_state(id_count)-2) / X_BOUND) % 2 == 0 else [cid_lt,id_rt]
        #instruction = [id_rt,cid_lt] if ((EX.this_state(id_count)-2) / (X_BOUND)) % 2== 0 else [cid_rt,id_lt]
        EX.update_state(instruction)
        recorder.record()

    ## Main loop
    while EX.this_state(id_count)-2 <= TOTAL_CYCLES:
        # make decisions, update the state
        EX.update_state()
        recorder.record()

    recorder.close()
    EX.remove_experiment()

