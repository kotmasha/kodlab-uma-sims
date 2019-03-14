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

    # environment
    ENV_TYPE = run_params['env_type']
    X_BOUND = run_params['env_sens']

    ENV_LENGTH={
        'interval': X_BOUND+1,
        'circle': X_BOUND,
        }
    env_length=ENV_LENGTH[ENV_TYPE]

    DIAM={
        'interval': X_BOUND,
        'circle': X_BOUND/2,
        }
    diam=DIAM[ENV_TYPE]

    Nsensors=2*X_BOUND

    # legal positions
    IN_BOUNDS={
        'interval': lambda pos: pos>=0 and pos<=X_BOUND,
        'circle': lambda pos: pos>=0 and pos<X_BOUND,
        }
    in_bounds=IN_BOUNDS[ENV_TYPE]
    
    # distance function
    DIST={
        'interval': lambda p,q: abs(p-q),
        'circle': lambda p,q: min(abs(p-q),X_BOUND-abs(p-q)),
        }
    dist=DIST[ENV_TYPE]
    
    # Parameters
    SnapType=run_params['SnapType']
    PeakType=run_params['PeakType'] # 'sharp'/'dull'
    
    RESCALING={
        ('qualitative','dull'): lambda r: 0 if r==0 else 1,
        ('qualitative','sharp'): lambda r: r,
        ('empirical','dull'): lambda r: 1+diam-r,
        ('empirical','sharp'): lambda r: pow(1+diam-r,4),
        ('discounted','dull'): lambda r: 1+diam-r,
        ('discounted','sharp'): lambda r: pow(1+diam-r,4),
        }
    rescaling=RESCALING[(SnapType,PeakType)]
    
    try:
        Discount=float(run_params['discount'])
    except KeyError:
        Discount=1.-1./(env_length+1.)

    try:
        Threshold=float(run_params['threshold'])
    except KeyError:
        if SnapType=='qualitative':
            Threshold=0
        else:
            Threshold=1./(2.*env_length)

    try:
        BEACON_WIDTH=int(run_params['beacon_width'])
    except KeyError:
        BEACON_WIDTH=0
            
    # agent parameters
    MOTION_PARAMS = {
        'type': SnapType,
        'AutoTarg': True,
        'threshold': Threshold,
        'q': Discount,
    }
    ARB_PARAMS = {
        'type': 'qualitative',
        'AutoTarg': True,
        'threshold': 0,
        'q': 0,
    }

    # initialize a new experiment
    EX = Experiment(test_name, UMARestService(host, port))
    id_dec = 'decision'
    id_count= 'counter'

    # register basic motion agents
    id_rt, cid_rt = EX.register_sensor('rt')
    id_lt, cid_lt = EX.register_sensor('lt')
    # register arbiter agent
    id_arb, cid_arb = EX.register_sensor('arb')
    
    # register motivation for motion agents
    # - this one is NOT dependent on agents except through the position, so
    #   it carries the default False tag.
    #id_at_targ, cid_at_targ = EX.register_sensor('atT')
    id_dist = EX.register('dist')
    id_sig = EX.register('sig')
    id_nav, cid_nav = EX.register_sensor('nav')

    # register randomized arbitration variable
    id_arb_coin = EX.register('cn')
    # register arbitration signal measurable
    id_sig_arb = EX.register('sig_arb')
    # register arbitration sensors
    id_toRT, cid_toRT = EX.register_sensor('toR') #intention of RT    
    id_toLT, cid_toLT = EX.register_sensor('toL') #intention of LT
    id_toF, cid_toF = EX.register_sensor('toF') #intention to enter failure state
    id_F, cid_F = EX.register_sensor('F') #failure state
    #
    ### Define basic motion agents
    #

    # RT definition
    def action_RT(state):
        rt_decided = (id_rt in state[id_dec][0])
        arb_decided= (id_arb in state[id_dec][0])
        if arb_decided: #If arbiter is active...
            if state[id_arb_coin][0]: #...and the coin-flip went "heads"...
                return not rt_decided #...flip the decision bit of RT.
            else: #If the coin-flip went "tails"...
                return rt_decided #...keep the original decision.
        else: #if the arbiter is inactive...
            return rt_decided #...keep the original decision.

    RT = EX.construct_agent(id_rt, id_sig, action_RT, MOTION_PARAMS)

    # LT definition
    def action_LT(state):
        lt_decided = (id_lt in state[id_dec][0])
        arb_decided= (id_arb in state[id_dec][0])        
        if arb_decided: #If arbiter is active...
            if state[id_arb_coin][0]: #...and the coin-flip went "heads"...
                return lt_decided #...keep the original decision.
            else: #If the coin-flip went "tails"...
                return not lt_decided #...flip the decision bit of LT.
        else: #if the arbiter is inactive...
            return lt_decided #...keep the original decision.

    LT = EX.construct_agent(id_lt, id_sig, action_LT, MOTION_PARAMS)

    # RT intention sensor for use of ARB
    def intention_RT(state):
        return id_rt in state[id_dec][0]
    EX.construct_sensor(id_toRT, intention_RT, decdep=True)

    # LT intention sensor for use of ARB
    def intention_LT(state):
        return id_lt in state[id_dec][0]
    EX.construct_sensor(id_toLT, intention_LT, decdep=True)

    #
    ### Arbitration
    #

    # arbitration state: a fair coin is tossed each update cycle
    def fair_coin(state):
        return bool(rnd(2))
    EX.construct_measurable(id_arb_coin, fair_coin, [bool(rnd(2))], 0, decdep=True)

    # failure mode for action $lt^rt$
    def about_to_enter_failure_mode(state):
        return state[id_toLT][0] and state[id_toRT][0]
    EX.construct_sensor(id_toF, about_to_enter_failure_mode, decdep=True)

    # definition of arbiter
    def action_ARB(state):
        return id_arb in state[id_dec][0]
    ARB = EX.construct_agent(id_arb, id_sig_arb, action_ARB, ARB_PARAMS) 

    # add sensors to arbiter
    ARB.add_sensor(id_toF)
    ARB.add_sensor(id_F)

    # Synchronization failure sensor (THIS is where we cheat!)
    #
    # Ideally, we should have a way of detecting synchronization
    # failures, rather than defining them from the get-go.
    #
    # The current arbitration scheme is overly simplistic: ARB merely needs
    # to learn the implications
    #     ARB  :  #id_toF < cid_F (failure prevented by ARB)
    #     ARB* :  #id_toF < id_F  (failure enforced by ARB*)
    # in order for it to be capable of enforcing the target.
    def failure_mode(state):
        return state[id_rt][0] and state[id_lt][0]
    INIT=False
    EX.construct_sensor(id_F, failure_mode, [INIT,INIT], decdep=False)
    
    #
    ### Access to agents' internal state
    #

    ACCESS={}
    for agent_id in [id_lt,id_rt]:
        for token in ['plus','minus']:
            ACCESS[(agent_id,token)]=UMAClientData(EX._EXPERIMENT_ID,agent_id,token,EX._service)


    #
    ### "mapping" system
    #

    ## introduce agent's position

    # select starting and target position
    #START = rnd(env_length)
    TARGET = rnd(env_length)
    START = TARGET

    # effect of motion on position
    id_pos = EX.register('pos')
    
    def motion_interval(state):
        triggers = {id_rt: 1, id_lt: -1}
        diff = 0
        for t in triggers:
            diff += triggers[t] * int(state[t][0])
        newpos = state[id_pos][0] + diff
        if in_bounds(newpos):
            return newpos
        else:
            return state[id_pos][0]

    def motion_circle(state):
        triggers = {id_rt: 1, id_lt: -1}
        diff = 0
        for t in triggers:
            diff += triggers[t] * int(state[t][0])
        newpos = (state[id_pos][0] + diff) % X_BOUND
        return newpos

    MOTION={
        'interval': motion_interval,
        'circle': motion_circle,
        }
    motion=MOTION[ENV_TYPE]

    EX.construct_measurable(id_pos, motion, [START, START])

    # position of target
    id_tpos=EX.register('tpos')
    def tpos_upd(state):
	return TARGET
    INIT=TARGET
    EX.construct_measurable(id_tpos, tpos_upd, [INIT,INIT])

    # set up position sensors
    def xsensor_interval(pos,par):
        return pos < par+1

    def xsensor_circle(pos,par):
        ind,width=par
        return dist(pos,ind)<=width

    XSENSOR={
        'interval': xsensor_interval,
        'circle': xsensor_circle,
        }
    xsensor=XSENSOR[ENV_TYPE]

    XPAR={
        'interval': lambda ind: ind,
        'circle': lambda ind: (ind,BEACON_WIDTH),
        }
    xpar=XPAR[ENV_TYPE]
    
    def xsensor_upd(state,par):
        return xsensor(state[id_pos][0],par)
    
    id_x={}
    cid_x={}
    FOOTPRINTS=[]
    all_comp=lambda x: [1-t for t in x]
    for ind in xrange(X_BOUND):
        id_x[ind], cid_x[ind] = EX.register_sensor('x' + str(ind))
        INIT=xsensor(START,xpar(ind))
        EX.construct_sensor(
            id_x[ind],
            partial(xsensor_upd,par=xpar(ind)),
            [INIT,INIT],
        )
        RT.add_sensor(id_x[ind])
        LT.add_sensor(id_x[ind])
        # record the sensor footprint
        tmp_footprint=[(1 if xsensor(pos,xpar(ind)) else 0) for pos in xrange(env_length)]
        FOOTPRINTS.append(tmp_footprint)
        FOOTPRINTS.append(all_comp(tmp_footprint))

    # sensor footprints for this run
    fp=lambda sensor_ind: np.array(FOOTPRINTS[sensor_ind])

    # get internal data for each agent
    id_acc={}
    def get_internal(state,snap_ind):
        return ACCESS[snap_ind].get_all()

    INIT={}
    for key in ACCESS.keys():
        aid,tok=key
        id_acc[key]=EX.register('acc_'+aid+('+' if tok=='plus' else '-'))
        EX.construct_measurable(id_acc[key],partial(get_internal,snap_ind=key),[INIT],depth=0)

    # construct footprint-type estimate of target position
    id_targ={}
    def look_up_target(state,snap_ind):
        target_fp=np.ones(env_length)
        for ind in xrange(Nsensors):
            target_fp=target_fp*fp(ind) if state[id_acc[snap_ind]][0]['target'][ind] else target_fp
        return target_fp.tolist()
    #- construct target estimate measurable for each observer
    INIT=np.zeros(env_length).tolist()
    for key in ACCESS.keys():
        aid,tok=key
        id_targ[key]=EX.register('targ_'+aid+('+' if tok=='plus' else '-'))
        EX.construct_measurable(id_targ[key],partial(look_up_target,snap_ind=key),[INIT],depth=0)

    # construct footprint-type estimate of predicted position
    id_pred={}
    def look_up_prediction(state,snap_ind):
        pred_fp=np.ones(env_length)
        for ind in xrange(Nsensors):
            pred_fp=pred_fp*fp(ind) if state[id_acc[snap_ind]][0]['prediction'][ind] else pred_fp
        return pred_fp.tolist()
    # construct predicted position measurable for each observer
    INIT=np.zeros(env_length).tolist()
    for key in ACCESS.keys():
        aid,tok=key
        id_pred[key]=EX.register('pred_'+aid+('+' if tok=='plus' else '-'))
        EX.construct_measurable(id_pred[key],partial(look_up_prediction,snap_ind=key),[INIT],depth=0)

    # construct divergence of prediction from target measurable        
    id_div={}
    def compute_divergence(state,snap_ind):
        tmp_targ=Signal(state[id_acc[snap_ind]][0]['target'])
        tmp_pred=Signal(state[id_acc[snap_ind]][0]['prediction'])
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
    ### value signal scales with distance to target
    #

    def sig(state):
        return rescaling(state[id_dist][0])
    # SIG_RANGE=diam
    # initial value for signal:
    SIG_INIT = rescaling(dist(START, TARGET))
    # construct the motivational signal
    EX.construct_measurable(id_sig, sig, [SIG_INIT,SIG_INIT])


    #
    ### Arbitration signal
    #

    def sig_arb(state):
        return 1 if (state[id_lt][0] and state[id_rt][0]) else 0
    SIG_INIT = 0
    EX.construct_measurable(id_sig_arb, sig_arb, [SIG_INIT,SIG_INIT])
    
    # -------------------------------------init--------------------------------------------

    for agent_id in EX._AGENTS:
        EX._AGENTS[agent_id].init()
    #!!! state update required to fill all data structures *before* delayed sensors are introduced
    EX.update_state([cid_lt,cid_rt,cid_arb])

    # INTRODUCE DELAYED BEACON SENSORS:
    for agent in [RT, LT, ARB]:
        for token in ['plus', 'minus']:
            delay_sigs = []
            #delay the beacon sensors:
            delay_sigs.extend([agent.generate_signal([sid],token) for sid in id_x.values()])
            #construct the delayed sensor in each snapshot:
            agent.delay(delay_sigs, token)

    # -------------------------------------RUN--------------------------------------------
    recorder=experiment_output(EX,run_params)
    #recorder.record()
    
    recorder.addendum('seed',SEED)
    recorder.addendum('Nsensors',Nsensors)
    recorder.addendum('env_length',env_length)
    recorder.addendum('diam',diam)

    ## Random walk period
    while EX.this_state(id_count) <= BURN_IN_CYCLES:
        # update the state
        instruction=[
            (id_lt if rnd(2) else cid_lt),
            (id_rt if rnd(2) else cid_rt),
            (id_arb if rnd(2) else cid_arb),
        ]
        EX.update_state(instruction)
        recorder.record()

    ## Main loop
    while EX.this_state(id_count) <= TOTAL_CYCLES:
        # make decisions, update the state
        EX.update_state()
        recorder.record()

    recorder.close()
    EX.remove_experiment()

