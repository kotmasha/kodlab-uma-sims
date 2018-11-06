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
    Nsensors=X_BOUND

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
            ACCESS[agent_id,token]=UMAClientData(EX._EXPERIMENT_ID,agent_id,token,EX._service)

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
    #FOOTPRINTS=[]
    #all_comp=lambda x: [1-t for t in x]
    for ind in xrange(X_BOUND):
        id_x[ind], cid_x[ind] = EX.register_sensor('x' + str(ind))
        INIT = (dist(START,ind) <= BEACON_WIDTH)
        EX.construct_sensor(id_x[ind], partial(xsensor,m=ind,w=BEACON_WIDTH),[INIT,INIT])
        RT.add_sensor(id_x[ind])
        LT.add_sensor(id_x[ind])
        # record the sensor footprint
        #tmp_footprint=[(1 if dist(pos,ind)<=BEACON_WIDTH else 0) for pos in xrange(X_BOUND)]
        #FOOTPRINTS.append(tmp_footprint)
        #FOOTPRINTS.append(all_comp(tmp_footprint))

    # sensor footprints for this run
    #fp=lambda sensor_ind: np.array(FOOTPRINTS[sensor_ind])

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
    #SIG_RANGE=xrange(X_BOUND/2)
    SIG_RANGE=xrange(2)
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

