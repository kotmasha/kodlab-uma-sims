import numpy as np
from numpy.random import randint as rnd
from functools import partial
from collections import deque
#import curses
import time
from UMA.som2_noEP import *
import sys
import os
import cPickle
from client.UMARest import *

def start_experiment(run_params):
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
    try:
        Discount=float(run_params['discount'])
    except KeyError:
        Discount=None

    # Environment
    X_BOUND = run_params['env_length']  # length

    def in_bounds(pos):
        return (pos >= 0 and pos <= X_BOUND)

    # distance function
    def dist(p, q):
        return abs(p - q)

    # "Qualitative" agent parameters

    MOTION_PARAMS = {
        'type': SnapType,
        'AutoTarg': True,
        'discount': Discount,
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
    id_at_targ, cid_at_targ = EX.register_sensor('atT')
    id_dist = EX.register('dist')
    id_sig = EX.register('sig')
    id_nav, cid_nav = EX.register_sensor('nav')

    # register arbiter variable whose purpose is provide a hard-wired response to a conflict
    # between agents 'lt' and 'rt'.
    id_arbiter = EX.register('ar')

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
            return not(state[id_arbiter][0])
        else:
            return lt_decided

    LT = EX.construct_agent(id_lt, id_sig, action_LT, MOTION_PARAMS)

    #
    ### "mapping" system
    #

    ## introduce agent's position

    # select starting position, target position
    START = rnd(X_BOUND + 1)
    TARGET = rnd(X_BOUND + 1)

    # effect of motion on position
    id_pos = EX.register('pos')

    def motion(state):
        triggers = {id_rt: 1, id_lt: -1}
        diff = 0
        for t in triggers:
            diff += triggers[t] * int(state[t][0])
        newpos = state[id_pos][0] + diff
        if in_bounds(newpos):
            return newpos
        else:
            return state[id_pos][0]

    EX.construct_measurable(id_pos, motion, [START, START])

    # set up position sensors
    def xsensor(state,m):  # along x-axis
        return state[id_pos][0] < m + 1

    id_x={}
    cid_x={}
    for ind in xrange(X_BOUND):
        id_x[ind], cid_x[ind] = EX.register_sensor('x' + str(ind))  # registers the sensor pairs
        INIT= (START < ind+1)
        EX.construct_sensor(id_x[ind], partial(xsensor,m=ind),[INIT,INIT])  # constructs the measurables associated with the sensor
        RT.add_sensor(id_x[ind])
        LT.add_sensor(id_x[ind])

    # distance to target
    # - $id_distM$ has already been registerd
    def dist_to_target(state):
        return dist(state[id_pos][0], TARGET)

    INIT = dist(START, TARGET)
    EX.construct_measurable(id_dist, dist_to_target, [INIT, INIT])

    ## value signal for agents LT and RT
    # signal scales with distance to target
    if SnapType=='qualitative':
        rescaling = lambda r: r
    else:
        #SnapType is default
        #rescaling = lambda r: 1.-(pow(1.-Discount,-X_BOUND))*np.log((1.+r)/(1.+X_BOUND))
        rescaling = lambda r: pow(1.-Discount,r-X_BOUND)

    def sig(state):
        return rescaling(state[id_dist][0])
        # initial value for signal:

    INIT = rescaling(dist(START, TARGET))
    # construct the motivational signal
    EX.construct_measurable(id_sig, sig, [INIT, INIT])

    # -------------------------------------init--------------------------------------------

    for agent_name in EX._AGENTS:
        EX._AGENTS[agent_name].init()
    #!!! state update required to fill all data structures *before* delayed sensors are introduced
    EX.update_state([cid_lt,cid_rt])

    #client data objects for the experiment
    #UMACD={}
    #for agent_id in EX._AGENTS:
    #    for token in ['plus','minus']:
    #        UMACD[(agent_id,token)]=UMAClientData(EX._EXPERIMENT_ID,agent_id,token,EX._service)

    # INTRODUCE DELAYED GPS SENSORS:
    for agent in [RT, LT]:
        for token in ['plus', 'minus']:
            delay_sigs = []
            # add delayed position sensors
            delay_sigs.extend([agent.generate_signal([sid], token) for sid in id_x.values()])
            agent.delay(delay_sigs, token)


    # -------------------------------------RUN--------------------------------------------
    recorder=experiment_output(EX,run_params)

    # incative step for the agents to "take in" the starting state
    EX.update_state([cid_lt,cid_rt])
    recorder.record()

    ## Random walk period
    while EX.this_state(id_count) < BURN_IN_CYCLES:
        # update the state
        instruction=[(id_lt if rnd(2) else cid_lt),(id_rt if rnd(2) else cid_rt)]
        #instruction = [(id_lt if (EX.this_state(id_count) / X_BOUND) % 2 == 0 else cid_lt),
                       #(id_rt if (EX.this_state(id_count) / X_BOUND) % 2 == 1 else cid_rt)]
        EX.update_state(instruction)
        recorder.record()

    ## Main loop
    while EX.this_state(id_count) < TOTAL_CYCLES:
        # make decisions, update the state
        EX.update_state()
        recorder.record()

    recorder.close()
    EX.remove_experiment()
