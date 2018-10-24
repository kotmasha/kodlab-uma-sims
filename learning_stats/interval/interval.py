import numpy as np
from numpy.random import randint as rnd
from collections import deque
#import curses
import time
from UMA.som2_noEP import *
import sys
import os
import json
from client.UMARest import *

def start_experiment(run_params):
    # System parameters
    test_name=run_params['test_name']
    host = run_params['host']
    port = run_params['port']

    # initialize a new experiment
    EX = Experiment(test_name, UMARestService(host, port))
    id_dec = 'decision'
    id_count= 'counter'

    # Recording options:
    record_mids=run_params['mids_to_record'] #[id_count,id_dist,id_sig]
    record_global=run_params['ex_dataQ'] #True
    record_agents=run_params['agent_dataQ'] #True
    #recorder will be initialized later, at the end of the initialization phase,
    #to enable collection of all available data tags

    # Decision cycles:
    TOTAL_CYCLES = run_params['total_cycles']
    # Parameters and definitions
    AutoTarg=True #bool(run_params['AutoTarg'])
    SnapType=run_params['SnapType']
    Variation=run_params['Variation'] #snapshot type variation to be used ('uniform' or 'value-based')
    Mode=run_params['Mode'] #mode by which Sniffy moves around: 'teleport'/'walk'/'lazy'

    # Parameters
    X_BOUND = run_params['env_length']  # no. of edges in discrete interval = no. of GPS sensors
    try:
        Discount=float(run_params['discount']) #discount coefficient, if any
    except KeyError:
        Discount=0.875
    try:
        Threshold=float(run_params['threshold']) #implication threshold, defaulting to the square of the probability of a single position.
    except KeyError:
        if SnapType=='qualitative':
            Threshold=0.
        else:
            Threshold=1./pow(X_BOUND,2)

    # Environment description
    def in_bounds(pos):
        return (pos >= 0 and pos <= X_BOUND)
    def dist(p, q):
        return abs(p - q) #distance between two points in environment

    # agent parameters according to .yml file

    MOTION_PARAMS = {
        'type': SnapType,
        'AutoTarg': AutoTarg,
        'discount': Discount,
        'threshold': Threshold,
    }

    #Register "observer" agent:
    #  This agent remains inactive throghout the experiment, in order to record 
    #  all the UNCONDITIONAL implications among the initial sensors (in its 'minus'
    #  snapshot).
    #  For this purpose, each sensor's FOOTPRINT in the state space (positions in
    #  the interval) is recorded, so that implications may be calaculated according
    #  to the inclusions among footprints.
    id_obs,cid_obs=EX.register_sensor('obs')

    # register motivation for motion agents
    # - this one is NOT dependent on agents except through the position, so
    #   it carries the default "decdep=False" tag.
    id_dist = EX.register('dist')
    # Value signals for different setups determined as a *function* of distance to target
    id_sig = EX.register('sig')
    # ...which function? THIS function (see $rescaling$ below):
    RESCALING={
        'qualitative':{
            'uniform': lambda r: r,
            'value-based': lambda r: r,
            },
        'discounted':{
            'uniform': lambda r: 1,
            'value-based': lambda r: pow(1.-Discount,r-X_BOUND),
            },
        'empirical':{
            'uniform': lambda r: 1,
            'value-based': lambda r: X_BOUND+1-r,
            },
        }
    rescaling=RESCALING[SnapType][Variation]


    # OBSERVER agent simply collects implications among the assigned sensors, always inactive
    def action_OBS(state):
        return False
    OBS = EX.construct_agent(id_obs,id_sig,action_OBS,MOTION_PARAMS)
    OBSCD = UMAClientData(EX._EXPERIMENT_ID,id_obs,'minus',EX._service)

    #
    ### "mapping" system
    #

    ## introduce agent's position

    # select starting position
    START = rnd(X_BOUND + 1)

    # effect of motion on position
    id_pos = EX.register('pos')

    def random_walk(state):
        diff = 2*rnd(2)-1
        newpos = state[id_pos][0] + diff
        if in_bounds(newpos):
            return newpos
        else:
            return state[id_pos][0]

    def lazy_random_walk(state):
        diff = rnd(3)-1
        newpos = state[id_pos][0] + diff
        if in_bounds(newpos):
            return newpos
        else:
            return state[id_pos][0]

    def teleport(state):
        return rnd(X_BOUND+1)

    def back_and_forth(state):
        last_diff=state[id_pos][0]-state[id_pos][1]
        thispos=state[id_pos][0]
        if last_diff!=0:
            newpos=thispos+last_diff
            if in_bounds(newpos):
                return newpos
            else:
                return thispos-last_diff
        else:
            if thispos<X_BOUND:
                return thispos+1
            else:
                return thispos-1

    motion={'simple':back_and_forth,'walk':random_walk,'lazy':lazy_random_walk,'teleport':teleport}
    EX.construct_measurable(id_pos,motion[Mode],[START,START])

    # generate target position
    TARGET = START
    while dist(TARGET, START)==0:
        TARGET = rnd(X_BOUND+1)

    # set up position sensors
    def xsensor(m):  # along x-axis
        return lambda state: state[id_pos][0] < m + 1
    #Construct initial sensors and record their semantics
    FOOTPRINTS=[]
    all_comp=lambda x: [1-t for t in x]
    for ind in xrange(X_BOUND):
        tmp_name = 'x' + str(ind)
        tmp_footprint=[(1 if pos<ind+1 else 0) for pos in xrange(X_BOUND+1)]
        id_tmp, id_tmpc = EX.register_sensor(tmp_name)  # registers the sensor pairs
        EX.construct_sensor(id_tmp, xsensor(ind))  # constructs the measurables associated with the sensor
        OBS.add_sensor(id_tmp)
        FOOTPRINTS.append(tmp_footprint)
        FOOTPRINTS.append(all_comp(tmp_footprint))

    #Construct footprint-type estimate of target position
    id_targ_footprint=EX.register('targ_foot')
    def target_footprint(state):
        targ=OBSCD.getTarget()
        prints=np.array([fp for index,fp in zip(targ,FOOTPRINTS) if index])
        return np.prod(prints,axis=0).tolist()
    INIT=np.zeros(X_BOUND+1).tolist()
    EX.construct_measurable(id_targ_footprint,target_footprint,[INIT],depth=0)    

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
    def sig(state):
        return rescaling(state[id_dist][0])
    INIT = rescaling(dist(START,TARGET))
    EX.construct_measurable(id_sig, sig, [INIT, INIT])

    #record the value at each position
    VALUES=[rescaling(dist(ind,TARGET)) for ind in xrange(X_BOUND+1)]


    # -------------------------------------init--------------------------------------------

    for agent_name in EX._AGENTS:
        EX._AGENTS[agent_name].init()

    #client data objects for the experiment
    UMACD={}
    for agent_id in EX._AGENTS:
        for token in ['plus','minus']:
            UMACD[(agent_id,token)]=UMAClientData(EX._EXPERIMENT_ID,agent_id,token,EX._service)

    QUERY_IDS={agent_id:{} for agent_id in EX._AGENTS}
    for agent_id in EX._AGENTS:
        for token in ['plus', 'minus']:

            # INTRODUCE DELAYED GPS SENSORS:
            #delay_sigs = [EX._AGENTS[agent_id].generate_signal(['x' + str(ind)], token) for ind in xrange(X_BOUND)]
            #EX._AGENTS[agent_id].delay(delay_sigs, token)

            # MAKE A LIST OF ALL SENSOR LABELS FOR EACH AGENT
            QUERY_IDS[agent_id][token]=EX._AGENTS[agent_id].make_sensor_labels(token)

    # START RECORDING
    EX.update_state([cid_obs])
    recorder=experiment_output(EX,run_params)
    recorder.addendum('footprints',FOOTPRINTS)
    recorder.addendum('query_ids',QUERY_IDS)
    recorder.addendum('values',VALUES)
    recorder.addendum('threshold',Threshold)

    # -------------------------------------RUN--------------------------------------------

    ## Main Loop:
    while EX.this_state(id_count) <= TOTAL_CYCLES:
        # update the state
        instruction=[
            cid_obs,    #OBS should always be inactive
            ]
        EX.update_state(instruction)
        recorder.record()

    # Wrap up and collect garbage
    recorder.close()
    EX.remove_experiment()



