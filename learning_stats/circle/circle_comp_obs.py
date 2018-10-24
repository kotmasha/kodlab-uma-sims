import numpy as np
from numpy.random import randint as rnd
from collections import deque
from functools import partial
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

    try:
        Discount=float(run_params['discount']) #discount coefficient, if any
    except KeyError:
        Discount=0.75
    try:
        Threshold=float(run_params['threshold']) #implication threshold, defaulting to the square of the probability of a single position.
    except KeyError:
        Threshold=1./pow(X_BOUND,2)

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
        'discount': Discount,
        'AutoTarg': True,
        'threshold': Threshold,
    }

    qualitative_observer={
        'type': 'qualitative',
        'AutoTarg': True,
        #'threshold': 0,
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
        '_Q':  lambda r: r,
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
        TARGET = rnd(X_BOUND)

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

    #Construct footprint-type estimate of target position
    id_targ={}
    def look_up_target(state,typ):
        return OBSACCESS[typ].getTarget()
    
    #- construct target estimate measurable for each observer
    INIT=np.zeros(X_BOUND).tolist()
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
    
    #record the value at each position, for each type:
    VALUES={typ:[RESCALING[typ](dist(pos,TARGET)) for pos in xrange(X_BOUND)] for typ in ORDERED_TYPES}

    # -------------------------------------init--------------------------------------------

    for agent_name in EX._AGENTS:
        EX._AGENTS[agent_name].init()

    QUERY_IDS={agent_id:{} for agent_id in EX._AGENTS}
    for agent_id in EX._AGENTS:
        for token in ['plus', 'minus']:

            # INTRODUCE DELAYED GPS SENSORS:
            #delay_sigs = [EX._AGENTS[agent_id].generate_signal(['x' + str(ind)], token) for ind in xrange(X_BOUND)]
            #EX._AGENTS[agent_id].delay(delay_sigs, token)

            # MAKE A LIST OF ALL SENSOR LABELS FOR EACH AGENT
            QUERY_IDS[agent_id][token]=EX._AGENTS[agent_id].make_sensor_labels(token)

    # START RECORDING
    default_instruction=[cid_obs[typ] for typ in ORDERED_TYPES]
    EX.update_state(default_instruction)
    recorder=experiment_output(EX,run_params)
    recorder.addendum('footprints',FOOTPRINTS)
    recorder.addendum('query_ids',QUERY_IDS)
    recorder.addendum('values',VALUES)
    recorder.addendum('threshold',Threshold)

    # -------------------------------------RUN--------------------------------------------

    ## Main Loop:
    while EX.this_state(id_count) <= TOTAL_CYCLES:
        # update the state
        EX.update_state(default_instruction)
        recorder.record()

    # Wrap up and collect garbage
    recorder.close()
    EX.remove_experiment()



