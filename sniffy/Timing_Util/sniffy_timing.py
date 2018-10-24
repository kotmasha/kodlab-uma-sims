import numpy as np
from numpy.random import randint as rnd
from collections import deque
#import time
from som2 import *
import sys
import cPickle


def start_experiment(pickler,size,runs,q):
    # wait (=0) / don't wait (=1) for spacebar to execute next cycle:
    NODELAY=1 #if delay_string=='nodelay' else 0
    # Number of decision cycles for burn-in period:
    BURN_IN=runs
    # experiment parameters and definitions
    X_BOUND=size #length
    def in_bounds(pos):
        return (pos>=0 and pos<=X_BOUND)

    # distance function
    def dist(pt1,pt2):
        return abs(pt1-pt2) 

    # agent discount parameters
    Q=1.-pow(2,-q)
    MOTION_PARAMS={
        'type':'default',
        'discount':Q,
        'AutoTarg':True,
        }
    
    # initialize a new experiment
    EX=Experiment()
    id_dec='decision'
    
    # register basic motion agents;
    # - $True$ tag means they will be marked as dependent (on other agents)
    id_rt, cid_rt = EX.register_sensor('rt')
    id_lt, cid_lt = EX.register_sensor('lt')

    # register motivation for motion agents
    # - this one is NOT dependent on agents except through the position, so
    #   it carries the default False tag.
    id_at_targ,cid_at_targ=EX.register_sensor('atT')
    id_dist=EX.register('dist')
    id_sig=EX.register('sig')
    id_nav, id_navc=EX.register_sensor('nav')

    # register arbiter variable whose purpose is provide a hard-wired response to a conflict
    # between agents 'lt' and 'rt'.
    id_arbiter=EX.register('ar')

    # add a counter
    id_count=EX.register('count')
    def ex_counter(state):
        return 1+state[id_count][0]
    EX.construct_measurable(id_count,ex_counter,[0])
    
    #
    ### Arbitration
    #

    # arbitration state
    def arbiter(state):
        return bool(rnd(2))
    EX.construct_measurable(id_arbiter,arbiter,[bool(rnd(2))],0, decdep=True)

    # intention sensors
    id_toRT, id_toRTc=EX.register_sensor('toR')
    def intention_RT(state):
        return id_rt in state[id_dec][0]
    EX.construct_sensor(id_toRT,intention_RT, decdep=True)
    
    id_toLT, id_toLTc=EX.register_sensor('toL')
    def intention_LT(state):
        return id_lt in state[id_dec][0]
    EX.construct_sensor(id_toLT,intention_LT, decdep=True)

    # failure mode for action $lt^rt$
    id_toF, id_toFc=EX.register_sensor('toF')
    def about_to_enter_failure_mode(state):
        return state[id_toLT][0] and state[id_toRT][0]
    EX.construct_sensor(id_toF,about_to_enter_failure_mode, decdep=True)

    # add basic motion agents with arbitration
    def action_RT(state):
        rt_decided=(id_rt in state[id_dec][0])
        if state[id_toF][0]:
            #return not(rt_decided) if state[id_arbiter][0] else rt_decided
            return state[id_arbiter][0]
        else:
            return rt_decided
    RT=EX.construct_agent(id_rt,id_sig,action_RT,MOTION_PARAMS)

    def action_LT(state):
        lt_decided=(id_lt in state[id_dec][0])
        if state[id_toF][0]:
            #return lt_decided if state[id_arbiter][0] else not(lt_decided)
            return not(state[id_arbiter][0])
        else:
            return lt_decided
    LT=EX.construct_agent(id_lt,id_sig,action_LT,MOTION_PARAMS)
 
    #
    ### "mapping" system
    #

    ## introduce agent's position

    # select starting position
    START=rnd(X_BOUND+1)

    # effect of motion on position
    id_pos=EX.register('pos')
    def motion(state):
        triggers={id_rt:1,id_lt:-1}
        diff=0
        for t in triggers:
            diff+=triggers[t]*int(state[t][0])
        newpos=state[id_pos][0]+diff
        if in_bounds(newpos):
            return newpos
        else:
            return state[id_pos][0]
    EX.construct_measurable(id_pos,motion,[START,START])

    # generate target position
    TARGET=START
    while dist(TARGET,START)<X_BOUND/8:
        TARGET=rnd(X_BOUND+1)

    # staying-at-target sensor
    def stay_at_targetQ(state):
        return state[id_pos][0]==TARGET and state[id_pos][1]==TARGET
    INIT=False
    EX.construct_sensor(id_at_targ,stay_at_targetQ,[INIT,INIT])
    #RT.add_sensor(id_at_targ)
    #LT.add_sensor(id_at_targ)

    # set up position sensors
    def xsensor(m): # along x-axis
        return lambda state: state[id_pos][0]<m+1

    for ind in xrange(X_BOUND):
        tmp_name = 'x'+str(ind)
        id_tmp,id_tmpc=EX.register_sensor(tmp_name) #registers the sensor pairs
        EX.construct_sensor(id_tmp,xsensor(ind)) #constructs the measurables associated with the sensor
        RT.add_sensor(id_tmp)
        LT.add_sensor(id_tmp)

    # distance to target
    # - $id_distM$ has already been registerd
    def dist_to_target(state):
        return dist(state[id_pos][0],TARGET)
    INIT=dist(START,TARGET)
    EX.construct_measurable(id_dist,dist_to_target,[INIT,INIT])
         
    ## value signal for agents LT and RT
    #signal scales logarithmically with distance to target
    rescaling=lambda r: -(1./(1.-Q))*np.log2((1.+r)/(2*(1.+X_BOUND)))
    #signal scales linearly with distance to target
    #rescaling=lambda r: 1.+(1+X_BOUND-r)

    def sig(state):
        #return 1.
        return 1.+state[id_sig][0] if state[id_at_targ][0] else rescaling(state[id_dist][0]) 
    #initial value for signal:
    #INIT=1.
    INIT=rescaling(dist(START,TARGET))
    #construct the motivational signal
    EX.construct_measurable(id_sig,sig,[INIT,INIT])

    # performance sensor ("am I better now than in the last cycle?")
    # - $id_nav$ has already been registered
    def better(state):
        return True if state[id_at_targ][0] else state[id_dist][0]<state[id_dist][1]
    EX.construct_sensor(id_nav,better)
    RT.add_sensor(id_nav)
    LT.add_sensor(id_nav)

    # STOPPED HERE
    #
    #RT.add_sensor(id_rt)
    #RT.add_sensor(id_lt)
    #LT.add_sensor(id_rt)
    #LT.add_sensor(id_lt)

    #-------------------------------------init--------------------------------------------

    for agent_name in EX._AGENTS:
        EX._AGENTS[agent_name].init()
        #EX._AGENTS[agent_name].validate()

    # ONE UPDATE CYCLE (without action) TO "FILL" THE STATE DEQUES
    #ex_report,agent_reports=EX.update_state([cid_rt,cid_lt])

    # INTRODUCE DELAYED GPS SENSORS:
    #for agent in [RT,LT]:
    #    for token in ['plus','minus']:
    #        delay_sigs=[agent.generate_signal(['x'+str(ind)]) for ind in xrange(X_BOUND)]
    #        agent.delay(delay_sigs,token)

    # SET ARTIFICIAL TARGET ONCE AND FOR ALL
    #for agent in [RT,LT]:
    #    for token in ['plus','minus']:
    #        tmp_target=agent.generate_signal([id_nav]).value().tolist()
    #        service_snapshot = ServiceSnapshot(agent._ID, token, service)
    #        service_snapshot.setTarget(tmp_target)

    # ANOTHER UPDATE CYCLE (without action)
    #ex_report,agent_reports=EX.update_state([cid_rt,cid_lt])

    #
    ### Run
    #

    def export_data(my_pickler,ex_report,reports):
        #collect durations and sizes for each agent and for the experiment
        time_rt=reports['rt']['exiting_decision_cycle']-reports['rt']['entering_decision_cycle']
        time_lt=reports['lt']['exiting_decision_cycle']-reports['lt']['entering_decision_cycle']
        ep_rt=reports['rt']['exiting_enrichment_and_pruning']-reports['rt']['entering_decision_cycle']
        ep_lt=reports['lt']['exiting_enrichment_and_pruning']-reports['lt']['entering_decision_cycle']
        size_rt=reports['rt']['size']
        size_lt=reports['lt']['size']
        size_total=len(EX._MID)
        time_total=ex_report['exiting_update_cycle']-ex_report['entering_update_cycle']

        #output as 2-dim np.arrays (points good for matplotlib plotting)
        my_output=((size_rt,time_rt,ep_rt),(size_lt,time_lt,ep_lt),(size_total,time_total))
        #send to the provided pickler
        my_pickler.dump(my_output)
        
    ## Random walk period
    while EX.this_state(id_count)<BURN_IN+2:
        # call output subroutine
        export_data(pickler,ex_report,agent_reports)
        # update the state
        instruction=[(id_lt if rnd(2) else cid_lt),(id_rt if rnd(2) else cid_rt)]
        ex_report,agent_reports=EX.update_state(instruction)


def main():
    #Get script parameters
    try:
        OUTPUT_FILE_NAME=sys.argv[1]
        INTERVAL_LENGTH=int(sys.argv[2])
        NUMBER_OF_RUNS=int(sys.argv[3])
        DISCOUNT_PARAM=int(sys.argv[4])
    except:
        print'Invalid input.\n\n Required arguments are, in order:\n-  output filename preamble\n-  length of interval (environment)\n-  number of runs\n-  values of discount parameter.\n\n'
        exit(0)

    #BATCH
    myf=open(str(OUTPUT_FILE_NAME)+'.dat','ab')
    PICKLER=cPickle.Pickler(myf)
    start_experiment(PICKLER,INTERVAL_LENGTH,NUMBER_OF_RUNS,DISCOUNT_PARAM)
    myf.close()
    
    exit(0)

if __name__=="__main__":
    main()