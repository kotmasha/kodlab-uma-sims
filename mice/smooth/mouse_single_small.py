import numpy as np
from numpy.random import randint as rnd
from numpy.random import seed
from collections import deque
#import curses
import time
from UMA.som2_noEP import *
import sys
import os
from client.UMARest import *
from mouse_base import *
from copy import copy

def printArena(arena):
    for obj in arena._objects:
        print obj._tag + ': '+str(obj._pos.convert) +'\n'


def start_experiment(run_params):
    # experiment parameters
    XBOUNDS=eval(run_params['xbounds'])
    YBOUNDS=eval(run_params['ybounds'])
    OOB=eval(run_params['out_of_bounds'])

    HORIZON=float(run_params['horizon'])
    ROTATION_ORDER=int(run_params['order'])
    MOUSE_PARAMS={'horizon':HORIZON, 'order':ROTATION_ORDER}

    cheesesPerView=float(run_params['cheesesPerView'])
    # initial number of cheeses is set to satisfy, on average, the cheesesPerView constraint
    cheeseNum=int(np.floor(cheesesPerView*((XBOUNDS[1]-XBOUNDS[0])*(YBOUNDS[1]-YBOUNDS[0]))/(np.pi*pow(HORIZON,2))))
    CHEESE_PARAMS={
        'nibbles':int(run_params['nibbles']),
        'maxCheeses':cheeseNum,
        'Ncheeses':cheeseNum,
        }
    # Decision cycles:
    RUN_CYCLES = int(run_params['run_cycles'])
    TRAINING_CYCLES = int(run_params['training_cycles'])
    # Recording options:
    record_mids=run_params['mids_to_record']
    record_global=bool(run_params['ex_dataQ'])
    record_agents=bool(run_params['agent_dataQ'])
    test_name=run_params['test_name']
    # connection to UMA core
    host = run_params['host']
    port = run_params['port']

    ### construct arenas
    #

    arena = Arena_wmouse(
        xbounds=XBOUNDS,
        ybounds=YBOUNDS,
        mouse_params=MOUSE_PARAMS,
        cheese_params=CHEESE_PARAMS,
        visualQ=run_params['visualQ'],
        out_of_bounds=OOB,
        )
    
    ### Mouse/Agents parameters
    #

    # probability of stepping (as opposed to turning)
    STEP_PROB = np.float(run_params['step_prob'])
    
    # agent start parameters: qualitative auto-targeting BUA.
    AGENT_PARAMS = {
        'type': 'qualitative',
        'AutoTarg': True,
        'discount': 0,
    }


    ### Initialize a new experiment
    #

    EX = Experiment(test_name, UMARestService(host, port))
    id_dec = 'decision'
    id_count = 'counter'
    
    # register basic motion agents;
    # - $True$ tag means they will be marked as dependent (on other agents)
    id_rt, cid_rt = EX.register_sensor('rt')
    id_lt, cid_lt = EX.register_sensor('lt')
    id_fd, cid_fd = EX.register_sensor('fd')
    id_bk, cid_bk = EX.register_sensor('bk')

    # register motivation for motion agents
    id_mot_turn = EX.register('mot_turn')
    id_mot_fwd = EX.register('mot_step')
    id_sig_turn = EX.register('sig_turn')
    id_sig_step = EX.register('sig_step')

    ### Arbitration
    #

    # register arbiter variables
    id_arb_turn = EX.register('arb_turn')
    id_arb_step = EX.register('arb_step')
    id_arb_top = EX.register('arb_top')
    
    # arbitration state
    def arbiter_turn(state):
        return bool(arena._rnd.randint(2))
    def arbiter_step(state):
        return bool(arena._rnd.randint(2))
    def arbiter_top(state):
        if arena._rnd.randint(100)<=100*STEP_PROB:
            return True #realizing probability of stepping
        else:
            return False #realizing probability of turning


    EX.construct_measurable(id_arb_turn, arbiter_turn, [arena._rnd.randint(2)], depth=0, decdep=True)
    EX.construct_measurable(id_arb_step, arbiter_step, [arena._rnd.randint(2)], depth=0, decdep=True )
    EX.construct_measurable(id_arb_top, arbiter_top, [arena._rnd.randint(2)], depth=0, decdep=True)

    #----------------turn arbitration----------------------------
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
    id_toF_turn, cid_toF_turn = EX.register_sensor('toF_turn')
    def about_to_enter_failure_mode_turn(state):
        return state[id_toLT][0] and state[id_toRT][0]
    EX.construct_sensor(id_toF_turn, about_to_enter_failure_mode_turn, decdep=False)

    # turn arbitration results:
    id_RT_pTa, cid_RT_pTa=EX.register_sensor('RT_pTa')
    def action_RT(state):
        rt_decided = (id_rt in state[id_dec][0])
        if state[id_toF_turn][0]:
            # return not(rt_decided) if state[id_arbiter][0] else rt_decided
            return state[id_arb_turn][0]
        else:
            return rt_decided
    EX.construct_sensor(id_RT_pTa,action_RT,decdep=False)
    
    id_LT_pTa, cid_LT_pTa=EX.register_sensor('LT_pTa')
    def action_LT(state):
        lt_decided = (id_lt in state[id_dec][0])
        if state[id_toF_turn][0]:
            # return lt_decided if state[id_arbiter][0] else not(lt_decided)
            return (not state[id_arb_turn][0])
        else:
            return lt_decided
    EX.construct_sensor(id_LT_pTa,action_LT,decdep=False)

    #sensor encoding intention to turn, after turn arbitration
    id_turn_post_arb, cid_turn_post_arb = EX.register_sensor('pTa')
    def intention_turn_post_arb(state):
        return state[id_LT_pTa][0] or state[id_RT_pTa][0]
    EX.construct_sensor(id_turn_post_arb,intention_turn_post_arb,decdep=False)

    #------------------step arbitration-----------------------
    id_toFD, cid_toFD = EX.register_sensor('toFD')
    def intention_FD(state):
        return id_fd in state[id_dec][0]
    EX.construct_sensor(id_toFD, intention_FD, decdep=False)

    id_toBK, cid_toBK = EX.register_sensor('toBK')
    def intention_BK(state):
        return id_bk in state[id_dec][0]
    EX.construct_sensor(id_toBK, intention_BK, decdep=False)

    # failure mode for STEP arbitration
    id_toF_step, cid_toF_step = EX.register_sensor('toF_step')
    def about_to_enter_failure_mode_step(state):
        return state[id_toFD][0] and state[id_toBK][0]
    EX.construct_sensor(id_toF_step, about_to_enter_failure_mode_step, decdep=False)

    # FD sensor, post STEP arbitration
    id_FD_pSa, cid_FD_pSa=EX.register_sensor('FD_pSa')
    def action_FD(state):
        fd_decided = (id_fd in state[id_dec][0])
        if state[id_toF_step][0]:
            return state[id_arb_step][0]
        else:
            return fd_decided
    EX.construct_sensor(id_FD_pSa,action_FD,decdep=False)

    # BK sensor, post STEP arbitration
    id_BK_pSa, cid_BK_pSa=EX.register_sensor('BK_pSa')
    def action_BK(state):
        bk_decided = (id_bk in state[id_dec][0])
        if state[id_toF_step][0]:
            return not (state[id_arb_step][0])
        else:
            return bk_decided
    EX.construct_sensor(id_BK_pSa,action_BK,decdep=False)

    #sensor encoding intention to step, after step arbitration
    id_step_post_arb, cid_step_post_arb = EX.register_sensor('pSa')
    def intention_step_post_arb(state):
        return state[id_FD_pSa][0] or state[id_BK_pSa][0]
    EX.construct_sensor(id_step_post_arb,intention_step_post_arb,decdep=False)
    
    #--------------------final arbitration----------------
    #back:
    id_step_bk, cid_step_bk = EX.register_sensor('step_back')
    EX.construct_sensor(id_step_bk, action_BK,decdep=False)
    #forward:
    id_step_fd, cid_step_forward = EX.register_sensor('step_forward')
    EX.construct_sensor(id_step_fd, action_FD,decdep=False)
    #Left:
    id_turn_lt, cid_turn_lt = EX.register_sensor('turn_left')
    EX.construct_sensor(id_turn_lt, action_LT,decdep=False)
    #Right:
    id_turn_rt, cid_turn_rt = EX.register_sensor('turn_right')
    EX.construct_sensor(id_turn_rt, action_RT,decdep=False)

    #top failure mode:
    id_toF, cid_toF = EX.register_sensor('toF')
    def about_to_enter_top_failure_mode(state):
        return state[id_step_post_arb][0] and state[id_turn_post_arb][0]
    EX.construct_sensor(id_toF,about_to_enter_top_failure_mode,decdep=False)


    #-------forward-------;
    def final_action_FD(state):
        if state[id_toF][0]:
            return state[id_arb_top][0] and state[id_step_fd][0]
        else:
            return state[id_step_fd][0]
    
    #-------back----------;
    def final_action_BK(state):
        return False
        #if state[id_toF][0]:
        #    return state[id_arb_top][0] and state[id_step_bk][0]
        #else:
        #    return state[id_step_bk][0]    
    
    #--------Left---------
    def final_action_LT(state):
        if state[id_toF][0]:
            return (not state[id_arb_top][0]) and state[id_turn_lt][0]
        else:
            return state[id_turn_lt][0]    


    #--------Right-------
    def final_action_RT(state):
        if state[id_toF][0]:
            return (not state[id_arb_top][0]) and state[id_turn_rt][0]
        else:
            return state[id_turn_rt][0]    

    #--------Construct the agents--------------
    FD = EX.construct_agent(id_fd, id_sig_step, final_action_FD, AGENT_PARAMS)

    #BK = EX.construct_agent(id_bk, id_sig_step, final_action_BK, AGENT_PARAMS)        
    EX.construct_sensor(id_bk,final_action_BK,[False,False])

    LT = EX.construct_agent(id_lt, id_sig_turn, final_action_LT, AGENT_PARAMS)
    RT = EX.construct_agent(id_rt, id_sig_turn, final_action_RT, AGENT_PARAMS)

    #pos before direction so pos uses state[direction][0] which is the direction of the last state
    #id_pos and id_direction are all that is used to recreate the mouse because there is only one mouse
    
    #--------Constructing the experiment-------------
    ###ARENA UPDATE
    id_arena=EX.register('arena')
    def arena_update(state):
        control=('move',(bool(state[id_fd][0]),bool(state[id_bk][0]),bool(state[id_lt][0]),bool(state[id_rt][0]),bool(state[id_arb_top][0])))
        #update operations on arena
        arena.update(control)
        return arena
    EX.construct_measurable(id_arena, arena_update,[arena],depth=0)

    ###MOUSE ACCESS
    id_mouse=EX.register('mouse')
    def mouse_update(state):
        return copy(state[id_arena][0].getObj('mus'))
    INIT=EX.this_state(id_arena).getObj('mus')
    EX.construct_measurable(id_mouse, mouse_update,[INIT],depth=0)

    # mouse position
    id_pos = EX.register('pos')
    def position(state):
        return state[id_mouse][0]._pos
    INIT=EX.this_state(id_mouse)._pos
    EX.construct_measurable(id_pos,position,[INIT],depth=0)
    # mouse position (output)
    id_pos_out = EX.register('pos_out')
    def pos_out(state):
        return (state[id_pos][0].real,state[id_pos][0].imag)
    INIT=(EX.this_state(id_pos).real,EX.this_state(id_pos).imag)
    EX.construct_measurable(id_pos_out,pos_out,[INIT],depth=0)

    # mouse pose
    id_direction = EX.register('pose')
    def direction(state):
        return state[id_mouse][0]._attr['pose']
    INIT=EX.this_state(id_mouse)._attr['pose']
    EX.construct_measurable(id_direction,direction,[INIT],depth=0)
    # mouse pose (output)
    id_dir_out = EX.register('pose_out')
    def dir_out(state):
        return (state[id_direction][0].real,state[id_direction][0].imag)
    INIT=(EX.this_state(id_direction).real,EX.this_state(id_direction).imag)
    EX.construct_measurable(id_dir_out,dir_out,[INIT],depth=0)

    # cheeses are recorded as a dict of [tag:pos]
    id_cheeses = EX.register('cheeses')
    def cheeses_update(state):
        tmp_objs=state[id_arena][0]._objects
        return {tag:tmp_objs[tag]._pos for tag in tmp_objs.keys() if tmp_objs[tag]._type=='cheese'}
    INIT={tag:arena._objects[tag]._pos for tag in arena._objects.keys() if arena._objects[tag]._type=='cheese'}
    EX.construct_measurable(id_cheeses,cheeses_update,[INIT],depth=0)
    # cheeses for output
    id_che_out = EX.register('che_out')
    def che_out(state):
        ch=state[id_cheeses][0]
        return {tag:(ch[tag].real,ch[tag].imag) for tag in ch}
    INIT={objtag:(arena._objects[objtag]._pos.real,arena._objects[objtag]._pos.imag) for objtag in arena._objects.keys() if arena._objects[objtag]._type=='cheese'}
    EX.construct_measurable(id_che_out,che_out,[INIT],depth=0)

    ### MOTIVATIONAL SIGNALS
    #

    #ell_one distance to nearest cheese
    id_mdist=EX.register('mdist')
    def getMinCheeseDist(state):
        return state[id_mouse][0].mdist()
    INIT=EX.this_state(id_mouse).mdist()
    EX.construct_measurable(id_mdist,getMinCheeseDist,[INIT,INIT],depth=1)

    # stepping motivational signal (already registered) QUALITATIVE AGENT!!!
    def step_signal(state):
        return 2+np.sign(state[id_mdist][0]-state[id_mdist][1]) if state[id_mdist][0]<0.5 else 0
    INIT=2
    EX.construct_measurable(id_sig_step,step_signal,[INIT,INIT])

    # turning motivational signal for QUALITATIVE AGENT!!!   
    rescaling_turn = lambda x: 0 if pow(4,1+x)>15 else 1
    def turn_motivation(state):
        return rescaling_turn(state[id_mouse][0].cos_grad(0))
    INIT=rescaling_turn(arena.getObj('mus').cos_grad(0))
    EX.construct_measurable(id_sig_turn,turn_motivation,[INIT],depth=0)

    #adding sensors to each agent
    #-----------------------length sensors-----------------------------


    #id_LS = {}
    #cid_LS = {}
    #def LS_upd(ind,dirn):
    #    #ind is a numerical index
    #    #dirn is an icomplex direction
    #    return lambda state: state[id_mouse][0].LS(ind,dirn)
    #
    #for dirn in [North,South,East,West]:
    #    for ind in xrange(6):
    #        id_LS[ind,dirn],cid_LS[ind,dirn]=EX.register_sensor('LS_'+get_dir_name(dirn)+'_'+str(ind))
    #        INIT=arena.getObj('mus').LS(ind,dirn)
    #        EX.construct_sensor(id_LS[ind,dirn],LS_upd(ind,dirn),[INIT,INIT])
    #        for agent in [RT,LT,FD]:#,BK]:
    #            agent.add_sensor(id_LS[ind,dirn])
        
    #-----------------------------turn sensors-----------------
    id_angle={}
    cid_angle={}
    def angles_upd(k,ind):
        return lambda state: pow(4.,1+state[id_mouse][0].cos_grad(k))>ind

    for k in xrange(ROTATION_ORDER):
        for ind in {4,10,15}:
            id_angle[k,ind],cid_angle[k,ind]=EX.register_sensor('a_'+str(k)+'_'+str(ind))
            INIT= pow(4.,1.+arena.getObj('mus').cos_grad(k))>ind
            EX.construct_sensor(id_angle[k,ind],angles_upd(k,ind),[INIT,INIT])
            for agent in [RT,LT,FD]:#,BK]:
                agent.add_sensor(id_angle[k,ind])
     
    
    #---------------------------- INIT ------------------------------------------
    for agent_name in EX._AGENTS:
        EX._AGENTS[agent_name].init()

    # an update state to fill the state deques
    EX.update_state([cid_rt, cid_lt, cid_fd, cid_bk])
    if not __name__=='__main__':
        recorder=experiment_output(EX,run_params)
        recorder.addendum('seed',arena._rnd_init_state)
        recorder.addendum('maxCheeses',CHEESE_PARAMS['maxCheeses'])
        recorder.addendum('Ncheeses',CHEESE_PARAMS['Ncheeses'])
        recorder.addendum('cheeseList',[(arena.getObj(tag)._pos.real,arena.getObj(tag)._pos.imag) for tag in arena.getCheeseList()])
        recorder.addendum('pos_out',(arena.getObj('mus').getPos().real,arena.getObj('mus').getPos().imag))
        recorder.addendum('pose_out',(arena.getObj('mus').getAttr('pose').real,arena.getObj('mus').getAttr('pose').imag))
        recorder.addendum('counter',arena.getMiscVal('counter'))

    #client data objects for the experiment
    UMACD={}
    for agent_id in EX._AGENTS:
        for token in ['plus','minus']:
            UMACD[(agent_id,token)]=UMAClientData(EX._EXPERIMENT_ID,agent_id,token,EX._service)

    # ASSIGN TARGET IF NOT AUTOMATED:
    if AGENT_PARAMS['AutoTarg']:
        pass
    else:
        # SET ARTIFICIAL TARGET ONCE AND FOR ALL
        for agent_id in EX._AGENTS:
            for token in ['plus','minus']:
                tmp_target=agent.generate_signal([id_nav],token).value().tolist()
                UMACD[(agent._ID,token)].setTarget(tmp_target)
                
    # introducing a delayed sensor for every initial sensor, into each snapshot:
    for agent in [RT,LT,FD]:
        for token in ['plus','minus']:
            delay_sigs = []
            for mid in agent._SNAPSHOTS[token]._SENSORS:
                delay_sigs.append(agent.generate_signal([mid],token))
            agent.delay(delay_sigs, token)


    if run_params['visualQ']:
        plt.show()
    
    while EX.this_state(id_count)<TRAINING_CYCLES:
        instruction = [
            (id_lt if arena._rnd.randint(2) else cid_lt),
            (id_rt if arena._rnd.randint(2) else cid_rt),
            (id_fd if arena._rnd.randint(2) else cid_fd),
            (id_bk if arena._rnd.randint(2) else cid_bk)]
        EX.update_state(instruction)
        if not __name__=='__main__':
            recorder.record()

    while EX.this_state(id_count)<TRAINING_CYCLES+RUN_CYCLES:
        EX.update_state()
        if not __name__=='__main__':
            recorder.record()
    
    # Wrap up:
    if not __name__=='__main__':
        recorder.close()

    EX.remove_experiment()
    
def main():
    plt.ion()
    RUN_PARAMETERS={
	'test_name': "default_test",
        'xbounds': "(0.,15.)",
        'ybounds': "(0.,15.)",
        'out_of_bounds': "lambda x:False",
        'order': "20",
        'horizon': "5",
        'step_prob': "0.5",
        'cheesesPerView': "1",
        'nibbles': "50",
        'visualQ': "True",
        'training_cycles': "10",
        'run_cycles': "10", 
        'mids_to_record': "[]",
        'ex_dataQ': "False",
        'agent_dataQ': "False",
        'host': 'localhost',
        'port': "8000",
        }
    start_experiment(RUN_PARAMETERS)

if __name__=='__main__':
    main()
    exit(0)

# make sure this thing runs and produces a movie
# add a default recording of the movie
# 
# introduce the learned arbiter for lt/rt

