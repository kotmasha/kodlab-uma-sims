import numpy as np
from numpy.random import randint as rnd
from collections import deque
import curses
import time
from som_platform import *


def start_experiment(stdscr,agent_to_examine):
    NODELAY=0
    # grid definitions
    X_BOUND=4 #length
    def in_bounds(pos):
        return (pos>=0 and pos<=X_BOUND)

    # distance function
    def dist(p,q):
        return abs(p-q) 

    # agent discount parameters
    MOTION_PARAMS=tuple([1.-pow(2,-6)])
    #ARBITRATION_PARAMS=tuple([1.-pow(2,-2)])
    #ARBITRATION_PEAK=2
    
    # initialize a new experiment
    EX=Experiment()
    id_dec=EX.nid('decision')
    
    # register basic motion agents;
    # - $True$ tag means they will be marked as dependent (on other agents)
    id_rt,id_rtc=EX.register_sensor('rt',True)
    id_lt,id_ltc=EX.register_sensor('lt',True)
    # register motivation for motion agents
    # - this one is NOT dependent on agents except through the position, so
    #   it carries the default False tag.
    id_distM=EX.register('distM')
    id_navM,cid_navM=EX.register_sensor('navM')
    # register supervising agent
    #id_super,id_superc=EX.register_sensor('super',True)

    # agent to be visualized
    id_lookat=EX.nid(agent_to_examine)
    
    # register arbiter variable whose purpose is to synchronize the responses
    # of agents 'lt' and 'rt' to the action of 'super', it does not depend on
    # agent decisions, hence carries the default False tag.
    id_arbiter=EX.register('ar',True)
    # register the failure mode sensor
    #id_fail,id_failc=EX.register_sensor('fl')

    # add a counter
    id_count=EX.register('count')
    def ex_counter(state):
        return 1+state[id_count][0]
    EX.construct_measurable(id_count,ex_counter,[0])
    
    # introduce arbitration
    def arbiter(state):
        return bool(rnd(2))
    EX.construct_measurable(id_arbiter,arbiter,[bool(rnd(2))],0)

    id_toRT,id_toRTc=EX.register_sensor('toR',True)
    def intention_RT(state):
        return id_rt in state[id_dec][0]
    EX.construct_sensor(id_toRT,intention_RT)
    
    id_toLT,id_toLTc=EX.register_sensor('toL',True)
    def intention_LT(state):
        return id_lt in state[id_dec][0]
    EX.construct_sensor(id_toLT,intention_LT)

    # failure mode for action $lt^rt$
    id_toF,id_toFc=EX.register_sensor('toF',True)
    def about_to_enter_failure_mode(state):
        return state[id_toLT][0] and state[id_toRT][0]
    EX.construct_sensor(id_toF,about_to_enter_failure_mode)

    # add basic motion agents
    def action_RT(state):
        rt_decided=(id_rt in state[id_dec][0])
        if state[id_toF][0]:
            #return not(rt_decided) if state[id_arbiter][0] else rt_decided
            return state[id_arbiter][0]
        else:
            return rt_decided
    RT=EX.construct_agent(id_rt,id_distM,action_RT,MOTION_PARAMS,True)

    def action_LT(state):
        lt_decided=(id_lt in state[id_dec][0])
        if state[id_toF][0]:
            #return lt_decided if state[id_arbiter][0] else not(lt_decided)
            return not(state[id_arbiter][0])
        else:
            return lt_decided
    LT=EX.construct_agent(id_lt,id_distM,action_LT,MOTION_PARAMS,True)
    
    # immediately introduce corresponding action sensors
    #EX.assign_sensor(id_rt,True,[id_lt])
    #EX.assign_sensor(id_lt,True,[id_rt])

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

    ## introduce effect of agent (not/)feeding (finding and consuming targets):

    # generate target position
    TARGET=START
    while dist(TARGET,START)<X_BOUND/8:
        TARGET=rnd(X_BOUND+1)

    # set up position sensors
    def xsensor(m): # along x-axis
        return lambda state: state[id_pos][0]<m+1

    for ind in xrange(X_BOUND):
        tmp_name='x'+str(ind)
        id_tmp,id_tmpc=EX.register_sensor(tmp_name) #registers the sensor pairs
        EX.construct_sensor(id_tmp,xsensor(ind)) #constructs the measurables associated with the sensor
        EX.assign_sensor(id_tmp,True,[id_rt,id_lt]) #assigns the sensor to all agents


    # normalized distance to playground (nav function #1)
    # - $id_distM$ has already been registerd 
    def distM(state):
        if state[id_pos][0]==TARGET and state[id_pos][1]==TARGET:
            return state[id_distM][0]+1
        else:
            #sharp (near-logarithmic) spike at the target 
            #return 1.-np.log((1.+dist(state[id_pos][0],TARGET))/(X_BOUND+2))
            #linear peak at the target
            return 1+X_BOUND-dist(state[id_pos][0],TARGET)

    INIT=1+X_BOUND-dist(START,TARGET)
    EX.construct_measurable(id_distM,distM,[INIT,INIT])

    def navM(state):
        return state[id_distM][0]-state[id_distM][1]>0
    EX.construct_sensor(id_navM,navM)
    EX.assign_sensor(id_navM,True,[id_rt,id_lt]) #assigns the sensor to all agents
    
    #
    ### Initialize agents on GPU
    #
    
    for agent_name in EX._AGENTS:
        tmp=EX._AGENTS[agent_name].init()

    #
    ### Introduce a conjunction
    #

    #for agent in [RT,LT]:
    #    agent.amper([agent.generate_signal([EX.nid('x0*'),EX.nid('x2')])])
    #    

    ### Introduce delayed position sensors for both agents
    
    for agent in [RT,LT]:
        delay_sigs=[agent.generate_signal([EX.nid('x'+str(ind))]) for ind in xrange(X_BOUND)]
        agent.delay(delay_sigs)


        
    # another update cycle
    message=EX.update_state()
    #message=EX.update_state()

    ## SET ARTIFICIAL TARGET ONCE AND FOR ALL
    for agent in [RT,LT]:
        for token in ['plus','minus']:
            tmp_target=agent.generate_signal([id_navM]).value_all().tolist()
            agent.brain._snapshots[token].setTarget(tmp_target)

            
    #
    ### Run
    #

    # prepare windows for output
    curses.curs_set(0)
    stdscr.erase()

    # color definitions
    curses.init_color(0,0,0,0)    #black=0
    curses.init_color(1,1000,0,0) #red=1
    curses.init_color(2,0,1000,0) #green=2
    curses.init_color(3,1000,1000,0) #yellow=3
    curses.init_color(4,1000,1000,1000) #white=4
    curses.init_color(5,1000,1000,500)
    
    curses.init_pair(1,0,1) #black on red
    curses.init_pair(2,0,2) #green on black
    curses.init_pair(3,0,3) #black on yellow
    curses.init_pair(4,4,0) #white on black
    curses.init_pair(5,1,0) #red on black
    curses.init_pair(6,0,5)
        
    REG_BG=curses.color_pair(4) | curses.A_BOLD
    POS_BG=curses.color_pair(2) | curses.A_BOLD
    NEG_BG=curses.color_pair(1) | curses.A_BOLD
    OBS_BG=curses.color_pair(6) | curses.A_BOLD
    BG=curses.color_pair(5) | curses.A_BOLD
    FG=curses.color_pair(3) | curses.A_BOLD
    
    WIN=curses.newwin(9,2*X_BOUND+3,5,7)
    WINs=curses.newwin(9,200,16,7)
    stdscr.nodelay(NODELAY)

    WIN.bkgdset(ord('.'),REG_BG)
    
    WIN.overlay(stdscr)
    WINs.overlay(stdscr)
    
    def print_state(text,id_agent):
        stdscr.clear()
        stdscr.addstr('W-E  A-R-E  R-U-N-N-I-N-G    (press [space] to stop) ')
        stdscr.addstr(2,3,text)
        stdscr.clrtoeol()
        stdscr.noutrefresh()
        WIN.clear()
        WIN.addstr(0,0,str(EX.this_state(id_count,1)))
        WIN.chgat(0,0,BG)
        
        ## Unpacking the output from the tested agent (RT/LT)
        
        #determining the start position of geographic sensors in signals
        geo_start=min(ind for ind,id_tmp in enumerate(LT._SENSORS) if id_tmp==EX.nid('x0'))
        #decompose extra info from agent
        agent=EX._AGENTS[id_agent]
        curr=agent._CURRENT
        targ=agent._TARGET
        pred=agent._PREDICTED
        
        #choose the signals to visualize (curr,targ or pred)
        
        for ind,lookat in enumerate([curr,pred,targ]): 
            #convert signals to region bounds
            bounds={'plus':[0,X_BOUND],'minus':[0,X_BOUND]}
            for token in ['plus','minus']:
                for x in xrange(0,X_BOUND):
                    if lookat[token].value(geo_start+2*x):
                        bounds[token][1]=x #pushing down the upper bound
                        break
                    else:
                        continue
                for x in xrange(X_BOUND-1,-1,-1):
                    if lookat[token].value(geo_start+2*x+1):
                        bounds[token][0]=x+1 #pushing up the lower bound
                        break
                    else:
                        continue
            
            #display the results
            tok_BG={'plus':POS_BG,'minus':NEG_BG}
            tok_line={'plus':3-ind,'minus':6+ind}
            for token in ['plus','minus']:
                min_pos=bounds[token][0]
                max_pos=bounds[token][1]
                for x in xrange(0,X_BOUND):
                    ori=ord('<') if lookat[token].value(geo_start+2*x) else (ord('>') if lookat[token].value(geo_start+2*x+1) else ord('*'))
                    this_BG=tok_BG[token] if (x>=min_pos and x<max_pos) else BG
                    WIN.addch(tok_line[token],2+2*x,ori,this_BG)

                    WIN.chgat(tok_line[token],1+2*min_pos,1+2*(max_pos-min_pos),tok_BG[token])
        # display targets with FG attributes
        WIN.addch(5,1+2*TARGET,ord('T'),FG)
        # display agent's position with FG attributes
        WIN.addch(4,1+2*EX.this_state(id_pos,1),ord('S'),FG)
        
        ## Unpacking extra information
        WINs.clear()
        WINs.addstr(0,0,'Observation:')
        WINs.addstr(4,0,'Chosen signed signal:')
        #
        tok_BG={'plus':POS_BG,'minus':NEG_BG}
        vpos={'plus':6,'minus':8}
        hpos=lambda x: 0 if x==0 else 2+len('  '.join(namelist[:x]))
        
        # CURRENT OBSERVATION
        OBS=agent._OBSERVE
        #OBS=Signal([EX.this_state(mid) for mid in agent._SENSORS])

        #SIGNED SIGNAL TO WATCH:
        #SIG=agent._CURRENT
        
        sig=agent.generate_signal([EX.nid('x0')])
        #sig=agent.generate_signal([EX.nid('{x0*;x2}')])
        #sig=agent.generate_signal([EX.nid('#x2')])
        #sig=agent.generate_signal([EX.nid('#x0*')])
        SIG=agent.brain.up(sig,False)
        #
        namelist=[EX.din(mid) for mid in agent._SENSORS]
        #
        for x,mid in enumerate(agent._SENSORS):
            this_BG=OBS_BG if OBS.value(x) else REG_BG
            WINs.addstr(2,hpos(x),namelist[x],this_BG)
            for token in ['plus','minus']:
                this_BG=tok_BG[token] if SIG[token].value(x) else REG_BG
                WINs.addstr(vpos[token],hpos(x),namelist[x],this_BG)
        
        # refresh the window
        WIN.overlay(stdscr)
        WIN.noutrefresh()
        WINs.noutrefresh()
        curses.doupdate()



        
    ## Main loop
    while stdscr.getch()!=ord(' '):

        # call output subroutine
        print_state('RUNNING:\n'+message,id_lookat)

        # make decisions, update the state
        message=EX.update_state()

    else:
        raise Exception('Aborting at your request...\n\n')
        
curses.wrapper(start_experiment,'rt')
exit(0)

