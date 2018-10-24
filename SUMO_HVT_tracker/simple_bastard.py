import cPickle
import numpy as np 
import curses
from numpy.random import randint as rnd
from copy import deepcopy

import myp #module for handling pickled frames
import mobs #module defining data structures for tracker agents

from som_platform import *

### Data file definitions.
#
# Preamble file (read only) contains pickled fixed parameters of the
#    scene and the tracking problem;
# Content file (read only) contains a pickled frame-by-frame description
#    of the scene;
# Stamp data structure (r/w) contains pickled frame-by-frame records by
#    the agent (one stamp file per "physical" agent)
#
## Preamble data file structure ('_name_.preamble'):
#  -  a double array of integers, $scene[y][x]$ whose values are:
#     0 if position is accessible to cars
#     n if position is inaccessible and occupied by entity labelled n
#          e.g. 1 for "pavement"
#               2 and on for "buildings"
#  -  a double array of integers, $occ[y][x]$ whose values are $False/True$ 
#     depending on whether a car in position (x,y) would be visible to the
#     tracker or not, respectively
#  -  initial high value target assignment $(target_id,start_frame)$
#  -  type dictionary (for cars), of the form $target_id:kind$, where each
#     $kind$ is a 3-bit color (0-7)
#  -  number of frames
#
#
# Content data file structure ('_name_.content'):
#  frames are pickled dictionaries of the form
#    $target_id:position,velocity$
#  where $position$ and $velocity$ are of type $mobs.icomplex$

def read_preamble(input_file_name):
    obj=myp.myp(input_file_name+'.preamble',0)
    scene=obj.content(0)
    obj.advance()
    occ=obj.content(0)
    obj.advance()
    hvt_info=obj.content(0)
    obj.advance()
    target_descriptions=obj.content(0)
    obj.advance()
    number_of_frames=obj.content(0)
    return scene,occ,hvt_info,target_descriptions,number_of_frames

def signedstr(num):
    return str(num) if num<0 else '+'+str(num)

def rpermutation(input_list):
    input_copy=deepcopy(input_list)
    new_list=[]
    while input_copy:
        new_list.append(input_copy.pop(rnd(len(input_copy))))
    return new_list

def hvt_tracker(input_file_name_with_path,ff_discount,zoom_discount,CYCLES):
    # input file name should include path but not extension
    # agent_file name should include path but not extension
    
    
    ###---------------------
    ### Start the experiment
    #
    CARS=Experiment()
    id_dec=CARS.nid('decision')
    
    ###------------------------------------------------------------
    ### Read the input file (with preamble) and prepare stamp files
    #
    if type(input_file_name_with_path)!=type('blah'):
        raise Exception('Input file name must be a string -- Aborting.')
    SCENE,OCC,HVT_INFO,TARGET_DESCRIPTIONS,TOTAL_FRAMES=read_preamble(input_file_name_with_path)
    
    # unpack initial data for tracker
    HVT_ID,START_FRAME_NUMBER=HVT_INFO # acquire initial data

    # Form the viewport by unpickling the input file
    TIME_WINDOW=1 # number of frames in viewport is 2*TIME_WINDOW+1
    window=myp.myp(input_file_name_with_path+'.content',TIME_WINDOW)

    # Read HVT ground truth position values; list to be used in computing TLE.
    HVT_ground_truth=[]
    for frame in xrange(0,TOTAL_FRAMES-1):
        if HVT_ID in window.content(0):
            HVT_ground_truth.append((frame,window.content(0)[HVT_ID][0]))
        window.advance()

    #for item in HVT_ground_truth:
    #    print item
        
    # Prepare window for run
    window.moveto(START_FRAME_NUMBER) # center $window$ on start frame
    START_POSITION,START_VELOCITY=window.content(0)[HVT_ID] 
    
    # Prepare stamp files
    RESOLUTION=2
    target_here=mobs.marker(input_file_name_with_path+'_target_trace.dat',mobs.ptz(RESOLUTION))
    
    
    ###-------------------
    ### SETTING UP ACTIONS
    #
    
    ### SCENE PARAMETERS
    Y_BOUND=len(SCENE) # SCENE is a X_BOUND*Y_BOUND double array...
    X_BOUND=len(SCENE[0])
    
    ZOOM_DEPTH=int(np.floor(np.log(min(X_BOUND,Y_BOUND))/np.log(RESOLUTION)))
    KIND_BIT_RESOLUTION_STEP=2
    HVT_CONFIRM_ZOOM_LEVEL=ZOOM_DEPTH-KIND_BIT_RESOLUTION_STEP
    
    ### ADV/REW agents and arbitration
    
    ## advance/rewind viewing window initialization
    FF_DISCOUNT=ff_discount
    INIT_PARAMS_FF=[FF_DISCOUNT]
    
    ## register relevant measurables:
    id_adv,cid_adv=CARS.register_sensor('adv',True) #dependent, action sensor
    id_rew,cid_rew=CARS.register_sensor('rew',True) #dependent, action sensor
    id_sig_adv=CARS.register('AdvSig') #independent, frame action signal
    id_sig_rew=CARS.register('RewSig') #independent, frame action signal
    id_arb_frames=CARS.register('FrArb') #independent, frame action arbiter
    
    #construct frame advancement arbiter variable
    def arbiter_frames(state):
        return bool(rnd(2))
    CARS.construct_measurable(id_arb_frames,arbiter_frames,[rnd(2)],0)
    
    #define and construct agents
    def advance_frame(state):
        if id_adv in state[id_dec][0] and id_rew in state[id_dec][0]:
            return state[id_arb_frames][0]
        else:
            return (id_adv in state[id_dec][0])
    
    def rewind_frame(state):
        if id_adv in state[id_dec][0] and id_rew in state[id_dec][0]:
            return not(state[id_arb_frames][0])
        else:
            return (id_rew in state[id_dec][0])
            
    ADV=CARS.construct_agent(id_adv,id_sig_adv,advance_frame,INIT_PARAMS_FF,True)
    REW=CARS.construct_agent(id_rew,id_sig_rew,rewind_frame,INIT_PARAMS_FF,True)

    ### Zoom agents and arbitration
    
    # zoom in/out initialization parameters
    ZOOM_DISCOUNT=zoom_discount
    INIT_PARAMS_ZOOM=[ZOOM_DISCOUNT]
    # register zoomout agent
    id_zout,cid_zout=CARS.register_sensor('zout',True)
    # register zoomin agents
    id_zin={}
    cid_zin={}
    ZOOM_KEYS=[mobs.icomplex(x,y) for x in xrange(RESOLUTION) for y in xrange(RESOLUTION)]
    for key in ZOOM_KEYS:
        tmp,ctmp=CARS.register_sensor('zin'+str(key),True)
        id_zin[key]=tmp
        cid_zin[key]=ctmp
    
    # register arbitration variable
    id_arb_zoom=CARS.register('ZArb')
    # register motivational signal for zoom agents
    id_sig_zout=CARS.register('ZoutSig')
    id_sig_zin=CARS.register('ZinSig')

    
    # construct the arbitration
    def arbiter_zoom(state):
        return bool(rnd(2)),rpermutation(ZOOM_KEYS)
    CARS.construct_measurable(id_arb_zoom,arbiter_zoom,[bool(rnd(2)),rpermutation(ZOOM_KEYS)],0)

    
    # construct zoomout action
    def zoomout(state):
        b,_=state[id_arb_zoom][0]
        zin_total=sum([(id_zin[key] in state[id_dec][0]) for key in ZOOM_KEYS])
        if cid_zout in state[id_dec][0]: #initial decision is not to act, so no conflict with the other zoom agents
            return False
        elif (id_zout in state[id_dec][0]) and (zin_total!=1): #conflict with more than one zoomin agent gives zoomout precedence
            return True
        else: #conflict with a single zoomin agent is resolved via the arbiter
            return b

            
    def zoomin(pos): # pos is of the form 'x+yj', that is: a id_zin key
        def f(state):
            b,perm=state[id_arb_zoom][0]
            zin_total=sum([(id_zin[key] in state[id_dec][0]) for key in ZOOM_KEYS])
            if not(id_zout in state[id_dec][0]): #if zout is off then use arbiter to pick one of the zins that are on
                perm=[key for key in perm if id_zin[key] in state[id_dec][0]]
                try:
                    return pos==perm[0]
                except:
                    return False
            elif (id_zout in state[id_dec][0]) and (zin_total!=1): #if zout is on but faces no / too much competition, zout suppresses the competition
                return False
            else: #zout is on and matches the competition
                return (not(b) and (id_zin[pos] in state[id_dec][0]))
        return f

    #construct zoomout agent
    CARS.construct_agent(id_zout,id_sig_zout,zoomout,INIT_PARAMS_ZOOM,True)

    #construct zoomin agents
    for key in ZOOM_KEYS:
        CARS.construct_agent(id_zin[key],id_sig_zin,zoomin(key),INIT_PARAMS_ZOOM,True)
        
    ###-----------------------------
    ### CONSTRUCTING THE MEASURABLES
    #
    ##  Leftover from above: id_sig_frames, id_sig_zout, id_sig_zin
    #
    
    # Convert ptz(RESOLUTION) data element into the corresponding center point
    # in the full visual scene
    def ptz_to_pos(ptz): #returns (approximate) centerpoint of ptz
        xb=X_BOUND
        yb=Y_BOUND
        tmp_ptz=ptz.state_all()
        tmp_x=0
        tmp_y=0
        while len(tmp_ptz)>0:
            m=tmp_ptz.popleft()
            tmp_x+=int(np.ceil((m.real+0.)*xb/RESOLUTION))
            tmp_y+=int(np.ceil((m.imag+0.)*yb/RESOLUTION))
            xb=min(xb,int(np.ceil((m.real+1.)*xb/RESOLUTION)))-tmp_x
            yb=min(yb,int(np.ceil((m.imag+1.)*yb/RESOLUTION)))-tmp_y
        return mobs.icomplex(0.5*(tmp_x+xb),0.5*(tmp_y+yb))
    
    # Convert position START_POSITION into ptz data elements
    def pos_to_ptz(pos):
        xb=X_BOUND #assuming pox.real has values from 0 to X_BOUND-1
        yb=Y_BOUND #and similarly for the y axis
        new_ptz=mobs.ptz(RESOLUTION)
        tmp_x=pos.real
        tmp_y=pos.imag
        for depth in xrange(ZOOM_DEPTH): # zoom in all the way onto pos
            nzx=int(np.floor((0.+RESOLUTION*tmp_x)/xb))
            nzy=int(np.floor((0.+RESOLUTION*tmp_y)/yb))
            new_ptz.zoomin(mobs.icomplex(nzx,nzy))
            tmp_x-=int(np.ceil((nzx+0.)*xb/RESOLUTION))
            tmp_y-=int(np.ceil((nzy+0.)*yb/RESOLUTION))
            xb=min(xb,int(np.ceil((nzx+1.)*xb/RESOLUTION)))-int(np.ceil((nzx+0.)*xb/RESOLUTION))
            yb=min(yb,int(np.ceil((nzy+1.)*yb/RESOLUTION)))-int(np.ceil((nzy+0.)*yb/RESOLUTION))
        return new_ptz
    
    START_PTZ=pos_to_ptz(START_POSITION)

    # time counter
    id_counter=CARS.register('count')
    def time_counter(state):
        return 1+state[id_counter][0]
    CARS.construct_measurable(id_counter,time_counter,[0],0)
    
    # frame counter
    id_frameno=CARS.register('frameno')
    def frame_counter(state):
        triggers={id_adv:1,id_rew:-1}
        return state[id_frameno][0]+sum([triggers[mid]*int(state[mid][0]) for mid in triggers])
    CARS.construct_measurable(id_frameno,frame_counter,[START_FRAME_NUMBER],0)

    target_here.mark(START_FRAME_NUMBER,START_PTZ)
    
    # access to viewport reader from experiment state
    id_active_viewport=CARS.register('active_viewport')
    def active_viewport(state):
        if state[id_adv][0]:
            state[id_active_viewport][0].advance()
        if state[id_rew][0]:
            state[id_active_viewport][0].rewind()
        return state[id_active_viewport][0]
    CARS.construct_measurable(id_active_viewport,active_viewport,[window],0)
    
    # viewport information
    id_viewport=CARS.register('viewport')
    def viewport(state):
        return state[id_active_viewport][0].content_all()
    CARS.construct_measurable(id_viewport,viewport)
    
    # the ptz state as a measurable
    id_ptz=CARS.register('ptz')
    def ptzmotion(state):
        #zooming is not commutative (or even composable!)
        if state[id_zout][0]:
            #zoomout trumps any and all zoomin actions
            return state[id_ptz][0].zoomout()
        #if no zoomout, collect the zoomin actions
        zoom_dirs=[key for key in ZOOM_KEYS if state[id_zin[key]][0]]
        if len(zoom_dirs)==1:
            #if only one zoomin action (and no zoomout), take it
            return state[id_ptz][0].zoomin(zoom_dirs[0])
        else:
            #for any other action, the tracker ptz state does not change
            return state[id_ptz][0]
                
    CARS.construct_measurable(id_ptz,ptzmotion,[START_PTZ])
    
    # zoomin depth
    id_zdepth=CARS.register('depth')
    def zoom_depth(state):
        return len(state[id_ptz][0].state_all())
    CARS.construct_measurable(id_zdepth,zoom_depth)
    
    
    ###--------
    ### MARKERS
    #
    
    # access to HVT marker state from experiment state
    id_active_HVT_marker=CARS.register('active_hvt_marker')
    def active_HVT_marker(state):
        #version without marker agents
        current_ptz=deepcopy(state[id_ptz][0])
        hvt_pos=state[id_viewport][0][TIME_WINDOW][HVT_ID][0]
        hvt_ptz=pos_to_ptz(hvt_pos)
        #raise Exception('aha....')
        if current_ptz<=hvt_ptz and not OCC[hvt_pos.imag][hvt_pos.real] and state[id_zdepth][0]>=HVT_CONFIRM_ZOOM_LEVEL:
            #raise Exception('aha....')
            state[id_active_HVT_marker][0].mark(state[id_frameno][0],current_ptz)
        return state[id_active_HVT_marker][0]
    
    CARS.construct_measurable(id_active_HVT_marker,active_HVT_marker,[target_here],0)
    
    # HVT marks observed in viewport
    id_HVT_marker=CARS.register('HVT_marker')
    def HVT_marker(state):
        return state[id_active_HVT_marker][0].report(state[id_frameno][0]-TIME_WINDOW,state[id_frameno][0]+TIME_WINDOW)
    
    CARS.construct_measurable(id_HVT_marker,HVT_marker,[target_here.report(START_FRAME_NUMBER-TIME_WINDOW,START_FRAME_NUMBER+TIME_WINDOW)])
    
    
    ###----------------------
    ### Tracker state sensors
    #
    
    ## define position sensors at varying depths
    id_dstate={}
    cid_dstate={}
    def dstate(d): # depth d in xrange(ZOOM_DEPTH)
        return lambda state: state[id_zdepth][0]<d+1
    
    id_xdstate={}
    cid_xdstate={}
    def xdstate(d,m): # starts from d=1
        return lambda state: state[id_ptz][0].state_all()[ZOOM_DEPTH-1-d].real<m+1 if d<state[id_zdepth][0] else False
    
    id_ydstate={}
    cid_ydstate={}
    def ydstate(d,m): # starts from d=1
        return lambda state: state[id_ptz][0].state_all()[ZOOM_DEPTH-1-d].imag<m+1 if d<state[id_zdepth][0] else False
    
    ## construct the corresponding sensors
    for d in xrange(ZOOM_DEPTH):
        id_dstate[d],cid_dstate[d]=CARS.register_sensor('d'+str(d))
        CARS.construct_sensor(id_dstate[d],dstate(d))
        CARS.assign_sensor(id_dstate[d],True,CARS._AGENTS.keys())
        if d>0:
            for x in xrange(RESOLUTION-1):
                id_xdstate[(x,d)],cid_xdstate[(x,d)]=CARS.register_sensor('d'+str(d)+'x'+str(x))
                CARS.construct_sensor(id_xdstate[(x,d)],xdstate(d,x))
                CARS.assign_sensor(id_xdstate[(x,d)],True,CARS._AGENTS.keys())
            for y in xrange(RESOLUTION-1):
                id_ydstate[(y,d)],cid_ydstate[(y,d)]=CARS.register_sensor('d'+str(d)+'y'+str(y))
                CARS.construct_sensor(id_ydstate[(y,d)],xdstate(d,y))
                CARS.assign_sensor(id_ydstate[(y,d)],True,CARS._AGENTS.keys())
            continue
        continue
                
    ###------------------------
    ### Target tracking sensors
    #
    
    
    ## HVT marker sensors
    
    # definitions
    id_is_view_marked={}
    cid_is_view_marked={}
    def is_my_view_HVT_marked(rel_frame):
        def tmp(state):
            this_ptz=state[id_ptz][0]
            this_frame=state[id_frameno][0]
            return this_ptz<=state[id_HVT_marker][0][rel_frame+TIME_WINDOW]
        return tmp
    
    id_do_I_see_HVT_marking={}
    cid_do_I_see_HVT_marking={}
    def do_I_see_HVT_marking(rel_pos,rel_frame):
        def tmp(state):
            # Check whether the agent's ptz state, zoomed in at $rel_pos$
            # is marked
            this_ptz=deepcopy(state[id_ptz][0])
            this_ptz.zoomin(rel_pos)
            this_frame=state[id_frameno][0]
            return this_ptz<=state[id_HVT_marker][0][rel_frame+TIME_WINDOW]
        return tmp

    #adding the sensors
    for rel_frame in xrange(-TIME_WINDOW,TIME_WINDOW+1):
        id_is_view_marked[rel_frame],cid_is_view_marked[rel_frame]=CARS.register_sensor('mk@ptz'+signedstr(rel_frame)+'f')
        CARS.construct_sensor(id_is_view_marked[rel_frame],is_my_view_HVT_marked(rel_frame))
        CARS.assign_sensor(id_is_view_marked[rel_frame],True,CARS._AGENTS.keys())
        for key in ZOOM_KEYS:
            id_do_I_see_HVT_marking[(key,rel_frame)],cid_do_I_see_HVT_marking[(key,rel_frame)]=CARS.register_sensor('mk@'+str(key)+signedstr(rel_frame)+'f')
            CARS.construct_sensor(id_do_I_see_HVT_marking[(key,rel_frame)],do_I_see_HVT_marking(key,rel_frame))
            CARS.assign_sensor(id_do_I_see_HVT_marking[(key,rel_frame)],True,CARS._AGENTS.keys())
            continue
        continue
    
    ## HVT location sensors
    
    #definitions
    id_do_I_see_the_HVT={}
    cid_do_I_see_the_HVT={}
    def do_I_see_the_HVT(rel_pos,rel_frame): #possible occlusions taken into account
        def tmp(state):
            current_ptz=deepcopy(state[id_ptz][0])
            current_ptz.zoomin(rel_pos)
            hvt_pos=state[id_viewport][0][rel_frame+TIME_WINDOW][HVT_ID][0]
            return current_ptz<=pos_to_ptz(hvt_pos) and not OCC[hvt_pos.imag][hvt_pos.real] and state[id_zdepth][0]>=HVT_CONFIRM_ZOOM_LEVEL
        return tmp
    
    #adding the sensors
    for rel_frame in xrange(-TIME_WINDOW,TIME_WINDOW+1):
        for key in ZOOM_KEYS:
            id_do_I_see_the_HVT[(key,rel_frame)],cid_do_I_see_the_HVT[(key,rel_frame)]=CARS.register_sensor('hv@'+str(key)+signedstr(rel_frame)+'f')
            CARS.construct_sensor(id_do_I_see_the_HVT[(key,rel_frame)],do_I_see_the_HVT(key,rel_frame))
            CARS.assign_sensor(id_do_I_see_the_HVT[(key,rel_frame)],True,CARS._AGENTS.keys())
            continue
        continue
    
    
    ## HVT x-velocity sensors
    
    #definitions
    id_HVT_vx={}
    cid_HVT_vx={}
    def HVT_x_velocity_exceeds(val):
        if val>0:
            def tmp(state):
                return state[id_viewport][0][TIME_WINDOW][HVT_ID][1].real>val
        elif val<0:
            def tmp(state):
                return state[id_viewport][0][TIME_WINDOW][HVT_ID][1].real<val
        else:
            def tmp(state):
                return state[id_viewport][0][TIME_WINDOW][HVT_ID][1].real!=0
        return tmp
    
    #adding the sensors
    #negative x-velocity
    for ind in xrange(ZOOM_DEPTH/2,ZOOM_DEPTH+1): #assuming targets are not too fast...
        val=-X_BOUND/pow(RESOLUTION,ind)
        id_HVT_vx[val],cid_HVT_vx[val]=CARS.register_sensor('vx'+signedstr(val))
        CARS.construct_sensor(id_HVT_vx[val],HVT_x_velocity_exceeds(val))
        CARS.assign_sensor(id_HVT_vx[val],True,CARS._AGENTS.keys())
        continue
    
        
    #non-zero x-velocity
    id_HVT_vx[0],cid_HVT_vx[0]=CARS.register_sensor('vx+0')
    CARS.construct_sensor(id_HVT_vx[0],HVT_x_velocity_exceeds(0))
    CARS.assign_sensor(id_HVT_vx[0],True,CARS._AGENTS.keys())
    
    #positive x-velocity
    for ind in range(ZOOM_DEPTH,ZOOM_DEPTH/2,-1): #assuming targets are not too fast...
        val=X_BOUND/pow(RESOLUTION,ind)
        id_HVT_vx[val],cid_HVT_vx[val]=CARS.register_sensor('vx'+signedstr(val))
        CARS.construct_sensor(id_HVT_vx[val],HVT_x_velocity_exceeds(val))
        CARS.assign_sensor(id_HVT_vx[val],True,CARS._AGENTS.keys())
    
    
    ## HVT y-velocity sensors
    id_HVT_vy={}
    cid_HVT_vy={}
    def HVT_y_velocity_exceeds(val):
        if val>0:
            def tmp(state):
                return state[id_viewport][0][TIME_WINDOW][HVT_ID][1].imag>val
        elif val<0:
            def tmp(state):
                return state[id_viewport][0][TIME_WINDOW][HVT_ID][1].imag<val
        else:
            def tmp(state):
                return state[id_viewport][0][TIME_WINDOW][HVT_ID][1].imag!=0
        return tmp
    
    #adding the sensors
    #negative y-velocity
    for ind in xrange(ZOOM_DEPTH/2,ZOOM_DEPTH+1): #assuming targets are not too fast...
        val=-Y_BOUND/pow(RESOLUTION,ind)
        id_HVT_vy[val],cid_HVT_vy[val]=CARS.register_sensor('vy'+signedstr(val))
        CARS.construct_sensor(id_HVT_vy[val],HVT_y_velocity_exceeds(val))
        CARS.assign_sensor(id_HVT_vy[val],True,CARS._AGENTS.keys())
    
        
    #non-zero x-velocity
    id_HVT_vy[0],cid_HVT_vy[0]=CARS.register_sensor('vy+0')
    CARS.construct_sensor(id_HVT_vy[0],HVT_y_velocity_exceeds(0))
    CARS.assign_sensor(id_HVT_vy[0],True,CARS._AGENTS.keys())
    
    #positive x-velocity
    for ind in range(ZOOM_DEPTH,ZOOM_DEPTH/2,-1): #assuming targets are not too fast...
        val=Y_BOUND/pow(RESOLUTION,ind)
        id_HVT_vy[val],cid_HVT_vy[val]=CARS.register_sensor('vy'+signedstr(val))
        CARS.construct_sensor(id_HVT_vy[val],HVT_y_velocity_exceeds(val))
        CARS.assign_sensor(id_HVT_vy[val],True,CARS._AGENTS.keys())
        continue
    
    
    ## any_target location sensors
    
    #definitions
    id_see_targs={}
    cid_see_targs={}
    def do_I_see_any_targets(rel_pos,rel_frame): #possible occlusions taken into account
        def tmp(state):
            current_ptz=deepcopy(state[id_ptz][0])
            current_ptz.zoomin(rel_pos)
            positions={state[id_viewport][0][rel_frame+TIME_WINDOW][target][0] for target in TARGET_DESCRIPTIONS}
            return any(current_ptz<=pos_to_ptz(positions[target]) and not OCC[positions[target].imag][positions[target].real] for target in TARGET_DESCRIPTIONS)
        return tmp
    
    #adding the sensors
    for key in ZOOM_KEYS:
        for rel_frame in xrange(-TIME_WINDOW,TIME_WINDOW+1):
            id_see_targs[(key,rel_frame)],cid_see_targs[(key,rel_frame)]=CARS.register_sensor('tg@'+str(key)+signedstr(rel_frame)+'f')
            CARS.construct_sensor(id_see_targs[(key,rel_frame)],do_I_see_any_targets(key,rel_frame))
            CARS.assign_sensor(id_see_targs[(key,rel_frame)],True,CARS._AGENTS.keys())
            continue
        continue
    
            
    ## target location sensors filtering by kind (which bit, what value)
    ## - bit 2 is the easiest to recognize, bit 0 is the hardest
    
    # definitions
    id_see_targs_bit={}
    cid_see_targs_bit={}
    def do_I_see_any_targets_by_bit(rel_pos,rel_frame,which_bit,bit_val):
        def tmp(state):
            relevant_targets={target for target in TARGET_DESCRIPTIONS if TARGET_DESCRIPTIONS[target] & (1<<which_bit) == (bit_val << which_bit)}
            current_ptz=deepcopy(state[id_ptz][0])
            current_ptz.zoomin(rel_pos)
            positions={target:state[id_viewport][0][rel_frame+TIME_WINDOW][target][0] for target in relevant_targets}
            return any(current_ptz<=pos_to_ptz(positions[target]) and not OCC[positions[target].imag][positions[target].real] for target in relevant_targets) and state[id_zdepth][0]>=HVT_CONFIRM_ZOOM_LEVEL-pow(KIND_BIT_RESOLUTION_STEP,which_bit+1)
        return tmp
    
    #adding the sensors
    for key in ZOOM_KEYS:
        for rel_frame in xrange(-TIME_WINDOW,TIME_WINDOW+1):
            for which_bit in xrange(3): #MAKE THE NUMBER OF BITS A SIMULATION PARAMETER
                for bit_val in xrange(2):
                    id_see_targs_bit[(key,rel_frame,which_bit,bit_val)],cid_see_targs_bit[(key,rel_frame,which_bit,bit_val)]=CARS.register_sensor('tg'+str(which_bit)+('T' if bit_val else 'F')+'@'+str(key)+signedstr(rel_frame)+'f')
                    CARS.construct_sensor(id_see_targs_bit[(key,rel_frame,which_bit,bit_val)],do_I_see_any_targets_by_bit(key,rel_frame,which_bit,bit_val))
                    CARS.assign_sensor(id_see_targs_bit[(key,rel_frame,which_bit,bit_val)],True,CARS._AGENTS.keys())
                    continue
                continue
            continue
        continue
    
    
    ###------------------------------------
    ### Motivational signals for the agents
    #
    
    # Agent's quality measure of the marking of a given frame 
    quality= lambda state,frame: pow(RESOLUTION,state[id_active_HVT_marker][0].report(frame).depth())
    
    ## ADV/REW agents attracted to unmarked "future"/"past" frames:
    ## - IDEA: Reduce distance to such frames along the frame axis
    #  -- so summation over [all] future frames of (1/quality_of_marking)
    #  -- Quality of marking of frame f = RESOLUTION^(depth of marking);
    #     this is the the [inverse square root of] the fraction of area (in
    #     the scene) occupied by the marking in that frame.
    #  - Shouldn't try to move far away from marked interval of frames
    #  -- for quick response to changing signal, pick a low value of the
    #     discount coefficient... something depending on the TIME_WINDOW
    #     parameter.
    
    def advance_signal(state):
        return sum([1./quality(state,frame) for frame in xrange(state[id_frameno][0]+1,TOTAL_FRAMES+1)])
    CARS.construct_measurable(id_sig_adv,advance_signal)
    
    def rewind_signal(state):
        return sum([1./quality(state,frame) for frame in xrange(1,state[id_frameno][0]-1)])
    CARS.construct_measurable(id_sig_rew,rewind_signal)
    
    ## Zoomout agent
    #  - Wants to maximize the number of frames in the viewport in which the target position is visible.
    
    def zoomout_signal(state):
        return sum([(state[id_is_view_marked[rel_frame]][0]) for rel_frame in xrange(-TIME_WINDOW,TIME_WINDOW+1)])
    CARS.construct_measurable(id_sig_zout,zoomout_signal)
    
    ## Zoomin agents are attracted to improving the quality of the marking
    #  in the current frame.
    
    def zoomin_signal(state):
        return quality(state,state[id_frameno][0])
    CARS.construct_measurable(id_sig_zin,zoomin_signal)


    ###------------------
    ### INITIALIZE AGENTS
    #

    for agent_id in CARS._AGENTS.keys():
        CARS._AGENTS[agent_id].init()
    
    ###----
    ### RUN
    #
    DistTLE=[]
    AreaTLE=[]
    while CARS.this_state(id_counter)<CYCLES:
        print '\t'+str(CARS.this_state(id_counter))
        #print CARS.this_state(id_counter)
        # compute and report track quality values
        if CARS.this_state(id_counter)%100==0:
            tmp_DTLE=0
            tmp_ATLE=0
            for frame,pos in HVT_ground_truth:
                tmp_mark=CARS.this_state(id_active_HVT_marker).report(frame)
                tmp_DTLE+=pow(abs(ptz_to_pos(tmp_mark)-pos),.5)*(pos_to_ptz(pos)>=tmp_mark) # average distance from center of mark to target
                tmp_ATLE+=quality(CARS.state_all(),frame)*(pos_to_ptz(pos)>=tmp_mark) # average quality of mark WRT target
            DistTLE.append(tmp_DTLE)
            AreaTLE.append(tmp_ATLE)
        #print tmp_DTLE,tmp_ATLE
        #print CARS.this_state(id_adv),CARS.this_state(cid_adv),id_adv in CARS._AGENTS, CARS.comp(cid_adv) in CARS._AGENTS, CARS.this_state(id_sig_adv)
        #print str([CARS.din(mid) for mid in CARS.this_state(id_dec)])
        #print str([CARS.din(mid) if CARS.this_state(mid) else name_comp(CARS.din(mid)) for mid in CARS._AGENTS])

        # update the state
        message=CARS.update_state()
    return DistTLE,AreaTLE
        

FILENAME='hello'
NUMBER_REPETITIONS=20
NUMBER_ITERATIONS=1000
ZOOM_DISCOUNT=1.-pow(2,-8)
FF_DISCOUNT=1.-pow(2,-8)

OUTPUT_FILE_Area=open(FILENAME+'.aout','wb')
OUTPUT_FILE_Dist=open(FILENAME+'.dout','wb')

for N in xrange(NUMBER_REPETITIONS):
    print N
    Dist_TLE,Area_TLE=hvt_tracker(FILENAME,FF_DISCOUNT,ZOOM_DISCOUNT,NUMBER_ITERATIONS)
    cPickle.dump(Dist_TLE,OUTPUT_FILE_Dist,protocol=cPickle.HIGHEST_PROTOCOL)
    cPickle.dump(Area_TLE,OUTPUT_FILE_Area,protocol=cPickle.HIGHEST_PROTOCOL)

OUTPUT_FILE_Area.close()
OUTPUT_FILE_Dist.close()
exit(0)
