#from multiprocessing import Pool
from collections import deque
import numpy as np
import uuid
from wrapper import *
import time
from numpy.random import randint as rnd

N_CORES=8

###
### Randomized inequality
###

PRECISION=pow(2,-64)
# Return the token associated with the smaller value if precision threshold
# is met and report a strong inequality; otherwise return a random token and
# report a weak(=uncertain) inequality.
def rlessthan(x,y,prec=PRECISION):
      xval,xtok=x
      yval,ytok=y
      if yval-xval>abs(prec):
            return xtok,True
      elif xval-yval>abs(prec):
            return ytok,True
      else:
            return (xtok if bool(rnd(2)) else ytok),False

###
### Handling Boolean functions
###

def func_amper(experiment,mid_list):
      def f(state):
            return all([experiment._DEFS[mid](state) for mid in mid_list])
      return f

def func_not(func):
      def f(state):
            return not(func(state))
      return f

def func_delay(midlist):
      def f(state):
            return all([state[mid][1] for mid in midlist])
      return f

###
### Shortcuts to numpy Boolean Logic functions
###

def negate(x):
      return np.logical_not(x)

def conjunction(x,y):
      return np.logical_and(x,y)

def disjunction(x,y):
      return np.logical_or(x,y)

def symmetric(x,y):
      return np.logical_xor(x,y)

def alltrue(n):
      return np.array([True for x in xrange(n)])

def allfalse(n):
      return np.array([False for x in xrange(n)])

###
### Name-handling functions
###

def name_comp(name):
      ### return the name of the complementary sensor
      return name+'*' if name[-1:]!='*' else name[:-1]

def name_invert(names):
      ### return the set of complemented names in the list/set names
      return set(name_comp(name) for name in names)

def name_delay(name):
      ### delay
      return '#'+str(name)

def name_ampersand(name_list):
      ### conjunction
      L=len(name_list)
      if L==0:
            raise Exception('\nEmpty conjunction not allowed.\n')
      elif L==1:
            return name_list[0]
      else:
            return '{'+';'.join(name_list)+'}'


###
### DATA STRUCTURES
###

class Signal(object):
      def __init__(self,value):
            if len(value)%2==0:
                  self._VAL=np.array(value,dtype=bool)
            else:
                  raise Exception('Objects of class Signal must have even length -- Aborting!\n')

      def __repr__(self):
            print self._VAL

      # set the signal
      def set(self,ind,value):
            self._VAL[ind]=value

      #inspect the signal
      def value(self,ind):
            return self._VAL[ind]
      
      #report the signal
      def value_all(self):
            return self._VAL

      #extend the signal
      def extend(self,value):
            self._VAL=np.concatenate((self._VAL,value))

                  
      ### negating a partial signal
      def star(self):
            return Signal([(self._VAL[i+1] if i%2==0 else self._VAL[i-1]) for i in xrange(len(self._VAL))])
      
      ### full complement of a signal
      def negate(self):
            return Signal(negate(self._VAL))

      ### subtracting Signal "other" from Signal "self"
      def subtract(self,other):
            return Signal(conjunction(self._VAL,negate(other._VAL)))

      def add(self,other):
            return Signal(disjunction(self._VAL,other._VAL))

      def intersect(self,other):
            return Signal(conjunction(self._VAL,other._VAL))


### Data type maintaining the "environment" state and its interactions
### with the agents

class Experiment(object):
      def __init__(self):
            # dictionary of agents in this experiment, by uuid
            self._AGENTS={}

            # registering the decision observable
            id_dec=str(uuid.uuid4())
            
            # Dictionaries translating user-assigned names to
            # system-assigned uuids
            self._NAME_TO_ID={'decision':id_dec}
            self._ID_TO_NAME={id_dec:'decision'}
            self._ID_TO_DEP={id_dec:False}
            
            # List of names of the experiment measurables:
            # - ordered to accommodate dependencies during the
            #   updating process; 
            # - initialized to contain the trivial sensors.
            self._MID=[id_dec]

            # ID-based representation of the experiment state
            # - each entry is $key:deque$;
            # - the trivial measurables initialized:
            self._STATE={id_dec:deque([[]],1)}
            ### ID-based representation of the currently evolving decision
            
            ### Name-based representation $self._DEFS$ of the state update:
            ### - each entry is $key:function$;
            ### - the function accepts a dictionary of the same format as
            ###   $self._STATE$.
            def ex_decision(state):
                  return state[id_dec][0]
            self._DEFS={id_dec:ex_decision}

      ## Register $name$ in experiment and return an id for it.
      #
      ## ANY MEASURABLE HAS TO BE REGISTERED before it can be added.
      #
      ## ALL AGENTS MUST BE REGISTERED WITH $decdep=True$.
      #
      #  - returns a new universal id string unless $name$ has already
      #    been registered, in which case an exception is raised.
      #  - $decdep$ is cast to boolean, and saved for future indication
      #    of whether the measurable depends on intermediate agent decisions.
      def register(self,name,decdep=False):
            if name in self._NAME_TO_ID:
                  raise Exception('The name \"'+str(name)+'\" is already in use.')
            else:
                  new_id=str(uuid.uuid4())
                  self._NAME_TO_ID[name]=new_id
                  self._ID_TO_NAME[new_id]=name
                  self._ID_TO_DEP[new_id]=bool(decdep)
                  return new_id

      def register_sensor(self,name,decdep=False):
            mid=self.register(name,decdep)
            midc=self.register(name_comp(name),decdep)
            return mid,midc

      
      ## Return the ID of a registered name, of $None$ if not registered
      def nid(self,name):
            try:
                  return self._NAME_TO_ID[name]
            except:
                  raise Exception('Measurable name is not registered.')

      ## Return the name corresponding to a registered id else $None$
      def din(self,mid):
            try:
                  return self._ID_TO_NAME[mid]
            except:
                  raise Exception('This id is not registered.') 

      def dep(self,mid):
            try:
                  return self._ID_TO_DEP[mid]
            except:
                  raise Exception('This id is not registered.')
            
      def comp(self,mid):
            try:
                  return self.nid(name_comp(self.din(mid)))
            except:
                  raise Exception('Attempted complementing an id not belonging to a registered sensor.')
      
      ## Front-end query (name-based) of an experiment state variable
      def this_state(self,mid,delta=0):
            try:
                  return self._STATE[mid][delta]
            except:
                  pass
                  
      ## Internal query of an experiment state variable
      def state_all(self):
            return self._STATE
 
      ## Set new state value
      def set_state(self,mid,value):
            self._STATE[mid].appendleft(value)
            return None


      ## Extend the experiment state by adding a measurable with id $mid$
      #  - $mid$ must be registered ahead of time
      #  - measurables hold current and preceding state by default
      #  - initialize with $init$: an iterable of values (out of which
      #    the last $depth+1$ will be used as initial values)
      #  - $definition$ should be a function of state
      def construct_measurable(self,mid,definition=None,init_value=None,depth=1):
            self._MID.append(mid)
            self._DEFS[mid]=definition
            if init_value==None:
                  self._STATE[mid]=deque([],depth+1)
                  try:
                        self.set_state(mid,definition(self._STATE))
                  except:
                        pass
            else:
                  self._STATE[mid]=deque(init_value,depth+1)
            return None
      
      ## Construct a binary sensor pair with ids supplied by a
      ##   preceding Experiment.register_sensor(name) command.
      #  - INPUT: $mid$, $definition$ (default=None), $depth$ (default=1)
      def construct_sensor(self,mid,definition=None,init_value=None,depth=1):
            #verify proper registration of sensor
            midc=self.comp(mid)
            #compute initial value of sensor
            if definition==None: # this is an action sensor, init value WILL be defaulted to None...
                  self.construct_measurable(mid,None,[False for ind in xrange(depth+1)],depth)
                  self.construct_measurable(midc,None,[True for ind in xrange(depth+1)],depth)
            else:
                  self.construct_measurable(mid,definition,init_value,depth)
                  self.construct_measurable(midc,func_not(definition),negate(init_value) if init_value else None,depth)
            return None


      # Add the indicated sensor to the listed agents (by mid)
      def assign_sensor(self,mid,is_initial,agent_id_list):
            for id_agent in agent_id_list:
                  try:
                        self._AGENTS[id_agent].add_sensor(mid,is_initial)
                  except:
                        raise Exception('Attempted sensor assignment to unregistered agent.')
            return None

      ## Decision phase
      #  - $instruction$ is a complete selection of action sensor names
      #    for 'command' mode; empty/disregarded in 'decide' mode

      ### Attach an agent to the experiment
      ### - $params$ is a tuple of initialization parameters for the agent
      ### - $name$ measurable is constructed in the experiment
      def construct_agent(self, id_agent, id_motivation, definition, params, using_log):
            #verify validity of measurable identifier $mid$
            try:
                  id_agentc=self.comp(id_agent)
            except:
                  raise Exception('New agent id has not been registered.')
            # construct new sensor and agent and append to agents list
            self.construct_sensor(id_agent,definition,[False,False])
            new_agent = Agent(self, id_agent, id_motivation, params, using_log)
            self._AGENTS[id_agent]=new_agent

            # construct the agent's snapshots
            new_agent.add_snapshot('plus', params, False, using_log)
            new_agent.add_snapshot('minus', params, True, using_log)
            # return a pointer to the agent
            return new_agent


      ## Update the experiment state and output a summary of decisions
      def update_state(self):
            id_dec=self.nid('decision')
            #reset the decision state variable
            self.set_state(id_dec,[])
            #prepare empty dictionary for agent messages
            messages={}
            
            ## Iterate over agents and decision-dependent measurables
            # - if id of an agent, then form a decision and append to id_dec
            # - if id of a measurable, update its value
            for mid in [tmp_id for tmp_id in self._MID if self.dep(tmp_id)]:
                  try:
                        midc=self.comp(mid)
                        agentQ=mid in self._AGENTS or midc in self._AGENTS
                  except:
                        agentQ=False
                        
                  if agentQ:
                        if mid in self._AGENTS: # if mid is an agent...
                              ## agent activity set to current reading
                              agent=self._AGENTS[mid]
                              agent.active=self.this_state(mid)

                              # agent analyzes outcomes of preceding decision
                              # cycle if the prediction is too broad, add a
                              # sensor
                              if sum(agent.report_current().subtract(agent.report_predicted()).value_all()):
                                    agent.delay([(agent.report_last().intersect(agent.report_target())).intersect(agent._INITMASK)])
                              #      agent.delay([agent.report_last().intersect(agent._INITMASK)])

                              ## agent makes a decision 
                              messages[mid]=agent.decide()
                              #print str([self.din(tmp) for tmp in self.this_state(id_dec)])
                        else: # if midc is an agent, decision already reached
                              pass
                  else: # neither mid nor midc is an agent, perform update
                        try: # attempt update using definition
                              self.set_state(mid,self._DEFS[mid](self._STATE))
                        except: # if no definition available, do nothing; this is a state variable evolving independently of the agent's actions, e.g., a pointer to a data structure.
                              pass

            # At this point, there is a complete decision vector
            action_signal=self.this_state(id_dec)
            for mid in self._MID:
                  try:
                        agentQ=mid in self._AGENTS or self.comp(mid) in self._AGENTS
                  except:
                        agentQ=False

                  depQ=self.dep(mid)
                  
                  if agentQ: # if mid is an agent (or its star)...
                        try: # try evaluating the definition for the agent
                              self.set_state(mid,(self._DEFS[mid](self._STATE)))
                              if mid in self._AGENTS and self.this_state(mid)!=(mid in action_signal): # if initial decision was changed
                                    messages[mid]+=', override by other (was '+(self.din(mid) if mid in action_signal else name_comp(self.din(mid)))+')'
                        except: # if no definition, set the value according to $action_signal$
                              self.set_state(mid,(mid in action_signal))
                  else:
                        try: # try updating using definition
                              self.set_state(mid,(self._DEFS[mid](self._STATE)))
                        except: # if no definition available then do nothing; this is a state variable evolving independently of the agent's actions, e.g., a pointer to a data structure.
                              #raise Exception('\n\n  AHA!!!\t'+self.din(mid))
                              pass

                        
            #aggregate and output the messages
            message_all=""
            ordered_agents=[mid for mid in self._MID if mid in self._AGENTS]
            for mid in ordered_agents:
                  name=self.din(mid)
                  outp='\t'+(name if self.this_state(mid) else name_comp(name))+' : '+messages[mid]+'\n'
                  message_all+=outp

            return message_all
      


class Agent(object):
      ### initialize an "empty" agent with prescribed learning parameters
      def __init__(self,experiment,id_agent,id_motivation,params,using_log=False):
            # a string naming the agent/action
            self._MID=id_agent
            id_agentc=experiment.comp(id_agent)
            self._MOTIVATION=id_motivation
            # the experiment serving as the agent's environment
            self._EXPERIMENT=experiment
            # the agent's parameters
            self._PARAMS=params
            # snapshot size is always even
            self._SIZE=0
            # ordered list of the Boolean measurables used by the agent:
            self._SENSORS=[]

            ## Boolean vectors ordered according to self._SENSORS

            # raw observation:
            self._OBSERVE=Signal(np.array([],dtype=np.bool))
            # previous state representation:
            self._LAST=Signal(np.array([],dtype=np.bool))
            # current state representation:
            self._CURRENT={
                  'plus':Signal(np.array([],dtype=np.bool)),
                  'minus':Signal(np.array([],dtype=np.bool))
            }
            # predicted state representation:
            self._PREDICTED={
                  'plus':Signal(np.array([],dtype=np.bool)),
                  'minus':Signal(np.array([],dtype=np.bool))
            }

            # target representation:
            self._TARGET={
                  'plus':Signal(np.array([],dtype=np.bool)),
                  'minus':Signal(np.array([],dtype=np.bool))
            }
            self._INITMASK=Signal(np.array([]))
            
            ## Calling the wrapper in charge of communication with CUDA side
            self.brain = brain(self)
            self.active = False

      ## Activate the agent when the initial definitions stage is completed
      def init(self):
            self.collect_observation()
            return self.brain.init()

      def collect_observation(self):
            self._OBSERVE=Signal([self._EXPERIMENT.this_state(mid) for mid in self._SENSORS])
            return None

      ## Form a snapshot structure
      def add_snapshot(self, mid, params, state, using_log=False):
            self.brain.add_snapshot(mid, params, state, using_log)
                              
      ### Adding a sensor to the agent
      ### - intended ONLY to be called by $experiment.new_sensor$
      ### - assumes the measurable $name$ and its complement are present
      ###   in the parent experiment
      def add_sensor(self,mid,is_initial):
            #verify proper registration of $mid$
            try:
                  midc=self._EXPERIMENT.comp(mid)
            except:
                  raise Exception('Sensor id not properly registered.')
            ## Extending the sensor lists
            if mid not in self._SENSORS:
                  self._SIZE+=2
                  self._SENSORS.extend([mid,midc])
                  ## update observed signal
                  self._OBSERVE.extend(np.array([self._EXPERIMENT.this_state(mid),self._EXPERIMENT.this_state(midc)],dtype=bool))
                  ## expand maintained signals
                  for token in ['plus','minus']:
                        self._CURRENT[token].extend(np.array([False,False],dtype=bool))
                        self._TARGET[token].extend(np.array([False,False],dtype=bool))
                        self._PREDICTED[token].extend(np.array([False,False],dtype=bool))
                        
                  self._LAST.extend(np.array([False,False],dtype=bool))
                  self._INITMASK.extend(np.array([is_initial,is_initial],dtype=bool))


      ## Report the last state
      def report_last(self):
            return self._LAST
      
      ## Report the current state for snapshot $token$, else for
      #  the active snapshot
      def report_current(self,token=None):
            try:
                  return self._CURRENT[token]
            except:
                  return self._CURRENT['plus' if self.active else 'minus']

      ## Report the target state for snapshot $token$, else for
      #  the active snapshot
      def report_target(self,token=None):
            try:
                  return self._TARGET[token]
            except:
                  return self._TARGET['plus' if self.active else 'minus']

      ## Report the current state for snapshot $token$, else for
      #  the active snapshot
      def report_predicted(self,token=None):
            try:
                  return self._PREDICTED[token]
            except:
                  return self._PREDICTED['plus' if self.active else 'minus']

      def decide(self):
            dist={}
            id_agent=self._MID
            id_agentc=self._EXPERIMENT.comp(id_agent)
            id_dec=self._EXPERIMENT.nid('decision')

            # mapping agent activity
            activity={'plus':self.active,'minus':not(self.active)}
            
            # acquire a new observation
            self.collect_observation()

            # move the latest record of the current state to "last state"
            self._LAST=self.report_current()

            #update the agent's snapshots
            for token in ['plus','minus']:
                  dist[token]=self.brain._snapshots[token].decide(self._OBSERVE._VAL.tolist(),self._EXPERIMENT.this_state(self._MOTIVATION),activity[token])
                  self._TARGET[token]=Signal(self.brain._snapshots[token].getTarget())
                  self._CURRENT[token]=Signal(self.brain._snapshots[token].getCurrent())
                  self._PREDICTED[token]=Signal(self.brain._snapshots[token].getPrediction())

            # make a decision
            token,quality=rlessthan((dist['plus'],'plus'),(dist['minus'],'minus'))

            # Update the decision vector:
            # - only the decision vector gets updated -- not the activity
            #   status of the agent -- because the latter might be altered
            #   by other agents in the process of updating the experiment
            self._EXPERIMENT.this_state(id_dec).append(id_agent if token=='plus' else id_agentc)
            
            # return comment
            return 'deliberate' if quality else 'random'

            
            
      ## PICK OUT SENSORS CORRESPONDING TO A SIGNAL
      #  - returns a list of mids corresponding to the signal  
      def select(self,signal):
            return [mid for indicator,mid in zip(signal.value_all().tolist(),self._SENSORS) if indicator]

      def generate_signal(self,mid_list):
            return Signal([(tmp_mid in mid_list) for tmp_mid in self._SENSORS])
      
      ### FORM NEW CONJUNCTIONS
      ### - $signals$ is a list of signals, each describing a conjunction
      ###   that needs to be added to the snapshot
      def amper(self,signals):
            new_signals=[]
            for signal in signals:
                  #if signal is trivial (1 or no sensors), skip it
                  if sum(signal.value_all()) < 2: 
                        continue
                  #transform signal into a list of mids
                  mid_list=self.select(signal)
                  #new name as a means of verifying redundancy (CHANGE THIS!!)
                  new_name=name_ampersand([self._EXPERIMENT.din(mid) for mid in mid_list])
                  #construct definition for new sensor
                  new_def=func_amper(self._EXPERIMENT,mid_list)
                  #determine dependency on initial decisions
                  new_dep=any([self._EXPERIMENT._ID_TO_DEP[mid] for mid in mid_list])
                  #register/construct/assign new sensor to self
                  try: #in case the sensor is not even registered...
                        #register the new sensor
                        new_mid,new_midc=self._EXPERIMENT.register_sensor(new_name,new_dep)
                        #construct the new sensor
                        self._EXPERIMENT.construct_sensor(new_mid,new_def)
                        #add the new sensor to this agent as non-initial
                        self.add_sensor(new_mid,False)
                        new_signals.append(signal)
                  except: #if the sensor was registered
                        new_mid=self._EXPERIMENT.nid(new_name)
                        if new_mid in self._SENSORS:
                              pass
                        else:
                              self.add_sensor(new_mid,False)
                              new_signals.append(signal)
            if new_signals:
                  self.brain.amper(new_signals)
            else:
                  pass
            return None
     
      ### Form a delayed sensor
      ### - $signals$ is a list of signals, each describing a delayed 
      ###   conjunction which must be added to the agent's snapshots
      def delay(self,signals):
            new_signals=[]
            for signal in signals:
                  #if signal is trivial (no sensors), skip it
                  if sum(signal.value_all()) < 1: 
                        continue
                  #transform signal into a list of mids
                  mid_list=self.select(signal)
                  #new name as a means of verifying redundancy (CHANGE THIS!!)
                  new_name=name_delay(name_ampersand([self._EXPERIMENT.din(mid) for mid in mid_list]))
                  #construct definition for new sensor
                  #def new_def(state):
                  #      return all([state[mid][1] for mid in mid_list])
                  new_def=func_delay(mid_list)
                  #determine dependency on initial decisions
                  new_dep=any([self._EXPERIMENT._ID_TO_DEP[mid] for mid in mid_list])
                  #register/construct/assign new sensor to self
                  try: #in case the sensor is not even registered...
                        #register the new sensor
                        new_mid,new_midc=self._EXPERIMENT.register_sensor(new_name,new_dep)
                        #construct the new sensor
                        self._EXPERIMENT.construct_sensor(new_mid,new_def)
                        #add the new sensor to this agent as non-initial
                        self.add_sensor(new_mid,False)
                        new_signals.append(signal)
                  except: #if the sensor was registered
                        new_mid=self._EXPERIMENT.nid(new_name)
                        if new_mid in self._SENSORS:
                              pass
                        else:
                              self.add_sensor(new_mid,False)
                              new_signals.append(signal)
            if new_signals:
                  self.brain.delay(new_signals)
            else:
                  pass
            return None

