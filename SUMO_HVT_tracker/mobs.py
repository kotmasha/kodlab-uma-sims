### UMA tracker agents: data structures and their operators
import cPickle
import bsddb
import numpy as np
from collections import deque
import copy


## A more standard rounding operation
def my_round(x):
    return np.floor(x) if  x<np.floor(x)+.5 else np.floor(x+1)

## version of str for signed numbers
def signedstr(x):
    return str(x) if x<0 else '+'+str(x)

## Complex integers class
#  - possibly there is something better, but I couldn't find it
#
class icomplex(object):
    def __init__(self,x,y):
        self.real=int(np.floor(x))
        self.imag=int(np.floor(y))

    def __repr__(self):
        return str(self.real)+signedstr(self.imag)+'j'

    def __str__(self):
        return str(self.real)+signedstr(self.imag)+'j'
    
    def __eq__(self,other):
        return self.real==other.real and self.imag==other.imag

    def __ne__(self,other):
        return not self==other
        
    def __add__(self,z):
        return icomplex(self.real+z.real,self.imag+z.imag)

    def __sub__(self,z):
        return icomplex(self.real-z.real,self.imag-z.imag)
        
    def __mul__(self,z):
        return icomplex(self.real*z.real-self.imag*z.imag,self.real*z.imag+self.imag*z.real)
    
    def conj(self):
        return icomplex(self.real,-self.imag)
    
    def __abs__(self):
        return (self*self.conj()).real

    def __complex__(self):
        return complex(self.real,self.imag)

    def __floordiv__(self,scale): # $scale$ must be a non-zero integer
        return icomplex(self.real / scale,self.imag / scale)

    def __mod__(self,scale): # $scale$ must be a non-zero integer
        return icomplex(self.real % scale,self.imag % scale)

    def pow(self,x):
        if isinstance(x,int):
            if x==0:
                return 1
            elif x>0:
                return self*pow(self,x-1)
            else:
                raise Exception('Complex integer has only non-negative integer powers.')
        else:
            raise Exception('Complex integer has only non-negative integer powers.')
            
    
    #def __coerce__(self,other):
    #    return complex(self.real,self.imag),complex(other)

up=icomplex(0,1)
down=icomplex(0,-1)
right=icomplex(1,0)
left=icomplex(-1,0)
nulik=icomplex(0,0)

    
## Tracker physical state (to be "pan-tilt-zoom" when this matures...)
#  - viewport is a grid of size $self._res$;
#  - viewport coords are come first ($self._state[0]$);
#  - $depth$ is the maximal zoom-in depth;
#  - viewports do not overlap except when containing each other.
#  - actions are:
#       zoomin, zoomout, pan by arbitrary vector in least-significant units

class ptz(object):
    def __init__(self,res):
        # res is a positive integer
        # init is an initial ptz value (has to already be a ptz, or absent)
        # state is a deque of complex integers in $range(self._res)$
        self._res=res
        self._state=deque([])
        
    def __repr__(self):
        return str(list(self._state))

    ## Least significant position at the tail of self._state
    def zoomin(self,pos):
        self._state.append(pos % self._res)
        return self
            
    def zoomout(self):
        if self._state:
            self._state.pop()
        return self
            
    def state_all(self):
        return deque(self._state)

    def depth(self):
        return len(self._state)
    
    ## Comparison operations ("refinement")
    #  - $other$ refines $self$ iff $self._state$ is a prefix of $other._state$
    def __le__(self,other):
        s=copy.deepcopy(self._state)
        o=copy.deepcopy(other.state_all())
        if len(s)>len(o):
            return False
        while len(s):
            if s.popleft()!=o.popleft():
                return False
        return True

    def __ge__(self,other):
        return other<=self

    def __eq__(self,other):
        return self<=other and other<=self
    
    ## Addition (more like GCD)
    def __add__(self,other):
        temp=copy.deepcopy(other)
        if self<=temp:
            return temp
        elif temp<=self:
            return self
        else:
            while not temp<=self:
                temp.zoomout()
            return temp

    ## Subtraction
    def __sub__(self,other):
        temp=copy.deepcopy(other)
        return temp.zoomout() if other<=self else self
    
    ## Panning by $panvec$, experessed in units of current level
    #  - $panvec$ is assumed to be of type $icomplex$
    #  - input rejected (returns None, state remains unchanged) if $panvec$
    #    is too long.
    #
    def pan(self,panvec):
        depth=len(self._state)
        bound=pow(self._res,depth)
        pos=panvec
        for ind in xrange(depth):
            pos+=self._state[ind]*pow(self._res,depth-1-ind)
        x=pos.real
        y=pos.imag
        if pos.real<bound and pos.imag<bound and pos.real>=0 and pos.imag>=0:
            new_state=[]
            for ind in xrange(depth):
                new_state.append(pos % self._res)
                pos //= self._res
            new_state.reverse()
            self._state=new_state
            return self
        else:
            return self

## Tracker marker (may have several): an integer-indexed database of
#    ptz objects of a fixed resolution.
#
class marker(object):
    def __init__(self,filename,default):
        self._db=bsddb.rnopen(filename,'n')
        self._default=default
        
    def mark(self,loc,content):
        if loc+1 in self._db:
            self._db[loc+1]=cPickle.dumps(cPickle.loads(self._db[loc+1])+content)
        else:
            self._db[loc+1]=cPickle.dumps(self._default+content)
        self.sync()

    def unmark(self,loc,content):
        if loc+1 in self._db:
            self._db[loc+1]=cPickle.dumps(cPickle.loads(self._db[loc+1])-content)
        else:
            self._db[loc+1]=cPickle.dumps(self._default-content)
        self.sync()
        
    def moveto(self,loc):
        if loc+1 not in self._db:
            self._db[loc+1]=cPickle.dumps(self._default)
        self._db.set_location(loc+1)
        return cPickle.loads(self._db[loc+1])

    def report(self,st,en=None):
        #returns the content from $current_loc - noffset$ to, and including
        # $current_loc+poffset$.
        rep=[]
        if en:
            for loc in xrange(st+1,en+2):
                if loc<=1 or loc not in self._db:
                    rep.append(self._default)
                else:
                    rep.append(cPickle.loads(self._db[loc]))
            return rep
        else:
            return cPickle.loads(self._db[st+1]) if st+1 in self._db else self._default 

    #def next(self):
    #    ind,pickle=self._db.next()
    #    return ind,cPickle.loads(pickle)
    #
    #def previous(self):
    #    ind,pickle=self._db.previous()
    #    return ind,cPickle.loads(pickle)

    def sync(self):
        self._db.sync()
