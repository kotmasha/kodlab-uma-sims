#cuda support
import pycuda.driver as cuda
import pycuda.tools
import pycuda.autoinit
import pycuda.gpuarray as gpuarray
from pycuda.compiler import SourceModule

BLOCK_SIZE=16

#std python
import sys
import cmath
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.transforms as mtr
import matplotlib as mpl
import cPickle

from numpy.random import RandomState as RS

from copy import deepcopy as cp
from collections import deque

## A more standard rounding operation
def my_round(x):
    return np.floor(x) if  x<np.floor(x)+.5 else np.floor(x+1)

## version of str for signed numbers
def signedstr(x):
        return str(x) if x<0 else '+'+str(x)

#Global Variables
Origin=complex(0,0)

#
# Auxiliary graphics wrappers: Rectangle, Circle and Arrow
# - the purpose is for these artists to be easily movable.
# - inputs (base,diag,center,vec) are assumed to be of type <complex>

class myRectangle(patches.Rectangle):
    def __init__(self,base,diag,color=(0.,0.,0.),alpha=1.):
        patches.Rectangle.__init__(self,xy=(base.real,base.imag),width=diag.real,height=diag.imag,color=color,alpha=alpha)
    
    def move(self,base,diag=None):
        # if diag is not provided, merely move the basepoint of the rectangle
        if not diag is None:
            self.set_height(diag.imag)
            self.set_width(diag.real)
        self.set_xy(base.real,base.imag)
        self._stale=True

class myCircle(patches.Circle):
    def __init__(self,center,radius,color=(0.,0.,0.),alpha=1.):
        patches.Circle.__init__(self,xy=(center.real,center.imag),radius=radius,color=color,alpha=alpha)

    def move(self,center,radius=None):
        # if radius is not provided, move the circle without changing the radius
        if not radius is None:
            self.set_radius(radius)
        self.center=(center.real,center.imag)
        self._stale=True

class myArrow(patches.Arrow):
    def __init__(self,base,vec,color=(0.,0.,0.),alpha=1.):
        patches.Arrow.__init__(self,x=base.real,y=base.imag,dx=vec.real,dy=vec.imag,color=color,alpha=alpha,width=.3)

    def move(self,base,vec):
        L=float(abs(vec))
        if L!=0:
            cv=float(vec.real)/L
            sv=float(vec.imag)/L
        else:
            cv=0
            sv=1
        tr1=mtr.Affine2D().scale(L,1)
        tr2=mtr.Affine2D.from_values(cv,sv,-sv,cv,0.,0.)
        tr3=mtr.Affine2D().translate(base.real,base.imag)
        tr=tr1+tr2+tr3
        self._patch_transform=tr.frozen()
        self._stale=True


#
### BASE CLASS FOR OBJECTS PLACEABLE IN THE ARENA
#

class obj(object):
    def __init__(self,ar,typ,tag,pos):
        if isinstance(typ,str) and isinstance(tag,str):
            self._ar=ar
            self._type=typ
            self._tag=tag
        else:
            raise Exception('Object type and tag must be strings.\n\n')
        if isinstance(pos,complex):
            self._pos=pos
        else:
            raise Exception('Object position must be of type complex')
        self._attr={}
        self._repn_tags=[]
        self._repns={}

    # Generic update method
    def update(self,control):
        # update the object state (position,attributes):
        self.update_state(control)
        # update the object's representations:
        for tag in self._repn_tags:
            self._repns[tag].update()

    # Generic state update method (do nothing)
    def update_state(self,control):
        pass

    def getPos(self):
        return self._pos

    # Handle attributes
    def getAttr(self,tag):
        return self._attr[tag]

    def setAttr(self,tag,val):
        self._attr[tag]=val

    # Handle Repns
    def addRepn(self,tag,repn): # generic add
        self._repn_tags.append(tag)
        self._repns[tag]=repn

    def getRepnTags(self):
        return self._repn_tags

    def getRepn(self,tag):
        return self._repns[tag]

    def removeRepn(self,tag): # generic remove of representation
        self._repns[tag].remove() # have the tagged representation remove its artists from dependent objects
        self._repn_tags.remove(tag) # remove the tag from the list of representation tags
        del self._repns[tag] # remove the tagged representation from parent object

    # Move self to specified/random position in container arena
    def teleport(self,posn=None):
        if posn is None: #teleport to a random position in the environment...
            self._pos=self._ar.rndPos()
        elif self._ar.inBounds(posn): #...unless a legitimate target position was specified
            self._pos=posn
        else:
            raise Exception('\nObject cannot be teleported out of bounds. ---ABORTING\n')
            
    # Remove self from container arena
    def remove(self):
        # remove all representations:
        for tag in self._repn_tags:
            self.removeRepn(tag)
        # remove tag from arena:
        self._ar._types_to_tags[self._type].remove(self._tag)
        # remove self from arena:
        del self._ar._objects[self._tag]

#
### [VISUAL] REPRESENTATION BASE CLASS
#

class Repn_base():
    def __init__(self,parent):
        self._parent=parent # parent object (mouse, cheese, arena...)
        self._content_tags=[]
        self._content={} # content is a dictionary of <tag:mpl_artist> pairs
        self._attr={} # attributes of the representation

    def update(self):
        pass

    def getContentTags(self):
        return self._content_tags

    def getContent(self,tag):
        return self._content[tag]

    def addContent(self,tag,artist):
        self._content_tags.append(tag)
        self._content[tag]=artist

    def getAttr(self,tag):
        return self._attr[tag]

    def setAttr(self,tag,val):
        self._attr[tag]=val

    def remove(self): #remove all artists from dependent axes
        for tag in self._content_tags:
            self._content[tag].remove()
            del self._content[tag]


#
### ARENA BASE CLASS
#

class Arena_base():
    ### ARENA BASE CLASS CONSTRUCTOR:

    def __init__(self,xbounds,ybounds,out_of_bounds=lambda x: False,random_state=RS()):
        # Physical Bounds
        # - rectangle to contain arena specified by xbounds=(xmin,xmax) and ybounds=(ymin,ymax)
        self._xbounds=xbounds
        self._ybounds=ybounds
        # - a function accepting an complex position and returning Boolean, to represent obstacles in the arena
        self._oob=out_of_bounds

        # Types and objects lists; misc information
        self._types=[] #a list of the possible object types, in the order in which they should be updated (e.g. mice before cheeses)
        self._types_to_tags={} #dict of type:tag_list pairs to help organize the update
        self._objects={} #dict of tag:object pairs occupying the arena
        self._misc={} #miscellaneous data
        
        #RandomState object for reproducibility purposes
        self._rnd=random_state
        self._rnd_init_state=cPickle.dumps(self._rnd.get_state())

        #Visualizations etc.
        self._repn_tags=[]
        self._repns={}

    ### ARENA BASE CLASS METHODS:

    # Get misc values
    def getMiscVal(self,tag):
        val,_,_=self._misc[tag]
        return val

    def getMiscDiff(self,tag):
        _,_,diff=self._misc[tag]
        return diff

    def getMiscUpd(self,tag):
        _,upd,_=self._misc[tag]
        return upd

    # Set the misc values to desired content
    # - upd is an optional function of <self>, used to update <self._misc[tag]>
    # - diff is an optional value which is always False if upd is trivial; it is not intended for manual setting, but, rather, as an indicator of change in the value of the state variable
    def setMisc(self,tag,content,upd=None,diff=False):
        self._misc[tag]=(content,upd,diff)

    # Set randomization in arena:
    def seed(self,random_state):
        self._rnd.set_state(random_state)
    
    # Base usage of the out-of-bounds function
    def inBounds(self, pos):
        if pos.real<self._xbounds[0] or pos.real>self._xbounds[1] or pos.imag<self._ybounds[0] or pos.imag>self._ybounds[1]:
            return False
        else:
            return not self._oob(pos)

    # Generate a random position in the arena (within bounds)
    def rndPos(self):
        rndx=self._xbounds[0]+(self._xbounds[1]-self._xbounds[0])*self._rnd.rand()
        rndy=self._ybounds[0]+(self._ybounds[1]-self._ybounds[0])*self._rnd.rand()
        pt=complex(rndx,rndy)
        if self._oob(pt):
            return self.rndPos()
        else:
            return pt

    ## Arena update
    def update(self,control):
        #update all objects
        self.updateObjs(control)
        #update all representations [representations tied to objects have ALREADY been updated!]
        for tag in self._repn_tags:
            self._repns[tag].update()
        #update misc state variables
        for key in self._misc:
            val,upd,_=self._misc[key]
            try:
                newval=upd(self)
            except:
                newval=val
            self.setMisc(key,newval,upd,val==newval)
        
    ## Handling objects and types

    # Update all objects given a control input to the arena
    def updateObjs(self,control):
        for typ in self._types:
            for tag in self._types_to_tags[typ]:
                self._objects[tag].update(control)
    
    # Check whether or not a type is present
    def isType(self,typ):
        return isinstance(typ,str) and typ in self._types

    # Obtain a specific object
    def getObj(self,tag):
        try:
            return self._objects[tag]
        except KeyError:
            raise Exception('Illegal object tag \"'+str(tag)+'\" requested.\n')

    # Obtain all objects of a given type    
    def getObjs_ofType(self,typ):
        if self.isType(typ):
            return self._types_to_tags[typ]
        else:
            raise Exception('Illegal type identifier (\"'+str(typ)+'\") requested.\n')

    # Obtain all types
    def getTypes(self):
        return self._types

    # Add a registered type
    def addType(self,typ):
        if isinstance(typ,str) and not self.isType(typ): #if typ is a string which is not a registered type...
            self._types.append(typ) #...register typ as a type
            self._types_to_tags[typ]=[]
        else: #otherwise, raise an exception.
            raise Exception('Illegal new type identifier, or type already there.\n')

    # Add an object of a given type:
    def addObj(self,ob):
        typ=ob._type
        tag=ob._tag
        if not self.isType(typ): #if ob is of unregistered type...
            self.addType(typ) #...register the type, then proceed
        if not tag in self._objects:
            if self.inBounds(ob._pos): # if object is in bounds, add it:
                self._types_to_tags[typ].append(ob._tag) #...add ob._tag to the update list
                self._objects[tag]=ob #...add ob itself to the objects dictionary
            else: # if object is out of bounds, discard it without comment:
                pass
        else:
            raise Exception('Adding an object with an existing tag is not allowed.\n')


    ## Handling attribute computation
    
    # Individual value of <attrName> at position <pos> from the object tagged <tag>
    def getAttr(self,tag,attrName):
        return self._objects[tag].getAttr(attrName)
    
    ## HANDLING [VISUAL] REPNS

    def addRepn(self,tag,repn):
        self._repn_tags.append(tag)
        self._repns[tag]=repn

    def removeRepn(self,tag):
        self._repns[tag].remove()
        del self._repns[tag]
        self._repn_tags.remove(tag)

    def getRepn(self,tag):
        return self._repns[tag]


#
### MICE AND CHEESES
#

#cheese class:
class Cheese(obj):
    def __init__(self,arena,tag,pos,nibbles,horizon):
        obj.__init__(self,arena,'cheese',tag,pos)
        self._attr['horizon']=float(horizon)
        self._attr['nibbles']=nibbles
        self._attr['counter']=0
        #diameter of arena
        xb0,xb1=arena._xbounds
        yb0,yb1=arena._ybounds
        diam=abs(complex(xb1-xb0,yb1-yb0))
        self._attr['scent']=lambda posn: max(0,self._attr['horizon']-abs(posn-self._pos))
        self._attr['scentGradient']=lambda posn: Origin if (posn==self._pos or abs(posn-self._pos)>self._attr['horizon']) else -(posn-self._pos)/abs(posn-self._pos)
        #self._attr['scent']=lambda posn: np.exp(-(abs(posn-self._pos)/5.)**2)
        #self._attr['scentGradient']=lambda posn: -(2./25.)*np.exp(-(abs(posn-self._pos)/5)**2)*(posn-self._pos)

    def update_state(self,command):
        nibbleP=0
        for tag in self._ar._types_to_tags['mouse']: #going over all mice...
            if abs(self._ar._objects[tag]._pos-self._pos)<1: #...if a mouse is close enough (less than a unit distance) to this cheese,
                nibbleP+=1 #...the mouse nibbles on it.
        if nibbleP: #if any mice nibbled on this cheese...
            self._attr['counter']+=nibbleP #...increase the nibble counter accordingly
        else: #otherwise
            self._attr['counter']=0 #reset the nibble counter
        if self._attr['counter']>=self._attr['nibbles']: #if too many nibbles...
            self.remove() #...remove the cheese.

#representation class for cheeses
class Repn_cheese(Repn_base):
    def __init__(self,ch):
        Repn_base.__init__(self,ch)
        # a circle patch to represent the cheese's position
        self.addContent('marker',myCircle(ch._pos,radius=0.25,color=(0.,0.,1.)))

#mouse class:
class Mouse(obj):
    def __init__(self,arena,tag,horizon,order,pos,pose=complex(0,0)):
        obj.__init__(self,arena,'mouse',tag,pos)
        ## Initialization attributes:
        #  - 'order':   an integer N representing the order of the mouse's turning action
        #  - 'horizon': view distance for the mouse
        self._attr['horizon']=float(horizon)
        self._attr['order']=int(order)

        ## Mouse attributes:
        #  - [left] turn multiplier:
        self._attr['turn']=lambda k: cmath.rect(1,2*k*np.pi/order)
        #  - starting pose vector:
        if pose==complex(0,0):
            self._attr['pose']=cmath.rect(1,2*np.pi*self._ar._rnd.randint(order)/order)
        else:
            self._attr['pose']=pose/abs(pose)
        #  - function computing scent at a position (mouse frame)
        self._attr['scent']=lambda relpos: sum([self._ar.getAttr(tag,'scent')(self.worldPos(relpos)) for tag in self._ar._types_to_tags['cheese']])
        #  - function computing scent gradient at a position (mouse frame)
        self._attr['scentGradient']=lambda relpos: sum([self._ar.getAttr(tag,'scentGradient')(self.worldPos(relpos)) for tag in self._ar._types_to_tags['cheese']])

    ## Mouse motion

    def update_state(self,control):
        mode,command=control
        if mode=='move':
            self.move(command)
        elif mode=='teleport':
            self.teleport(command)
        else:
            raise Exception('\nInvalid command issued to Mouse '+str(self._tag)+' --- Aborting.\n')

    def move(self,command):
        # assume command is a Boolean tuple of the form:
        fd,bk,lt,rt,arb=command
        response_step=False
        response_turn=False
        if arb:
            response_step=self.step(fd,bk)
        else:
            response_turn=self.turn(lt,rt)
        #return response_step and response_turn # report whether or not the move suceeded

    def step(self,fwd,bck):
        # assume fwd and bck are boolean
        newpos=self._pos+self._attr['pose']*((1 if fwd else 0)+(-1 if bck else 0))
        if self._ar.inBounds(newpos):
            self._pos=newpos
            return True # move succeeded
        else:
            return False # move did not succeed

    def turn(self,lt,rt):
        # assume lt and rt are boolean
        self._attr['pose']=self._attr['pose']*(self._attr['turn'](1) if lt else 1)*(self._attr['turn'](-1) if rt else 1)
        return True # turn move always succeeds

    def teleport(self,command=None):
        if command is None: # teleport randomly
            self._pos=self._ar.rndPos() # new position is drawn from arena
            self._attr['pose']=cmath.rect(1,2*np.pi*self._ar._rnd.rand()) #new pose vector is a random unit vector
        else:
            posn,pose=command
            if isinstance(posn,complex) and self._ar.inBounds(posn): 
                self._pos=posn
            else:
                raise Exception('\nMouse cannot be teleported out of bounds. ---ABORTING\n')
            if isinstance(pose,complex) and pose!=complex(0,0):
                self._attr['pose']=pose/abs(pose)
            else:
                raise Exception('\nSpecified mouse pose must be a non-zero complex number. ---ABORTING\n')

    ## World frame to mouse frame and back:

    def worldPos(self,relpos):
        # switch from agent-centric coordinates to world coordinates
        return self._pos-complex(0,1)*self._attr['pose']*relpos

    def relPos(self,pos):
        # switch from global coordinates to local coordinates relative to mouse position and pose
        # pos is assumed to be of type icomplex.
        return (pos-self._pos)*complex(0,1)*self._attr['pose'].conjugate()

    def mdist(self):
        return min([abs(self._pos-self._ar.getObj(tag)._pos) for tag in self._ar._types_to_tags['cheese']])

    ## Mouse measurements
    def scent_gradient(self):
        return self._attr['scentGradient'](Origin)

    def scent(self): #elevation of mouse
        return self._attr['scent'](Origin)

    def cos_grad(self,k): # cosine of landscape gradient to the relative pose times exp(2\pi i k):
        cos=lambda v: 0 if abs(v)==0 else (v*(self._attr['pose']*self._attr['turn'](k)).conjugate()).real/abs(v)
        return cos(self.scent_gradient())

#representation class for mouse


#representation
class Repn_mouse(Repn_base):
    def __init__(self,mouse):
        Repn_base.__init__(self,mouse)
        # add a patch to represent the mouse
        posn=mouse._pos
        pose=mouse.getAttr('pose')
        grad=mouse.scent_gradient()
        grad=grad/abs(grad) if grad!=0 else complex(0,0)
        rad=.5 # marker thickness
        self.addContent('marker',myCircle(posn,rad))
        # add an arrow representing the mouse pose
        self.addContent('pose_vector',myArrow(posn,pose,color=(0.,0.,1.)))
        # add an arrow representing the scent gradient
        self.addContent('grad_vector',myArrow(posn,grad,color=(1.,0.,0.)))

    def update(self):
        posn=self._parent._pos
        pose=self._parent.getAttr('pose')
        grad=self._parent.scent_gradient()
        # update position patch
        self.getContent('marker').move(posn)
        # update pose arrow
        self.getContent('pose_vector').move(posn,pose)
        #update grad arrow
        self.getContent('grad_vector').move(posn,grad)


class Arena_wmouse(Arena_base):
    ## Forming the arena
    def __init__(
        self,
        xbounds,
        ybounds,
        mouse_params={'horizon':5,'order':4},
        mouse_init={},
        cheese_params={'maxCheeses':20,'Ncheeses':20,'nibbles':5},
        cheese_list=[],
        visualQ=True,
        out_of_bounds=lambda x: False,
        random_state=RS(),
        ):

        #initialize
        Arena_base.__init__(self,xbounds,ybounds,out_of_bounds,random_state)
        
        #register the types (note that cheeses will be updated before mice)
        self.addType('cheese')
        self.addType('mouse')

        #add the mouse...
        if mouse_init=={}: #... at a random position with random pose
            self.addObj(Mouse(
                self,
                tag='mus',
                pos=self.rndPos(),
                horizon=mouse_params['horizon'],
                order=mouse_params['order']
                ))
        else: #... at a prescribed position & pose
            self.addObj(Mouse(
                self,
                tag='mus',
                pos=mouse_init['pos'],
                pose=mouse_init['pose'],
                horizon=mouse_params['horizon'],
                order=mouse_params['order']
                ))
        #add states
        self.setMisc('counter',0,lambda ar: ar.getMiscVal('counter')+1) # time counter
        self.setMisc('maxCheeses',cheese_params['maxCheeses']) # maximum number of cheeses
        self.setMisc('Ncheeses',cheese_params['Ncheeses'],lambda ar: len(ar.getObjs_ofType('cheese'))) # cheese counter
        self.setMisc('diam',abs(complex(xbounds[1]-xbounds[0],ybounds[1]-ybounds[0])))
        self.setMisc('horizon',mouse_params['horizon'])

        #add the requested number of random cheeses
        if cheese_params['Ncheeses']<=cheese_params['maxCheeses'] and len(cheese_list)<=cheese_params['maxCheeses']:
            if cheese_list==[]:
                for ind in xrange(cheese_params['Ncheeses']):
                    self.addObj(Cheese(self,'ch'+str(ind+1),self.rndPos(),cheese_params['nibbles'],mouse_params['horizon']))
            else:
                for ind,pos in enumerate(cheese_list):
                    self.addObj(Cheese(self,'ch'+str(ind+1),pos,cheese_params['nibbles'],mouse_params['horizon']))
        else:
            raise Exception('\nNumber of requested cheeses cannot exceed maximum. --- Aborting\n')

        #add global map representation:
        if visualQ:
            self.addRepn('global_view',Repn_global_arena_wmouse(self))

    def getCheeseList(self):
        return self.getObjs_ofType('cheese')

    def getCheese(self,tag):
        ob=self.getObj(tag)
        if ob._type=='cheese':
            return ob
        else:
            raise Exception('Object \"'+str(tag)+'\" is not a cheese.\n')

    def counter(self):
        return self.getMiscVal('counter')

    def cheeseDiff(self):
        return self.getMiscDiff('Ncheeses')

class Repn_global_arena_wmouse(Repn_base):
    def __init__(self,arena,res=10): #res is the magnification/resolution factor of the visual representation
        if isinstance(arena,Arena_wmouse):
            Repn_base.__init__(self,arena)
            self.res=np.int32(res)
            # Create a figure and axis for global map of arena
            fig,ax=plt.subplots(1)
            # Construct the elements of the figure:
            #  - caption etc.
            fig.suptitle('Single mouse experiment, cycle No. '+str(arena.getMiscVal('counter')),fontsize=16)

            #  - prepare image parameters
            self.xb0,self.xb1=np.array(arena._xbounds,dtype=np.int32)
            self.yb0,self.yb1=np.array(arena._ybounds,dtype=np.int32)
            self.xpix=1+self.res*np.abs(self.xb1-self.xb0) # number of pixels in the x direction
            self.ypix=1+self.res*np.abs(self.yb1-self.yb0) # number of pixels in the y direction
            self.dx=np.float32(1./self.res) # x-spacing between contiguous pixel centers
            self.dy=np.float32(1./self.res) # y-spacing between contiguous pixel centers
            self.xBlocks=1+np.floor_divide(self.xpix-1,BLOCK_SIZE) # number of blocks required in the x direction
            self.yBlocks=1+np.floor_divide(self.ypix-1,BLOCK_SIZE) # number of blocks required in the y direction
            self.xsize=self.xBlocks*BLOCK_SIZE # padded number of pixels in x direction
            self.ysize=self.yBlocks*BLOCK_SIZE # padded number of pixels in y direction
            #  - prepare arrays and functionality on GPU
            self.setAttr(
                'in_bounds',
                gpuarray.to_gpu(
                    np.array(
                        [[arena.inBounds(
                            self.xb0+i*self.dx+(self.yb0+j*self.dy)*1j
                            ) for i in xrange(self.xsize)] for j in xrange(self.ysize)],
                        dtype=np.float32
                        )
                    )
                )
            self.setAttr(
                'landscape',
                gpuarray.zeros(shape=(self.ysize,self.xsize),dtype=np.float32)
                )
            #  - prepare CUDA module definition
            module_definition='''
            // Kernel: Fill the landscape array with zeros
            __global__ void wipe(float *l) {
                int nx = blockDim.x * gridDim.x;
                int i = threadIdx.x + blockDim.x * blockIdx.x;
                int j = threadIdx.y + blockDim.y * blockIdx.y;
                int threadId = i + nx * j;
                l[threadId]=0;
            }

            // Kernel: Add the contribution of cheese at position (x,y) to the landscape
            __global__ void landscape_update_by_cheese(float *land, float *mask, float *a, float *b, int cheesenum, float horizon, float xb0, float xb1, float dx, float yb0, float yb1, float dy) {
                //thread indexing
                int nx = blockDim.x * gridDim.x;
                int i = threadIdx.x + blockDim.x * blockIdx.x;
                int j = threadIdx.y + blockDim.y * blockIdx.y;
                int k;
                int threadId = i + nx * j;

                __shared__ float A[100];
                __shared__ float B[100];
                
                for(k=0;k<cheesenum;k++) {
                    A[k]=a[k];
                    B[k]=b[k];
                }
                __syncthreads();

                //location in image
                float cx = xb0 + i * dx; // x-center of current pixel
                float cy = yb0 + j * dy; // y-center of current pixel

                //value of landscape at this location
                if(mask[threadId]>0) {
                    for(k=0;k<cheesenum;k++) {
                        land[threadId]+=fmaxf(0., horizon - powf(powf(A[k]-cx,2) + powf(B[k]-cy,2), 0.5));
                    }
                }
                else {
                    land[threadId]=-1;
                }
            }

            '''
            #  - initialize the CUDA module
            global_map_module=SourceModule(module_definition)
            cheese_contribution=global_map_module.get_function('landscape_update_by_cheese')
            wipe_landscape=global_map_module.get_function('wipe')
            self.setAttr('map_module',global_map_module)
            self.setAttr('cheese_contribution',cheese_contribution)
            self.setAttr('wipe_landscape',wipe_landscape)
            #  - update the landscape on GPU
            cheeses=[arena.getObj(tag)._pos for tag in arena.getCheeseList()]
            cheeses.append(complex(0,0))
            self.setAttr('cheeses_real',gpuarray.to_gpu(np.array([pos.real for pos in cheeses],dtype=np.float32)))
            self.setAttr('cheeses_imag',gpuarray.to_gpu(np.array([pos.imag for pos in cheeses],dtype=np.float32)))
            cheese_contribution(
                self.getAttr('landscape'),
                self.getAttr('in_bounds'),
                self.getAttr('cheeses_real'),
                self.getAttr('cheeses_imag'),
                np.int32(len(cheeses)-1),
                np.float32(arena.getMiscVal('horizon')),
                self.xb0,self.xb1,self.dx,
                self.yb0,self.yb1,self.dy,
                block=(BLOCK_SIZE,BLOCK_SIZE,1),
                grid=(self.xBlocks,self.yBlocks,1),
                )

            #  - prepare axis for plotting
            #ax.set_xlim((xb0-1,xb1+1))
            #ax.set_ylim((yb0-1,yb1+1))
            ax.xaxis.set_ticks(np.arange(self.xb0,self.xb1+1,1))
            ax.yaxis.set_ticks(np.arange(self.yb0,self.yb1+1,1))
            ax.tick_params(labelbottom=False,labelleft=False)
            #  - scent landscape map
            landscape_extent=(
                self.xb0-.5*self.dx,
                self.xb0+(self.xsize+.5)*self.dx,
                self.yb0-.5*self.dx,
                self.yb0+(self.ysize+.5)*self.dy,
                )
            scent_map=ax.imshow(
                self.getAttr('landscape').get(),
                cmap='Reds',
                extent=landscape_extent,
                origin='lower',
                vmin=-1,
                vmax=gpuarray.max(self.getAttr('landscape')).get(),
                )
            #  - construct repns for cheeses
            for objtag in arena.getObjs_ofType('cheese'):
                ch=arena.getObj(objtag) # get the cheese object
                repn=Repn_cheese(ch) # construct a reprensentation for it
                ch.addRepn('marker',repn) # append repn to the cheese object
                for tag in repn.getContentTags():
                    ax.add_artist(repn.getContent(tag)) # attach artists from repn to <ax>
            
            #  - construct repns for the mouse
            mus=arena.getObj('mus') # get the mouse object
            repn=Repn_mouse(mus) # construct its representation
            mus.addRepn('piece',repn) # append representation to mouse object
            for tag in repn.getContentTags():
                ax.add_artist(repn.getContent(tag)) # attach artists from representation to <ax>
            #  - construct artists for arena
            self.addContent('global_scent',scent_map)
            self.addContent('global_ax',ax)
            self.addContent('global_fig',fig)
            fig.show()
        else:
            raise Exception('Repn only valid for arenas of type arena_wmice. --- Aborting.\n')

    def update(self):
        arena=self._parent
        self.getContent('global_fig').suptitle('Single mouse experiment, cycle No. '+str(arena.getMiscVal('counter')),fontsize=16)
        #  - access to the CUDA module.
        cheese_contribution=self.getAttr('cheese_contribution')
        wipe_landscape=self.getAttr('wipe_landscape')
        #  - update the landscape on GPU.
        #    -- wipe the landscape:
        wipe_landscape(
            self.getAttr('landscape'),
            block=(BLOCK_SIZE,BLOCK_SIZE,1),
            grid=(self.xBlocks,self.yBlocks,1),
            )
        #    -- recalculate the landscape:
        cheeses=[arena.getObj(tag)._pos for tag in arena.getCheeseList()]
        #pad the cheese list to ensure pycuda doesn't try mallocking for an empty array
        cheeses.append(np.complex(0,0))
        self.setAttr('cheeses_real',gpuarray.to_gpu(np.array([pos.real for pos in cheeses],dtype=np.float32)))
        self.setAttr('cheeses_imag',gpuarray.to_gpu(np.array([pos.imag for pos in cheeses],dtype=np.float32)))
        cheese_contribution(
            self.getAttr('landscape'),
            self.getAttr('in_bounds'),
            self.getAttr('cheeses_real'),
            self.getAttr('cheeses_imag'),
            np.int32(len(cheeses)-1), #take padding into account!
            np.float32(arena.getMiscVal('horizon')),
            self.xb0,self.xb1,self.dx,
            self.yb0,self.yb1,self.dy,
            block=(BLOCK_SIZE,BLOCK_SIZE,1),
            grid=(self.xBlocks,self.yBlocks,1)
            )
        #    -- update the imshow plot:
        self.getContent('global_scent').set_data(self.getAttr('landscape').get())
        self.getContent('global_fig').canvas.draw()
        self.getContent('global_fig').canvas.flush_events()

                
def main():
    #intercative plot
    plt.ion()

    # provide obstacles as ellinfty balls of a given radius about a provided list of "bad seeds"
    ellinfty=lambda pos1,pos2: max(abs(pos1.real-pos2.real),abs(pos1.imag-pos2.imag))
    bad_seeds=[(7+10j,2.), (15+5j,1.5), (16+5j,1.5)]
    #bad_seeds=[]

    #initialize arena
    arena = Arena_wmouse(
        xbounds=(0,20),
        ybounds=(0,15),
        mouse_params={'horizon':5.,'order':20},
        cheese_params={'maxCheeses':20,'Ncheeses':10,'nibbles':2},
        visualQ=True,
        #out_of_bounds=lambda posn: False,
        out_of_bounds=lambda posn: any([ellinfty(posn,pos)<=dist for pos,dist in bad_seeds]),
        random_state=RS(),
        )

    plt.show()
    COMMANDS={
        'w':('move',(True,False,False,False,True)),
        's':('move',(False,True,False,False,True)),
        'a':('move',(False,False,True,False,False)),
        'd':('move',(False,False,False,True,False)),
        }
    while True:
        userInput = raw_input("what would you like to do now? w,a,s,d or st to stop\n")
        if userInput=='st':
            exit(0)
        elif userInput in COMMANDS.keys():
            arena.update(COMMANDS[userInput])
        else:
            pass
        
if __name__=='__main__':
    main()
    exit(0)
