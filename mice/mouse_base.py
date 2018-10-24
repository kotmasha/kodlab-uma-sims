import sys
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib as mpl

from numpy.random import randint as rnd
from copy import deepcopy as cp
from collections import deque



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

    def ellone(self):
        return abs(self.real)+abs(self.imag)

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
 
    def convert(self):
        return complex(self.real,self.imag)

    def strip(self):
        return (self.real,self.imag)

#Global Variables
North=icomplex(0,1)
South=icomplex(0,-1)
West=icomplex(-1,0)
East=icomplex(1,0)
Origin=icomplex(0,0)
DIRS=[North,East,South,West]
REL_DIRS={'fd':East,'bk':West,'lt':North,'rt':South}

#arena base class
class Arena_base():
    def __init__(self,xbounds, ybounds,out_of_bounds=lambda x:False):
        self._xbounds=xbounds #is a tuple of the form (xmin,xmax), bounds for drawing
        self._ybounds=ybounds #is a tuple of the form (ymin,ymax)
        self._objects={}
        self._oob=out_of_bounds #a function accepting an icomplex position and returning Boolean
        self._map=np.zeros((xbounds[1]-xbounds[0],ybounds[1]-ybounds[0]))
        self._ax=None
        self._fig=None
        self._cheeses = []
        self._misc='0'

    def putmisc(self,txt):
        self._misc=str(txt)

    def inBounds(self, pos):
        if pos.real<self._xbounds[0] or pos.real>=self._xbounds[1] or pos.imag<self._ybounds[0] or pos.imag>=self._ybounds[1]:
            return False
        else:
            return not self._oob(pos)

    def attrCalc(self,pos,attributeName):
        if attributeName.find('Gradient')>=0:
            return sum([self._objects[objtag]._rescaling[attributeName](pos-self._objects[objtag]._pos) for objtag in self._objects])
        return sum([self._objects[objtag]._rescaling[attributeName](abs(self._objects[objtag]._pos-pos)) for objtag in self._objects])

    def addMouse(self,tag,pos,attributes,viewports={}):
        self._objects[tag]=mouse(self,tag,pos,attributes,viewports)
        #self._objects.append(mouse(self,tag,pos,attributes,viewports))

    def addRandomMouse(self,tag,viewsize=5,viewports={}):
        mouseAttr={'viewSize':viewsize}
        mouseAttr['direction']=[North,West,South,East][rnd(4)]
        mouse_start_pos = icomplex(rnd(self._xbounds[1]-self._xbounds[0]),rnd(self._ybounds[1]-self._ybounds[0]))
        self.addMouse(tag,mouse_start_pos,mouseAttr,viewports)

    def addRandomMice(self,how_many,viewsize=5,viewports={}):
        # construct x random mice
        for ind in xrange(how_many):
            self.addRandomMouse('mus'+str(ind+1),viewsize,viewports)

    def addCheese(self,tag,pos,params):
        self._objects[tag]=cheese(self,tag,pos,params)
        #self._objects.append(cheese(self,tag,pos,params,attributes))

    def addRandomCheese(self,ind,params):
        self.addCheese(
            'ch'+str(ind),
            icomplex(rnd(self._xbounds[1]-self._xbounds[0]),rnd(self._ybounds[1]-self._ybounds[0])),
            params
            )
    
    def addRandomCheeses(self,how_many,params):
        for ind in xrange(how_many):
            self.addRandomCheese(ind+1,params)

    def getMaxElev(self):
        return np.amax(self._map)

    def getMice(self,tag='all'):
        if tag=='all':
            tmpMice={}
            for objtag in self._objects.keys():
                if self._objects[objtag]._type=='mouse':
                    tmpMice[tag]=self._objects[objtag]
            return tmpMice
        elif self._objects[tag]._type=='mouse':
            return self._objects[tag]
        else:
            raise Exception('No objects with tag \''+str(tag)+'\' in arena.\n\n')

    def update_objs(self,tup):
        for objtag in self._objects:
            self._objects[objtag].update(tup)
        
    def generateHeatmap(self,attributeName):
        for x in xrange (0,self._xbounds[1]-self._xbounds[0]):
            for y in xrange (0,self._ybounds[1]-self._xbounds[0]):
                self._map[y][x]=self.attrCalc(icomplex(x,y),attributeName)
        self._fig, self._ax = plt.subplots(1)
        self._fig.suptitle('Mouse experiment, cycle No. '+str(self._misc),fontsize=16)
        self._ax.xaxis.set_ticks(-0.5+np.arange(self._xbounds[0],self._xbounds[1]+2,5))
        self._ax.yaxis.set_ticks(-0.5+np.arange(self._ybounds[0],self._ybounds[1]+2,5))
        self._ax.tick_params(labelbottom=False,labelleft=False)
        self._ax.imshow(self._map, cmap = 'Spectral', vmin = -1, vmax = self.getMaxElev())
        for objtag in self._objects.keys():
            obj=self._objects[objtag]
            if obj._type == 'cheese':
                Circle = plt.Circle((obj._pos.real,obj._pos.imag),0.2,color='black')
                self._cheeses.append(self._ax.add_artist(Circle))
            elif obj._type == 'mouse':
                self._mouse=self._ax.add_patch(patches.Rectangle((obj._pos.real-0.5,obj._pos.imag-0.5),1,1,color=(0.0,0.0,0.0)))
                self._direction=self._ax.arrow(obj._pos.real,obj._pos.imag,obj._attr['direction'].real,obj._attr['direction'].imag,head_width=1,head_length=1.5)
                self._Gradient=self._ax.arrow(obj._pos.real,obj._pos.imag,self.attrCalc(obj._pos,attributeName + 'Gradient').real,self.attrCalc(obj._pos,attributeName + 'Gradient').imag,head_width=1,head_length=1.5,color='r')
            else:
                raise Exception('wrong object type')
        self._ax.invert_yaxis()

    def updateHeatmapFull(self,attributeName):
        self._ax.clear()
        for x in xrange (0,self._xbounds[1]-self._xbounds[0]):
            for y in xrange (0,self._ybounds[1]-self._xbounds[0]):
                self._map[y][x]=self.attrCalc(icomplex(x,y),attributeName)
        self._ax.xaxis.set_ticks(-0.5+np.arange(self._xbounds[0],self._xbounds[1]+2,5))
        self._ax.yaxis.set_ticks(-0.5+np.arange(self._ybounds[0],self._ybounds[1]+2,5))
        self._ax.tick_params(labelbottom=False,labelleft=False)
        self._ax.imshow(self._map, cmap = 'Spectral', vmin = -1, vmax = self.getMaxElev())
        for objtag in self._objects.keys():
            obj=self._objects[objtag]
            if obj._type == 'cheese':
                Circle = plt.Circle((obj._pos.real,obj._pos.imag),0.2,color='w')
                self._ax.add_artist(Circle)
            elif obj._type == 'mouse':
                self._mouse=self._ax.add_patch(patches.Rectangle((obj._pos.real-0.5,obj._pos.imag-0.5),1,1,color=(0.0,0.0,0.0)))
                self._direction=self._ax.arrow(obj._pos.real,obj._pos.imag,obj._attr['direction'].real,obj._attr['direction'].imag,head_width=1,head_length=1.5)
                self._Gradient=self._ax.arrow(obj._pos.real,obj._pos.imag,self.attrCalc(obj._pos,attributeName + 'Gradient').real,self.attrCalc(obj._pos,attributeName + 'Gradient').imag,head_width=1,head_length=1.5,color='r')
            else:
                raise Exception('wrong obj type')
        self._ax.invert_yaxis()

    def updateHeatmap(self,attributeName):
        obj = self.getMice('mus')
        self._mouse.remove()
        self._Gradient.remove()
        self._direction.remove()
        self._fig.suptitle('Mouse experiment, cycle No. '+str(self._misc),fontsize=16)
        self._mouse=self._ax.add_patch(patches.Rectangle((obj._pos.real-0.5,obj._pos.imag-0.5),1,1,color=(0.0,0.0,0.0)))
        self._direction=self._ax.arrow(obj._pos.real,obj._pos.imag,obj._attr['direction'].real,obj._attr['direction'].imag,head_width=1,head_length=1.5)
        self._Gradient=self._ax.arrow(obj._pos.real,obj._pos.imag,self.attrCalc(obj._pos,attributeName + 'Gradient').real,self.attrCalc(obj._pos,attributeName + 'Gradient').imag,head_width=1,head_length=1.5,color='r')
                

#objects placeable in the arena
class obj(object):
    def __init__(self,ar,typ,tag,pos):
        if type(typ)==type('0') and type(tag)==type('0'):
            self._ar=ar
            self._type=typ
            self._tag=tag
            self._rescaling = {
                'elevation': lambda x: 0,
                'elevationGradient':lambda x: 0,
                }
        else:
            raise Exception('Object type and tag must be strings.\n\n')
        if type(pos)==type(Origin):
            self._pos=pos
        else:
            raise Exception('Object position must be of type icomplex')
        self._attr={}

    def update(self,command=(False,False,False,False,False)):
        return None

    def remove(self):
        del self._ar._objects[self._tag] #note each object is uniquely represented on the list of objects of an arena.

#mouse class:
class mouse(obj):
    def __init__(self,ar,tag,pos,attributes,viewport={}):
        obj.__init__(self,ar,'mouse',tag,pos)
        self._attr=attributes # the attribute 'viewSize' is assumed to be present, and non-negative int-valued.
        # create viewports for different numerical attributes
        self._viewport={}
        for item in viewport:
            try:
                #self._viewport[item]=np.zeros((attributes['viewSize']*2+1,attributes['viewSize']*2+1))
                self._viewport[item]={}
                self._viewport[item]['data']=np.zeros((attributes['viewSize']*2+1,attributes['viewSize']*2+1))
                self._viewport[item]['fig'],self._viewport[item]['ax']=plt.subplots(1)
                ax=self._viewport[item]['ax']
                ax.tick_params(labelbottom=False,labelleft=False)
                ax.xaxis.set_ticks(-0.5+np.arange(0,2*self._attr['viewSize']+2,1))
                ax.yaxis.set_ticks(-0.5+np.arange(0,2*self._attr['viewSize']+2,1))
                #self.updateViewport(attributeName)
                self._viewport[item]['map']=ax.imshow(
                    self._viewport[item]['data'],
                    cmap = 'Spectral',
                    vmin = -1,
                    vmax = self._ar.getMaxElev(),
                    animated=True,
                    )
                #marking the mouse in the viewport
                ax.add_patch(patches.Rectangle((self._attr['viewSize']-0.5,self._attr['viewSize']-0.5),1,1,color=(0.0,0.0,0.0)))
                #marking the mouse pose in the viewport
                ax.arrow(self._attr['viewSize'],self._attr['viewSize'],0,1,head_width=0.3, head_length=0.6)
                #drawing the gradient in the viewport
                self._viewport[item]['gradient']=ax.arrow(self._attr['viewSize'],self._attr['viewSize'],(self._ar.attrCalc(self._pos,item+'Gradient')*complex(0,1)/self._attr['direction'].convert()).real,(self._ar.attrCalc(self._pos,item+'Gradient')*complex(0,1)/self._attr['direction'].convert()).imag,color='r',head_width=0.3, head_length=0.6)
                #textboxes in the viewport
                boxprops=dict(boxstyle='round',facecolor='white')
                self._viewport[item]['ltbox']=ax.text(0.05,0.475,max([ind for ind in xrange(18) if self.angle(ind,'lt')]),transform=ax.transAxes,fontsize=14,bbox=boxprops)
                self._viewport[item]['bkbox']=ax.text(0.475,0.05,max([ind for ind in xrange(18) if self.angle(ind,'bk')]),transform=ax.transAxes,fontsize=14,bbox=boxprops)
                self._viewport[item]['rtbox']=ax.text(0.9,0.475,max([ind for ind in xrange(18) if self.angle(ind,'rt')]),transform=ax.transAxes,fontsize=14,bbox=boxprops)
                self._viewport[item]['fdbox']=ax.text(0.475,0.9,max([ind for ind in xrange(18) if self.angle(ind,'fd')]),transform=ax.transAxes,fontsize=14,bbox=boxprops)
                self._viewport[item]['extra']=ax.text(0.05,0.9,self.mdist(),transform=ax.transAxes,fontsize=14,bbox=boxprops)
                ax.invert_yaxis()
            except KeyError:
                raise('\n\n===> Mouse must have a \"viewSize\" attribute.\n\n')

    def update(self,command=(False,False,False,False,False)):
        return self.move(command)

    def move(self,command=(False,False,False,False,False)):
        # assume command is a Boolean tuple of the form:
        fd,bk,lt,rt,arb=command
        response_step=False
        response_turn=False
        if arb:
            response_step=self.step(fd,bk)
        else:
            response_turn=self.turn(lt,rt)
        return response_step and response_turn # report whether or not the move suceeded

    def step(self,fwd,bck):
        # assume fwd and bck are boolean
        newpos=self._pos+self._attr['direction']*((1 if fwd else 0)+(-1 if bck else 0))
        if self._ar.inBounds(newpos):
            self._pos=newpos
            return True # move succeeded
        else:
            return False # move did not succeed

    def turn(self,lt,rt):
        # assume lt and rt are boolean
        self._attr['direction']=self._attr['direction']*(North if lt else 1)*(South if rt else 1)
        return True # turn move always succeeds

    def teleport(self,posn,pose):
        # set mouse position, if legal
        if not self._ar._oob(posn):
            self._pos=posn
        else:
            raise Exception('Invalid mouse teleport.\n')
        
        # set mouse pose, if legal
        if pose in [North,South,East,West]:
            self._attr['direction']=pose
        else:
            raise Exception('Invalid mouse teleport.\n')

    def worldPos(self,v):
        # switch from viewport coordinates to world coordinates
        # assuming v.real and v.imag are between 0 and 2*self._attr['viewSize']
        loc=v-icomplex(self._attr['viewSize'],self._attr['viewSize'])
        return self._pos+loc*(South*self._attr['direction'])

    def viewportPos(self,pos):
        # switch from global coordinates to local coordinates relative to mouse position and pose
        # pos is assumed to be of type icomplex.
        return (pos-self._pos)*North*self._attr['direction'].conj()

    def mdist(self):
        return min([(self._pos-self._ar._objects[tag]._pos).ellone() for tag in self._ar._objects.keys() if self._ar._objects[tag]._type=='cheese'])

    def updateViewport(self, attribute): # updates the viewport corresponding to a given attribute
        # update the viewport:
        vp=lambda x,y,p: self._ar.attrCalc(p,attribute) if self._ar.inBounds(p) else -1
        self._viewport[attribute]['data']=[[vp(x,y,self.worldPos(icomplex(x,y))) for x in xrange(2*self._attr['viewSize']+1)] for y in xrange(2*self._attr['viewSize']+1)]
        self._viewport[attribute]['map'].set_data(self._viewport[attribute]['data'])
        self._viewport[attribute]['map'].autoscale()
        
        # update the gradient:
        self._viewport[attribute]['gradient'].remove()
        self._viewport[attribute]['gradient']=self._viewport[attribute]['ax'].arrow(self._attr['viewSize'],self._attr['viewSize'],(self._ar.attrCalc(self._pos,attribute+'Gradient')*complex(0,1)/self._attr['direction'].convert()).real,(self._ar.attrCalc(self._pos,attribute+'Gradient')*complex(0,1)/self._attr['direction'].convert()).imag,color='r',head_width=0.3, head_length=0.6)

        # update the text boxes:
        self._viewport[attribute]['ltbox'].set_text(max([ind for ind in xrange(18) if self.angle(ind,'lt')]))
        self._viewport[attribute]['bkbox'].set_text(max([ind for ind in xrange(18) if self.angle(ind,'bk')]))
        self._viewport[attribute]['rtbox'].set_text(max([ind for ind in xrange(18) if self.angle(ind,'rt')]))
        self._viewport[attribute]['fdbox'].set_text(max([ind for ind in xrange(18) if self.angle(ind,'fd')]))
        self._viewport[attribute]['extra'].set_text(self.mdist())


    #-----------------------------measurements---------------------------------

    def calculate_gradient(self,attributeName):
        return self._ar.attrCalc(self._pos,attributeName+'Gradient')

    def elevation(self): #elevation of mouse
        return self._ar.attrCalc(self._pos,'elevation')

    def averageElevation(self): #average elevation of mouse viewport
        Sum=0
        for x in self._viewport['elevation']:
            for y in self._viewport['elevation'][x]:
                Sum+=y
        return Sum/pow(self._attr['viewSize']*2+1,2)

    def calculate_cos_grad(self,attribute):
        # assuming x will the gradient of the elevation landscape, of type complex, the cosine of the angle between x and the relative pose:
        cos=lambda v: 0 if abs(v)==0 else (v*self._attr['direction'].conj().convert()).real/abs(v)
        return cos(self.calculate_gradient(attribute))        

    def angle(self,ind,dirn):
        # assuming dirn will be 'fd'/'bk'/'lt'/'rt' wrt the pose vector:
        relposec=(REL_DIRS[dirn]*self._attr['direction']).conj().convert()
        # assuming x will the gradient of the elevation landscape, of type complex, the cosine of the angle between x and the relative pose:
        cos=lambda v: 0 if abs(v)==0 else (v*relposec).real/abs(v)
        # rescaling function to be applied to the cosine of the angle between grad and dirn:
        rescaling=lambda x: pow(4.,1.+x)
        return rescaling(cos(self.calculate_gradient('elevation')))>=ind

#cheese class:
class cheese(obj):
    def __init__(self,ar,tag,pos,params = {'nibbles':1,'nibbleDist':1}):
        obj.__init__(self,ar,'cheese',tag,pos)
        self._rescaling['elevation'] = lambda x: 10*np.exp(-x/25)
        self._rescaling['elevationGradient']=lambda x: -10/25*np.exp(-abs(x)/100)*2*complex(x.real,x.imag)
        self._params = params # params is a dictionary of initialization parmeters
        self._attr={'counter':0}

    def update(self,liszt=[False,False,False,False,False]):
        nibbleP=False
        for objtag in self._ar._objects:
            item=self._ar._objects[objtag]
            if item._type == 'mouse' and np.sqrt(abs(item._pos-self._pos))<=self._params['nibbleDist']:
                nibbleP=True
                break

        if nibbleP:
            self._attr['counter']+=1
        else:
            self._attr['counter']=0

        if self._attr['counter']>=self._params['nibbles']:
            self.remove()

   
    
        
                
def main():
    plt.ion()
    xbounds = 0,20
    ybounds = 0,20
    arena = Arena_base(xbounds, ybounds)
    cheeseParams={
        'nibbles':2,
        'nibbleDist':1,
        }
    arena.addRandomCheeses(5,cheeseParams)
    arena.addRandomMouse('mus',viewports={'elevation'})
    arena.generateHeatmap('elevation')
    #arena._objects['mus'].generateHeatmap('elevation')
    arena._objects['mus'].updateViewport('elevation')
    arena._fig.show()
    arena._objects['mus']._viewport['elevation']['fig'].canvas.draw()
    mouse = arena.getMice('mus')
    commands={
        'w':(True,False,False,False,True),
        's':(False,True,False,False,True),
        'a':(False,False,True,False,False),
        'd':(False,False,False,True,False),
        }
    while True:
        userInput = raw_input("what would you like to do now? w,a,s,d or st to stop\n")
        if userInput=='st':
            exit(0)
        elif userInput in ['w','s','a','d']:
            #lenObj = len(arena._objects.keys())
            print mouse.update(commands[userInput])
            for objtag in arena._objects.keys():
                obj=arena._objects[objtag]
                if obj._type == 'cheese':
                    obj.update()
            #if lenObj> len(arena._objects):
            #    arena.updateHeatmapFull('elevation')
            #    lenObj = len(arena._objects)
            #else:
            #    arena.updateHeatmap('elevation')
            arena.updateHeatmapFull('elevation')
            mouse.updateViewport('elevation')
            arena._fig.show()
            mouse._viewport['elevation']['fig'].canvas.draw()
            mouse._viewport['elevation']['fig'].canvas.flush_events()
            #userInput = raw_input("what would you like to do now? w,a,s,d or st to stop\n")
        else:
            pass
        
if __name__=='__main__':
    main()
    exit(0)