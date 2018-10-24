# This module is designed for reading a number of consecutive pickled objects
# from input file $file$.
#   $file$ is assumed to contain *only* a sequence of Python pickles.
from collections import deque
import cPickle

class myp(object):
    def __init__(self,filename,width):
        self._width=2*width+1 #width is assumed to be a non-negative int
        self._source=open(filename,'rb') #filename is assumed to be a complete path or a local filename.
        self._unp=cPickle.Unpickler(self._source) #unpickler for source file
        self._ref=[0] #reference list containing the positions in $file$ of all the pickles encountered so far
        self._content=deque([],self._width)
        #now make the first reading
        for ind in xrange(self._width):
            self._content.append(self._unp.load())
            self._ref.append(self._source.tell())

    def __repr__(self):
        tmp_str='['
        for x in self._ref:
            if x!=self._source.tell():
                tmp_str+=str(x)
            else:
                tmp_str+='*'+str(x)+'*'
            if x!=self._ref[-1]:
                tmp_str+=','
            else:
                tmp_str+=']'
        return str(self._content)+'\n'+tmp_str

    def center(self): # report the center of the pickle frame
        ref=self._ref.index(self._source.tell())-(self._width-1)/2-1
        return ref #,self._content[ref],self._ref[ref]
    
    def content(self,index=0): # return the content of the PickleReader with given offset from center
        return self._content[(index+(self._width-1)/2)%self._width]

    def content_all(self): # return the content of the PickleReader
        return self._content

    #THESE NEED TO BE REFORMULATED TO SYMMETRIC VERSIONS
    #def expand(self): # 
    #    self._width+=1
    #    self._content.append(self._unp.load())
    #    pos=self._source.tell()
    #    if pos>self._ref[-1]:
    #        self._ref.append(self._source.tell())
    #    return self
    #
    #def contract(self):
    #    if self._width>0:
    #        self._width-=1
    #        self._content=self._content[1:]
    #    return self
            
    def advance(self):
        #read the new pickle
        try:
            new_pickle=self._unp.load()
        except EOFError:
            new_pickle=None
        #update the content
        self._content.append(new_pickle)
        #update the reference list
        new_pos=self._source.tell()
        if new_pos>self._ref[-1]:
            self._ref.append(new_pos)
        return None
            
    def rewind(self):
        #unread a pickle
        
        pos=self._source.tell() #current read position in input file
        nextp=self._ref.index(pos) #index of next pickle to be read
        if nextp-self._width<1:
            new_pickle=None
        else:
            self._source.seek(self._ref[nextp-self._width-1])
            new_pickle=self._unp.load()
        #update content
        self._content.appendleft(new_pickle)
        #reset read position to end of window
        self._source.seek(self._ref[nextp-1 if nextp-1>=0 else 0])
        return None

    def moveto(self,ind):
        ref=self.center()
        if ind>ref:
            while ind>self.center():
                self.advance()
                if self.center()==ref:
                    return ref,None
                else:
                    ref+=1
        elif ind<ref:
            while ind<self.center():
                self.rewind()
                if self.center()==ref:
                    return ref,None
                else:
                    ref-=1

        return ref,self.content()
        #return None
