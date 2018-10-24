from myp import *

MY_MYPS={}

def listmyps():
    return MY_MYPS.keys()

def openmyp(filename):
    tmp=myp(filename+'.content',0)
    MY_MYPS[filename]=tmp
    return tmp

def closemyp(filename):
    try:
        tmp=MY_MYPS[filename]
    except:
        return None

    MY_MYPS.pop(filename)
    del(tmp)
    return 'File \"'+filename+'\" removed and closed'
    
def report(filename,veh):
    try:
        f=MY_MYPS[filename]
    except:
        return None
    
    f.moveto(0)
    while f.content()!=None:
        if veh in f.content():
            print f.content()[veh]
        f.advance()
    return None
