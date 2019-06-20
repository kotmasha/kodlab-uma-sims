# This is the launcher for UMA Simulation
import sys
import os
import shutil
import importlib
import json
from cluster.cluster import *

def check_fields(data):
    fmt = "{} field is missing from yaml file!"
    for s in ['script', 'func', 'Nruns', 'first_run', 'params']:
        if s not in data:
            raise Exception(fmt.format(s))

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print "Usage of launcher: launcher.py <test.yml>"
        exit()

    test_yml = sys.argv[1]
    data = YamlManager(test_yml).get_dict()

    check_fields(data)

    script = importlib.import_module(data['script'])
    func = getattr(script, data['func'])

    Nruns = int(data['Nruns'])
    first_run=int(data['first_run'])
    data['params']['Nruns']=data['Nruns']
    data['params']['first_run']=data['first_run']
    params = data['params']
    #print params
    test_name = data['params']['name']
    test_yml_abs_path = os.path.join(os.getcwd(), test_yml)
    print "Simulation label: %s" % test_name
    print "Simulation yml file: %s" % test_yml_abs_path
    print "Using script: %s" % str(script.__name__)
    print "Function called from script: %s" % str(func.__name__)
    print "Will execute %d simulation runs.\n" % Nruns

    #dump_pickle(test_name, params)
    # create preamble file in subdirectory named $test_name$
    script_working_directory=os.path.join(os.getcwd(),test_name)
    try:
        os.mkdir(test_name)
    except:
        shutil.rmtree(script_working_directory)
        os.mkdir(test_name)

    #script working directory:
    preamble_file_name=os.path.join(script_working_directory,test_name+'.pre')           
    preamblef = open(preamble_file_name,'wb')
    json.dump(params, preamblef)
    preamblef.close()

    # run the specified script with given .yml input
    cluster = ClusterManager()
    pool = PoolManager()
    pool.start(func, test_yml_abs_path, Nruns, first_run, cluster.get_Ninstances(), cluster.get_port(), cluster.get_host())

    print "All runs are done.\n"
