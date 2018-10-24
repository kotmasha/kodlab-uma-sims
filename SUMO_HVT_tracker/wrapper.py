import som_platform
import atexit
import UMA_NEW
import numpy as np
from numpy.random import randint as rnd


def getLog(brain):
    file = open("log_" + brain.agent._EXPERIMENT.din(brain.agent._MID) + '.txt', 'w')
    file.write("#This is the testing log file\n\n")

    for snapshot_name, snapshot in brain._snapshots.iteritems():
        logging_info = snapshot.get_log()
        file.write('\n\n' + logging_info.agent_name + '\n')
        #file.write('\n\n' + str(brain.agent._INITMASK.value_all()) + '\n')

        # file.write("-----------performance stats--------------\n")
        # file.write("OVERALL CPU USAGE:\n")
        # file.write(str(logging_info.CPU_MEM) + 'Bytes(' + str(logging_info.CPU_MEM * 1.0 / 1024 / 1024) + 'M)\n')
        # file.write("OVERALL GPU USAGE:\n")
        # file.write(str(logging_info.GPU_MEM) + 'Bytes(' + str(logging_info.GPU_MEM * 1.0 / 1024 / 1024) + 'M)\n')
        # file.write("UPDATE_WEIGHT:\n")
        # write_stats_info(file, logging_info.STAT_UPDATE_WEIGHT)
        # file.write("ORIENT_ALL:\n")
        # write_stats_info(file, logging_info.STAT_ORIENT_ALL)
        # file.write("PROPAGATION:\n")
        # write_stats_info(file, logging_info.STAT_PROPAGATION)
        # file.write("-----------performance stats--------------\n\n")
	    # 
        # file.write("----------------Simulation----------------\n")
        # file.write(logging_info.str_num_sim)
        # file.write("----------------Simulation----------------\n\n")
        file.write("-----------simulation process--------------\n")
        file.write(logging_info.str_process)
        file.write("-----------simulation process--------------\n\n")
        # file.close()
        # file.close()
    file.close()

def saveData(wrapper):
    wrapper.saveData()

def write_stats_info(file, perf_stats):
    file.write("num:   " + str(perf_stats.n) + "\n")
    file.write("time  :" + str(round(perf_stats.acc_t, 3)) + "\n")
    file.write("avg_t :" + str(round(perf_stats.avg_t, 3)) + "\n")
    file.write("\n")

class brain:
    def __init__(self, agent):
        self.agent = agent
        self._snapshots = {}
        atexit.register(getLog, self)

    def init(self, filename=""):
        #initialize agent
        sensor_names = self.agent._SENSORS
        for snapshot_name, snapshot in self._snapshots.iteritems():
            snapshot.init_data(snapshot_name, self.agent._SIZE / 2, sensor_names, filename)

    def add_snapshot(self, name, params, using_agent, using_log):
        discount = params[0]
        snapshot = UMA_NEW.Agent_Discounted(discount, using_agent)
        self._snapshots[name] = snapshot
    
    def up(self, sig, stability):
        sig_new={}
        for token in ['plus','minus']:
            self._snapshots[token].up_GPU(sig.value_all().tolist(),stability)
            sig_new[token]=som_platform.Signal(self._snapshots[token].getSignal())
        return sig_new


    #def propagate(self, signal, load):
    #    self.brain.propagate_GPU(signal._VAL.tolist(), load._VAL.tolist())
    #    result = self.brain.getLoad()
    #    return som_platform.Signal(result)

    def get_log_update_weight(self):
        n = self.brain.get_n_update_weight()
        t = self.brain.get_t_update_weight()
        return n, t, t / n

    def get_log_orient_all(self):
        n=self.brain.get_n_orient_all()
        t=self.brain.get_t_orient_all()
        return n, t, t / n

    def get_log_propagation(self):
        n = self.brain.get_n_propagation()
        t = self.brain.get_t_propagation()
        return n, t, t / n

            
    def amper(self, signals):
        for snapshot_name, snapshot in self._snapshots.iteritems():
            snapshot.amper([signal._VAL.tolist() for signal in signals])

    def delay(self, signals):
        for snapshot_name, snapshot in self._snapshots.iteritems():
            snapshot.delay([signal._VAL.tolist() for signal in signals])

    def blocks(self, dists, delta):
        return self.brain.blocks_GPU(dists, delta)
        
    def saveData(self):
        self.brain.savingData('test_data')

    def get_delta_weight_sum(self, signal):
        return self.brain.get_delta_weight_sum(signal._VAL.tolist())
        
