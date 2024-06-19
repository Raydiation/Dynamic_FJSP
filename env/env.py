import gym
import copy
from env.utils.instance import JSP_Instance
from env.utils.mach_job_op import *
from env.utils.graph import Graph

class JSP_Env(gym.Env):
    def __init__(self, args):
        self.args = args
        self.jsp_instance = JSP_Instance(args)

    def step(self, step_op):
        self.jsp_instance.assign(step_op)
        avai_ops = self.jsp_instance.current_avai_ops()
        return avai_ops, self.done()
    
    def reset(self):
        self.jsp_instance.reset()
        return self.jsp_instance.current_avai_ops()
       
    def done(self):
        return self.jsp_instance.done()

    def get_tardiness(self):
        return int(sum([max(0, j.operations[-1].finish_time - j.due_date) for j in self.jsp_instance.jobs]))
    
    def get_makespan(self):
        return max(m.avai_time() for m in self.jsp_instance.machines)    

    def get_graph_data(self):
        return self.jsp_instance.get_graph_data()
        
    def load_instance(self, filename, block_breakdown=False):
        self.jsp_instance.load_instance(filename, block_breakdown)
        return self.jsp_instance.current_avai_ops()

