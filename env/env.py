import gym
import copy
from env.utils.instance import JSP_Instance
from env.utils.re_instance import Reschedule_JSP_Instance
from env.utils.mach_job_op import *
from env.utils.graph import Graph
#from tools.plotter import Plotter

class JSP_Env(gym.Env):
    def __init__(self, args):
        self.args = args
        if args.reschedule == True:
            self.jsp_instance = Reschedule_JSP_Instance(args)
        else:
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
    
    def check_valid(self):
        job_proc = np.zeros((250, 250, 2),)
        for m in self.jsp_instance.machines:
            current_time = 0
            for his in m.processed_op_history:
                if his['start_time'] < current_time:
                    return False
                current_time = his['start_time'] + his['process_time']
                job_proc[his['job_id']][his['op_id']][0] = his['start_time']
                job_proc[his['job_id']][his['op_id']][1] = his['process_time']
        
        for i in range(len(self.jsp_instance.jobs)):
            job = self.jsp_instance.jobs[i]
            for j in range(1, job.op_num):
                if job_proc[i][j][0] < job_proc[i][j-1][0] + job_proc[i][j-1][1]:
                    return False
        
        return True

    def count_efficiency(self):
        ms = self.get_makespan()
        # print(ms)
        for m in self.jsp_instance.machines:
            continue
            # print(m.processed_op_history)

