import bisect
import random
from env.utils.mach_job_op import *
from env.utils.generator import *
from env.utils.graph import Graph
import torch
from djsp_logger import DJSP_Logger
import time
from itertools import accumulate
from copy import deepcopy
from collections import deque

BREAKDOWN = -1

class JSP_Instance:
    def __init__(self, args):
        self.args, self.process_time_range = args, [1, args.max_process_time]
        self.job_num, self.machine_num = 0, 0
        self.jobs, self.unarr_jobs, self.machines = [], deque(), []
        self.graph, self.max_process_time = None, 0
        self.current_time, self.time_stamp = 0, []

        self.use_log, self.logger = self.args.use_log, DJSP_Logger()

        self.env_back = None

    ##### basic functions
    def generate_case(self):
        self.graph = Graph(self.args, self.machine_num)
        self.register_time(0)

        for _ in range(self.ini_job_num):
            job = self.create_jobs(0)
            self.jobs.append(job)
            self.job_num += 1
            self.graph.add_job(job)

        if self.args.train_arr:
            # job insertion
            job_arrival_time = deque(accumulate(map(int, np.random.exponential(scale=self.args.arrival_time_dist, size=self.args.new_job_event))))
            for arr_time in job_arrival_time:
                self.register_time(arr_time)

            for arr_time in job_arrival_time:
                for _ in range(self.args.new_job_per_num):
                    self.unarr_jobs.append(self.create_jobs(arr_time))

        if self.args.train_break:
            for machine in self.machines:
                machine.generate_breakdown()
                self.register_time(machine.breakdown_time)
                self.register_time(machine.breakdown_time + machine.repair_time)

        
    def create_jobs(self, arrival_time):
        job_id = len(self.jobs) + len(self.unarr_jobs)
        op_config = gen_operations_FJSP(self.machine_num, self.process_time_range)
        
        return Job(args=self.args, job_id=job_id, op_config=op_config, arrival_time=arrival_time)

    def reset(self):
        self.ini_job_num, self.new_job_num, self.machine_num = self.args.ini_job_num, self.args.new_job_event * self.args.new_job_per_num, self.args.machine_num
        self.job_num = 0

        self.jobs, self.unarr_jobs = [], deque()
        self.machines = [Machine(int(machine_id), self.args) for machine_id in range(self.machine_num)]
        self.current_time = 0
        self.time_stamp = []
        self.graph = None
        self.generate_case()
        self.logger.reset()

        self.env_back = deepcopy(self)

    def restart(self):
        self = self.env_back

    def load_instance(self, filename, block_breakdown=False):
        self.jobs, self.unarr_jobs, self.time_stamp = [], deque(), []
        self.current_time, self.job_num = 0, 0
        self.new_job_num = 0

        with open(filename) as f:
            self.ini_job_num, self.new_job_num, self.machine_num = map(int, f.readline().split())
            self.machines = [Machine(int(machine_id), self.args) for machine_id in range(self.machine_num)]

            self.graph = Graph(self.args, self.machine_num)
            self.register_time(0)
        
            for i in range(self.ini_job_num + self.new_job_num):
                op_config = []
                line = f.readline().split()
                arrival_time, op_num = int(line[0]), int(line[1])
                cur = 2
                for j in range(op_num):
                    mach_ptime = []
                    machine_num = int(line[cur])
                    cur += 1
                    for _ in range(machine_num):
                        machine_id, process_time = int(line[cur]), int(line[cur + 1])
                        mach_ptime.append((machine_id - 1, process_time))
                        cur += 2
                    op_config.append({"id": j, "machine_and_processtime": mach_ptime})

                if arrival_time == 0:
                    self.jobs.append(Job(args=self.args, job_id=i, op_config=op_config, arrival_time=0))
                    self.graph.add_job(self.jobs[-1])
                    self.job_num += 1
                else:
                    self.register_time(arrival_time)
                    self.unarr_jobs.append(Job(args=self.args, job_id=i, op_config=op_config, arrival_time=arrival_time))
                    
            if self.args.block_breakdown == False:
                line = f.readline() # line = "BREAKDOWN EVENT : "
                for i in range(self.machine_num):
                    self.machines[i].breakdown_event.append((0, 0))
                    line = f.readline().replace('[', ' ').replace(']', ' ').replace('(', ' ').replace(')', ' ').replace(',', ' ').split()
                    for idx in range(len(line) // 2):
                        self.machines[i].breakdown_event.append((int(line[idx * 2]), int(line[idx * 2 + 1])))
                    self.machines[i].next_breakdown_event()
                    self.register_time(self.machines[i].breakdown_time)
                    self.register_time(self.machines[i].breakdown_time + self.machines[i].repair_time)

        self.logger.reset()
        self.env_back = deepcopy(self)
        
    def done(self):
        if not all(job.done() for job in self.jobs):
            return False

        if not (self.job_num == self.ini_job_num + self.new_job_num):
            return False

        if not all(m.breakdown_time >= m.avai_time() for m in self.machines):
            return False

        return True

    def current_avai_ops(self):
        return self.available_ops()

    def get_graph_data(self):
        self.graph.update_feature(self.jobs, self.machines, self.current_time)
        data, op_unfinished, job_srpt = self.graph.get_data()
        return data.to(self.args.device), op_unfinished, job_srpt.to(self.args.device)
        
    def assign(self, step_op):
        job_id, op_id, node_id, machine_id = step_op['job_id'], step_op['op_id'], step_op['node_id'], step_op['m_id']
        op = self.jobs[job_id].current_op()
        op_info = {
            "job_id"        : job_id,
            "op_id"         : op_id,
            "start_time"    : max(self.current_time, op.avai_time),
            "process_time"  : step_op['process_time'],
            "node_id"       : node_id
        }

        op_finished_time = self.machines[machine_id].process_op(op_info)
        self.jobs[job_id].current_op().update(self.current_time, op_info["process_time"])

        if self.jobs[job_id].next_op() != -1:
            self.jobs[job_id].update_current_op(avai_time=op_finished_time)
        self.register_time(int(op_finished_time))

        if self.args.delete_node:
            self.graph.remove_node(job_id, self.jobs[job_id].operations[op_id])

    def log(self, job_id, op_id, machine_id, start_time, process_time):
        op = Operation(None, job_id, None, 0)
        op.selected_machine_id, op.op_id = machine_id, op_id
        op.start_time, op.process_time, op.finish_time = start_time, process_time, start_time + process_time
        self.logger.add_op(op)

    def register_time(self, time):
        index = bisect.bisect_left(self.time_stamp, time)
        if index == len(self.time_stamp) or self.time_stamp[index] != time:
            self.time_stamp.insert(index, time)

    def update_time(self):
        self.current_time = self.time_stamp.pop(0)

        self.check_breakdown_occur()
        self.check_job_arrival()

    def check_breakdown_occur(self):
        for machine in self.machines:
            if machine.breakdown_time == self.current_time:
                #  breakdown occur and interrupt
                if machine.avai_time() > self.current_time:
                    if self.args.breakdown_handler == 'reschedule':
                        self.reschedule(machine)
                    if self.args.breakdown_handler == 'postpone':
                        self.postpone(machine)
                        
                #  breakdown occur without interrupt
                else:
                    machine.process_op(
                        {
                            "job_id"        : -1,
                            "op_id"         : -1,
                            "node_id"       : -1,
                            "start_time"    : machine.breakdown_time,
                            "process_time"  : machine.repair_time
                        }
                    )
                    if self.use_log:
                        raise "not"
                        self.log(-1, 0, machine.machine_id, machine.breakdown_time, machine.repair_time)
                    machine.next_breakdown_event()
                    self.register_time(machine.breakdown_time)
                    self.register_time(machine.breakdown_time + machine.repair_time)

    def reschedule(self, machine):
        # reset machine state
        interrupted_op = machine.processed_op_history.pop()
        job_id, op_id, node_id = interrupted_op['job_id'], interrupted_op['op_id'], interrupted_op['node_id']

        machine.process_op(
            {
                "job_id"        : -1,
                "op_id"         : -1,
                "node_id"       : -1,
                "start_time"    : machine.breakdown_time,
                "process_time"  : machine.repair_time
            }
        )

        # reset job state
        self.jobs[job_id].current_op_id = op_id
        self.jobs[job_id].reset_from(op_id)
        self.jobs[job_id].update_current_op(self.current_time)

        # graph recover operation node
        if self.args.delete_node == True:
            connect_node_ids = list(range(node_id, self.jobs[job_id].operations[-1].node_id + 1))
            self.graph.current_op[job_id] = op_id
            self.graph.add_node(node_id, connect_node_ids, self.jobs[job_id].operations[op_id].machine_and_processtime)
        
        if self.use_log:
            raise "not"
            self.log(-1, 0, machine.machine_id, machine.breakdown_time, machine.repair_time)
        machine.next_breakdown_event()
        self.register_time(machine.breakdown_time)
        self.register_time(machine.breakdown_time + machine.repair_time)

    def postpone(self, machine):
        interrupted_op = machine.processed_op_history.pop()
        job_id, op_id, node_id = interrupted_op['job_id'], interrupted_op['op_id'], interrupted_op['node_id']

        finished_time = machine.process_op(
            {
                "job_id"        : job_id,
                "op_id"         : op_id,
                "node_id"       : node_id,
                "start_time"    : interrupted_op["start_time"],
                "process_time"  : interrupted_op["process_time"] + machine.repair_time
            }
        )

        # reset job state

        # reset the information of interrupted op
        self.jobs[job_id].current_op_id = op_id
        self.jobs[job_id].current_op().update(interrupted_op["start_time"], interrupted_op["process_time"] + machine.repair_time)
        
        # reset the avai time of next operation
        self.jobs[job_id].reset_from(op_id + 1)
        if self.jobs[job_id].next_op() != -1:
            self.jobs[job_id].update_current_op(avai_time=finished_time)
        self.register_time(finished_time)
        
        if self.use_log:
            raise "not"
            self.log(-1, 0, machine.machine_id, machine.breakdown_time, machine.repair_time)
        machine.next_breakdown_event()
        self.register_time(machine.breakdown_time)
        self.register_time(machine.breakdown_time + machine.repair_time)

    def check_job_arrival(self):
        while len(self.unarr_jobs) and self.current_time >= self.unarr_jobs[0].arrival_time:
            self.jobs.append(self.unarr_jobs.popleft())
            self.graph.add_job(self.jobs[-1])
            self.job_num += 1

    def available_ops(self):
        if self.done():
            return None

        avai_ops = []
        avai_mat = np.zeros((self.machine_num, self.job_num),)

        for m in self.machines:
            if m.avai_time() > self.current_time or m.get_status(self.current_time) == BREAKDOWN:
                continue
            for job in self.jobs:
                if job.done() or job.current_op().avai_time > self.current_time:
                    continue

                for mach_ptime in job.current_op().machine_and_processtime:
                    if m.machine_id == mach_ptime[0]:
                        avai_mat[mach_ptime[0]][job.job_id] = mach_ptime[1]

        # for m in machine?
        for i in range(self.job_num):
            avai_m_idx = np.nonzero(avai_mat[:,i])
            if len(avai_m_idx) == 1 and np.count_nonzero(avai_mat[avai_m_idx[0]]) == 1:
                self.assign({
                    'm_id'          : avai_m_idx[0].item(),
                    'job_id'        : i,
                    'op_id'         : self.jobs[i].current_op_id,
                    'node_id'       : self.jobs[i].current_op().node_id,
                    'process_time'  : avai_mat[avai_m_idx[0].item()][i],
                })
                avai_mat[avai_m_idx[0].item()][i] = 0
        
        candidates = np.where(avai_mat > 0)
        avai_ops = [{
            'm_id'          : int(candidates[0][i]),
            'job_id'        : int(candidates[1][i]),
            'op_id'         : self.jobs[candidates[1][i]].current_op_id,
            'node_id'       : self.jobs[candidates[1][i]].current_op().node_id,
            'process_time'  : avai_mat[candidates[0][i]][candidates[1][i]]
        } for i in range(len(candidates[0]))]
        
        if not avai_ops:
            self.update_time()
            return self.available_ops()
        return avai_ops
        
    def get_max_process_time(self):
        return self.graph.get_max_process_time()
