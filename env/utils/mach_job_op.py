import numpy as np
import random
from collections import deque
MAX = 1e6
from itertools import accumulate

BREAKDOWN = -1
AVAILABLE, PROCESSED, FUTURE, COMPLETED = 0, 1, 2, 3

class Machine:
    def __init__(self, machine_id, args):
        self.machine_id = machine_id
        self.processed_op_history = []

        self.failure_dist = np.random.randint(*args.MTBF)  # freq
        self.repair_dist = np.random.randint(*args.MTTR)  # length

        self.breakdown_event = deque()
        self.breakdown_time, self.repair_time = MAX, 0
    
    def process_op(self, op_info):
        machine_avai_time = self.avai_time()
        start_time = max(op_info["start_time"], machine_avai_time)
        op_info["start_time"] = start_time
        finished_time = start_time + op_info["process_time"]
        self.processed_op_history.append(op_info)
        return finished_time
        
    def avai_time(self):
        if len(self.processed_op_history) == 0:
            return 0
        else:
            return self.processed_op_history[-1]["start_time"] + self.processed_op_history[-1]["process_time"]

    def get_status(self, current_time):
        if self.breakdown_time <= current_time and current_time < self.breakdown_time + self.repair_time:
            return BREAKDOWN
        return AVAILABLE if current_time >= self.avai_time() else PROCESSED

    def generate_breakdown(self):
        breakdown_time, repair_time = 0, 0
        for _ in range(100):
            breakdown_time += repair_time + int(np.random.exponential(scale=self.failure_dist)) + 1
            repair_time = int(np.random.exponential(scale=self.repair_dist)) + 1
            self.breakdown_event.append((breakdown_time, repair_time))

        self.breakdown_time = self.breakdown_event[0][0]
        self.repair_time = self.breakdown_event[0][1]
        
    def next_breakdown_event(self):
        if len(self.breakdown_event) > 0:
            self.breakdown_event.popleft()

        if len(self.breakdown_event) == 0:
            self.breakdown_time = MAX
            self.repair_time = MAX
            return
            # raise "No event"
        self.breakdown_time, self.repair_time = self.breakdown_event[0]

class Job:
    def __init__(self, args, job_id, op_config, arrival_time):
        self.args, self.job_id, self.op_num, self.arrival_time = args, job_id, len(op_config), arrival_time
        self.operations = [Operation(self.args, self.job_id, config, arrival_time) for config in op_config]
        self.current_op_id = 0 # ready to be processed

        self.acc_expected_process_time = list(accumulate([op.expected_process_time for op in self.operations[::-1]]))[::-1]
        self.due_date = int(arrival_time + self.acc_expected_process_time[0] * args.DDT)

    def current_op(self):
        return self.operations[self.current_op_id] if self.current_op_id != -1 else None
    
    def update_current_op(self, avai_time):
        self.operations[self.current_op_id].avai_time = avai_time 
    
    def next_op(self):
        if self.current_op_id + 1 < self.op_num:
            self.current_op_id += 1
        else:
            self.current_op_id = -1
        return self.current_op_id
    
    def done(self):
        return self.current_op_id == -1

    def reset_from(self, op_id):
        for o_id in range(op_id, len(self.operations)):
            if self.operations[o_id].avai_time == MAX:
                break
            self.operations[o_id].reset()

class Operation:
    def __init__(self, args, job_id, config, arrival_time):
        self.args, self.job_id, self.op_id = args, job_id, None
        if config is not None:
            self.op_id, self.machine_and_processtime = config['id'], config['machine_and_processtime']
            self.avai_time = arrival_time if self.op_id == 0 else MAX
            self.expected_process_time = sum(pair[1] for pair in self.machine_and_processtime) / len(self.machine_and_processtime)

        self.node_id, self.start_time, self.finish_time = -1, -1, -1
        self.selected_machine_id, self.process_time = -1, -1 #for logger

    def update(self, start_time, process_time):
        self.start_time, self.finish_time = start_time, start_time + process_time
    
    def get_status(self, current_time):
        # Before processed
        if self.start_time == -1:
            return AVAILABLE if current_time >= self.avai_time else FUTURE
            
        # After processed
        return COMPLETED if current_time >= self.finish_time else PROCESSED

    def reset(self):
        self.avai_time, self.start_time, self.finish_time = MAX, -1, -1
        self.selected_machine_id, self.process_time = -1, -1

