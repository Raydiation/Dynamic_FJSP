import numpy as np
import random

MAX = 1e6

def heuristic_tardiness(env, avai_ops, rule):

    job_num_lis = []

    if rule == "MOR":
        # job_num_lis = [sum(job.done() != 1 for job in env.jsp_instance.jobs)]
        while True:
            action_idx = MOR(avai_ops, env.jsp_instance.jobs)
            avai_ops, done = env.step(avai_ops[action_idx])
            if done:
                return env.get_tardiness()
    if rule == "FIFO":
         while True:
            action_idx = FIFO(avai_ops, env.jsp_instance.jobs)
            avai_ops, done = env.step(avai_ops[action_idx])
            if done:
                return env.get_tardiness()
    if rule == "SPT":
        while True:
            action_idx = SPT(avai_ops)
            avai_ops, done = env.step(avai_ops[action_idx])
            job_num_lis.append((sum(job.done() != 1 for job in env.jsp_instance.jobs), env.jsp_instance.current_time))
            if done:
                return job_num_lis, env.get_tardiness()
                # return env.get_tardiness()
            
    if rule == "MWKR":
        while True:
            action_idx = MWKR(avai_ops, env.jsp_instance.jobs)
            avai_ops, done = env.step(avai_ops[action_idx])
            if done:
                return env.get_tardiness()

    if rule == "EDD":
        while True:
            action_idx = EDD(avai_ops, env.jsp_instance.jobs)
            avai_ops, done = env.step(avai_ops[action_idx])
            # job_num_lis.append((sum(job.done() != 1 for job in env.jsp_instance.jobs), env.jsp_instance.current_time))
            if done:
                # return job_num_lis, env.get_tardiness()
                return env.get_tardiness()

    if rule == "EDD_SPT_min":
        while True:
            action_idx = EDD_SPT_min(avai_ops, env.jsp_instance.jobs)
            avai_ops, done = env.step(avai_ops[action_idx])
            job_num_lis.append((sum(job.done() != 1 for job in env.jsp_instance.jobs), env.jsp_instance.current_time))
            if done:
                return job_num_lis, env.get_tardiness()

    if rule == "EDD_SPT_rng":
        while True:
            action_idx = EDD_SPT_rng(avai_ops, env.jsp_instance.jobs)
            # print(avai_ops[action_idx]['process_time'])
            avai_ops, done = env.step(avai_ops[action_idx])
            # job_num_lis.append((sum(job.done() != 1 for job in env.jsp_instance.jobs), env.jsp_instance.current_time))
            if done:
                # return job_num_lis, env.get_tardiness()
                return env.get_tardiness()

    if rule == "SRPT":
        while True:
            action_idx = SRPT(avai_ops, env.jsp_instance.jobs, env.jsp_instance.current_time)
            avai_ops, done = env.step(avai_ops[action_idx])
            if done:
                return env.get_tardiness()

    if rule == "STPT":
         while True:
            action_idx = STPT(avai_ops, env.jsp_instance.jobs)
            avai_ops, done = env.step(avai_ops[action_idx])
            if done:
                return env.get_tardiness()

    if rule == "LTPT":
         while True:
            action_idx = LTPT(avai_ops, env.jsp_instance.jobs)
            avai_ops, done = env.step(avai_ops[action_idx])
            if done:
                return env.get_tardiness()


def heuristic_makespan(env, avai_ops, rule):
    if rule == "MOR":
        while True:
            action_idx = MOR(avai_ops, env.jsp_instance.jobs)
            avai_ops, done = env.step(avai_ops[action_idx])
            if done:
                return env.get_makespan()
    if rule == "FIFO":
         while True:
            action_idx = FIFO(avai_ops, env.jsp_instance.jobs)
            avai_ops, done = env.step(avai_ops[action_idx])
            if done:
                return env.get_makespan()
    if rule == "SPT":
         while True:
            action_idx = SPT(avai_ops)
            avai_ops, done = env.step(avai_ops[action_idx])
            if done:
                return env.get_makespan()
    if rule == "MWKR":
        while True:
            action_idx = MWKR(avai_ops, env.jsp_instance.jobs)
            avai_ops, done = env.step(avai_ops[action_idx])
            if done:
                return env.get_makespan()

    if rule == "EDD":
        while True:
            action_idx = EDD(avai_ops, env.jsp_instance.jobs)
            avai_ops, done = env.step(avai_ops[action_idx])
            if done:
                return env.get_makespan()

    if rule == "SRPT":
        while True:
            action_idx = SRPT(avai_ops, env.jsp_instance.jobs, env.jsp_instance.current_time)
            avai_ops, done = env.step(avai_ops[action_idx])
            if done:
                return env.get_makespan()

def rollout(env, avai_ops):
    epsilon = 0.1
    while True:
        magic_num = random.random()
        if magic_num < epsilon:
            action_idx = Random(avai_ops)
        else:
            action_idx = MOR(avai_ops, env.jsp_instance.jobs)
        avai_ops, done = env.step(avai_ops, action_idx)
        if done:
            return env.get_makespan()

def Random(avai_ops):
    return np.random.choice(len(avai_ops), size=1)[0]

def MOR(avai_ops, jobs):
    max_remaining_op = -1
    action_idx = -1

    for i in range(len(avai_ops)):
        op_info = avai_ops[i]
        job = jobs[op_info['job_id']]

        if len(job.operations) - op_info['op_id'] >= max_remaining_op:
            action_idx = i
            max_remaining_op = len(job.operations) - op_info['op_id']
            
    return action_idx

def MWKR(avai_ops, jobs):
    action_idx = -1
    max_work_remaining = -1
    
    for i in range(len(avai_ops)):
        op_info = avai_ops[i]
        job = jobs[op_info['job_id']]
        if job.acc_expected_process_time[op_info['op_id']] > max_work_remaining:
            max_work_remaining = job.acc_expected_process_time[op_info['op_id']]
            action_idx = i
            
    return action_idx


def FIFO(avai_ops, jobs):
    min_avai_time = MAX
    action_idx = -1
    for i in range(len(avai_ops)): 
        op_info = avai_ops[i]
        op = jobs[op_info['job_id']].operations[op_info['op_id']]

        if op.avai_time < min_avai_time:
            action_idx = i
            min_avai_time = op.avai_time

    return action_idx


def SPT(avai_ops):
    min_process_time = MAX
    action_idx = -1
    for i in range(len(avai_ops)):
        op_info = avai_ops[i]
        if op_info['process_time'] < min_process_time:
            action_idx = i
            min_process_time = op_info['process_time']

    return action_idx

def EDD(avai_ops, jobs):
    earliest_due_date = MAX
    action_idx = -1
    for i in range(len(avai_ops)): 
        if jobs[avai_ops[i]['job_id']].due_date < earliest_due_date:
            earliest_due_date = jobs[avai_ops[i]['job_id']].due_date
            action_idx = i
        # elif jobs[avai_ops[i]['job_id']].due_date == earliest_due_date:
        #     if avai_ops[i]['process_time'] < avai_ops[action_idx]['process_time']:
        #         action_idx = i

    return action_idx

def EDD_SPT_rng(avai_ops, jobs):
    earliest_due_date = MAX
    action_idxs = []
    for i in range(len(avai_ops)): 
        if jobs[avai_ops[i]['job_id']].due_date < earliest_due_date:
            earliest_due_date = jobs[avai_ops[i]['job_id']].due_date
            action_idxs = [i]
        elif jobs[avai_ops[i]['job_id']].due_date == earliest_due_date:
            action_idxs.append(i)

    return random.choice(action_idxs)

def EDD_SPT_min(avai_ops, jobs):
    earliest_due_date = MAX
    action_idx = -1
    for i in range(len(avai_ops)): 
        if jobs[avai_ops[i]['job_id']].due_date < earliest_due_date:
            earliest_due_date = jobs[avai_ops[i]['job_id']].due_date
            action_idx = i
        elif jobs[avai_ops[i]['job_id']].due_date == earliest_due_date:
            if avai_ops[i]['process_time'] < avai_ops[action_idx]['process_time']:
                action_idx = i

    return action_idx

def SRPT(avai_ops, jobs, current_time):
    action_idx = -1
    min_srpt = 1e6
    for i in range(len(avai_ops)):
        job = jobs[avai_ops[i]['job_id']]
        rpt = job.acc_expected_process_time[job.current_op_id]
        srpt = (job.due_date - current_time - rpt) / rpt
        if srpt < min_srpt:
            min_srpt = srpt
            action_idx = i
    return action_idx

def STPT(avai_ops, jobs):
    min_total_process_time = MAX
    action_idx = -1
    for i in range(len(avai_ops)):
        job = jobs[avai_ops[i]['job_id']]
        if job.acc_expected_process_time[0] < min_total_process_time:
            action_idx = i
            min_total_process_time = job.acc_expected_process_time[0]

    return action_idx

def LTPT(avai_ops, jobs):
    max_total_process_time = 0
    action_idx = -1
    for i in range(len(avai_ops)):
        job = jobs[avai_ops[i]['job_id']]
        if job.acc_expected_process_time[0] > max_total_process_time:
            action_idx = i
            max_total_process_time = job.acc_expected_process_time[0]

    return action_idx
