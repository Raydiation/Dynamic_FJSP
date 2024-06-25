import numpy as np
import random
import os
import time
# random.seed(6404)
# np.random.seed(6404)

from itertools import accumulate

# FJSP
def gen_operations_FJSP(machine_num, op_process_time_range): 
    op = []
    
    op_num = random.randint(1, machine_num + 2)
    # op_num = random.randint(2, 4)

    for op_id in range(op_num):
        random_size = random.randint(1, machine_num) # the number of usable machine for this operation
        # random_size = random.randint(1, 3) # the number of usable machine for this operation
        m_id = sorted(np.random.choice(machine_num, size=random_size, replace=False)) # the set of index of usable machine id with size random_size
        mach_ptime = []
        for id in m_id:
            process_time = np.random.randint(*op_process_time_range)
            mach_ptime.append((id, process_time))
        op.append({"id": op_id, "machine_and_processtime": mach_ptime})
    # print(op)
    return op

def gen_instance_FJSP(config):

    dir_name = './({}+{})x{}_TEST'.format(config['ini_job_num'], config['new_job_events'] * config['new_job_per_num'], config['machine_num'])
    if config['job_arrival']:
        dir_name += '_newjob'
    if config['machine_breakdown']:
        dir_name += '_breakdown'
    os.makedirs(dir_name, exist_ok=True)

    for case in range(config['task_num']):
        if config['job_arrival'] is True:
            random_arrival_time = np.random.exponential(scale=config['job_arrival_time_dist'], size=config['new_job_events']) + 1
            job_arrival_time = list(accumulate(map(int, random_arrival_time)))

        cnt = 0.0
        op_num = 0.0
        job_configs = []
        configs = []
        for i in range(config['ini_job_num'] + config['new_job_events'] * config['new_job_per_num']):
            job_conf = gen_operations_FJSP(config['machine_num'], config['process_time_range'])
            op_num += len(job_conf)
            job_configs.append(job_conf)
            for op in job_conf:
                cnt += len(op['machine_and_processtime'])


        with open('{}/{}.fjs'.format(dir_name, case), "w") as outfile:
            outfile.write('{} {} {}\n'.format(config['ini_job_num'], config['new_job_events'] * config['new_job_per_num'], config['machine_num']))

            for idx, job_conf in enumerate(job_configs):
                if idx < config['ini_job_num']:
                    outfile.write('{} {} '.format(0, len(job_conf)))
                else:
                    outfile.write('{} {} '.format(job_arrival_time[(idx - config['ini_job_num']) // config['new_job_per_num']], len(job_conf)))

                for op in job_conf:
                    outfile.write('{} '.format(len(op['machine_and_processtime'])))
                    for pair in op['machine_and_processtime']:
                        outfile.write('{} {} '.format(pair[0] + 1, pair[1]))
                outfile.write('\n')

        if config['machine_breakdown'] is True:
            machine_breakdown_events = list()
            for idx in range(config['machine_num']):
                failure_dist = np.random.randint(*config['MTBF'])  # freq
                repair_dist = np.random.randint(*config['MTTR'])  # length
                breakdown_event = list()
                breakdown_time, repair_time = 0, 0
                for _ in range(100):
                    breakdown_time += repair_time + int(np.random.exponential(scale=failure_dist)) + 1
                    repair_time = int(np.random.exponential(scale=repair_dist)) + 1
                    breakdown_event.append((breakdown_time, repair_time))
                machine_breakdown_events.append(breakdown_event)

            with open('{}/{}.fjs'.format(dir_name, case), "a") as outfile:
                outfile.write('BREAKDOWN EVENTS:\n')
                for breakdown_event in machine_breakdown_events:
                    outfile.write('{}\n'.format(breakdown_event))

if __name__ == '__main__':

    config = {
        'ini_job_num' : 15,
        'new_job_events' : 2,
        'new_job_per_num' : 5,
        'machine_num' : 8,
        'process_time_range' : [1,10],
        'job_arrival' : True,
        'machine_breakdown' : True,
        'job_arrival_time_dist' : 20,
        'MTBF' : [50, 70],
        'MTTR' : [10, 20],
        'task_num' : 100
    }

    # basic
    gen_instance_FJSP(config)
    # for arrival_time_dist in arrival_time_dists:
    #     for new_job_num in new_job_nums:
    #         gen_instance_FJSP_job_insert(ini_job_num, new_job_num, machine_num, arrival_time_dist)
   
