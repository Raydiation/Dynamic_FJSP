import numpy as np
import torch
from params import get_args
from env.env import JSP_Env
from model.REINFORCE import REINFORCE
import torch.nn.functional as F
import time
import json
import os
from heuristic import *
import copy

def test(test_sets=None):

    if args.instance_type == 'FJSP':
        test_dir = './datasets/DFJSP/Base_mk04'
        if test_sets is None:
            test_sets = [
                            # 'seed_2011_newjob_Tarr=20_breakdown_Tbreak=[60, 80]',
                            # 'seed_8039_newjob_Tarr=20_breakdown_Tbreak=[40, 60]',
                            # 'seed_8914_newjob_Tarr=15_breakdown',
                            # 'seed_6404_newjob_Tarr=20_breakdown',
                            # 'seed_1468_newjob_Tarr=25_breakdown',
                            'seed_1855_newjob_Tarr=30_breakdown',
                        ]

    else:
        test_dir = './datasets/DJSP'
        if test_sets is None:
            test_sets = ['(10+20)x10_DJSP']

    os.makedirs('./result/{}'.format(args.date), exist_ok=True)

    for _set in test_sets:
        for size in sorted(os.listdir(os.path.join(test_dir, _set))):
            size_set = os.path.join(test_dir, _set, size)
            for instance in sorted(os.listdir(size_set)):
                best_tard = 1e6
                file = os.path.join(size_set, instance)

                if args.test_sample_times > 1 :
                    # apex
                    N = args.test_sample_times
                    alpha, epsilon = 7, 0.4
                    apex = np.array([epsilon ** (1 + i / (N - 1) * alpha) for i in range(N)])
                    apex /= sum(apex)
                    all_T = np.random.choice(sorted(np.random.uniform(0, 1.0, size=N)), N, p=apex)

                for cnt in range(args.test_sample_times):

                    avai_ops = env.load_instance(file)

                    while True:
                        data, op_unfinished, job_srpt= env.get_graph_data()
                        if cnt == 0:
                            action_idx, action_prob = policy(avai_ops, data, op_unfinished, job_srpt, env.jsp_instance.graph.max_process_time, greedy=True)
                        else:
                            # action_idx, action_prob = policy(avai_ops, data, op_unfinished, job_srpt, env.jsp_instance.graph.max_process_time, greedy=True)
                        avai_ops, done = env.step(avai_ops[action_idx])
                        
                        if done:
                            best_tard = min(best_tard, env.get_tardiness())

                            print("instance : {}, tard : {}".format(file, env.get_tardiness()))
                            break
                with open('./result/{}/test_result.txt'.format(args.date),"a") as outfile:
                    outfile.write(f'instance : {file:50} tard : {best_tard:10} \n')

if __name__ == '__main__':
    args = get_args()
    print(args)
    env = JSP_Env(args)
    policy = REINFORCE(args).to(args.device)
    
    policy.load_state_dict(torch.load('./weight/{}/154000'.format(args.date), map_location=args.device), False)
    with torch.no_grad():
        test()
                    