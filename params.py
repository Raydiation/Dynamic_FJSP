import argparse

def get_args():
    parser = argparse.ArgumentParser(description='Arguments for RL_GNN_DJSP')
    # args for normal setting
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--use_log', type=bool, default=False)
    # args for env
    parser.add_argument('--reschedule', type=bool, default=False)
    parser.add_argument('--instance_type', type=str, default='FJSP')
    parser.add_argument('--ini_job_num', type=int, default=15)
    parser.add_argument('--machine_num', type=int, default=8)
    parser.add_argument('--max_process_time', type=int, default=10, help='Maximum Process Time of an Operation')
    parser.add_argument('--delete_node', type=bool, default=False)
    parser.add_argument('--train_arr', type=bool, default=False)
    parser.add_argument('--train_break', type=bool, default=False)

    parser.add_argument('--DDT', type=float, default=1)

    # args for new job arrival
    parser.add_argument('--new_job_event', type=int, default=2, help="the number of new job arrival event")
    parser.add_argument('--new_job_per_num', type=int, default=5, help="the number of arrival job in each new job arrival event")
    parser.add_argument('--arrival_time_dist', type=int, default=20, help='arrival time distribution between two job')

    # args for machine breakdown
    parser.add_argument('--MTBF', type=list, default=[50, 70], help='mean time between failure, T~exp(M_mtbf)')
    parser.add_argument('--MTTR', type=list, default=[10, 20], help='mean time to repair, T~exp(M_mttr)')

    # args for RL
    parser.add_argument('--entropy_coef', type=float, default=1e-2)
    parser.add_argument('--episode', type=int, default=200001)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--step_size', type=float, default=1000)
    # args for policy network
    parser.add_argument('--hidden_dim', type=int, default=256) #256
    parser.add_argument('--T', type=float, default=1) #256
    # args for GNN
    parser.add_argument('--GNN_num_layers', type=int, default=3)
    parser.add_argument('--act', type=str, default='leaky_relu')
    # args for policy
    parser.add_argument('--policy_num_layers', type=int, default=2)
    
    # args for nameing
    parser.add_argument('--date', type=str, default='test')
    parser.add_argument('--detail', type=str, default="no")
    parser.add_argument('--rule', type=str, default='EDD_SPT_rng')

    # args for val/test
    parser.add_argument('--valid_sets', type=str, default=None, help="split by ,")
    parser.add_argument('--test_sets', type=str, default=None, help="split by ,")
    parser.add_argument('--test_sample_times', type=int, default=1)


    parser.add_argument('--block_breakdown', type=bool, default=False)

    args = parser.parse_args()
    return args
