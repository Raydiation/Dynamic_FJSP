import numpy as np
import torch
from torch_geometric.data import HeteroData
from heuristic import MAX
import time
import bisect

BREAKDOWN = -1
AVAILABLE = 0
PROCESSED = 1
COMPLETE = 3
FUTURE = 2

def binary_search(list_, target):
    left, right = 0, len(list_)
    pos = bisect_left(list_, target, left, right)
    return pos if pos != right and list_[pos] == target else -1
    
class Graph:
    def __init__(self, args, machine_num):
        self.op_op_edge_src_idx = torch.empty(size=(1,0), dtype=torch.int64)                    # for op<->op
        self.op_op_edge_tar_idx = torch.empty(size=(1,0), dtype=torch.int64)                    # for op<->op
        self.op_edge_idx = torch.empty(size=(1,0), dtype=torch.int64)                           # for op<->m
        self.m_edge_idx = torch.empty(size=(1,0), dtype=torch.int64)                            # for op<->m
        self.m_m_edge_idx = torch.tensor([[i for i in range(machine_num)]], dtype=torch.int64)  # for m<->m
        self.edge_x = torch.empty(size=(1,0), dtype=torch.int64)

        self.op_x = []
        self.m_x = []
        self.job_srpt = []  # job slack time / job remaining process time

        self.args = args
        self.machine_num = machine_num
        self.op_num = 0
        self.op_unfinished = []
        self.current_op = []  # MIN(NOT complete op id) in JOB J

        self.max_process_time = 0

    def get_data(self):
        data = HeteroData()

        data['op'].x    = torch.FloatTensor(self.op_x)
        data['m'].x     = torch.FloatTensor(self.m_x)

        data['op', 'to', 'op'].edge_index   = torch.cat((self.op_op_edge_src_idx, self.op_op_edge_tar_idx), dim=0).contiguous()
        data['op', 'to', 'm'].edge_index    = torch.cat((self.op_edge_idx, self.m_edge_idx), dim=0).contiguous()
        data['m', 'to', 'op'].edge_index    = torch.cat((self.m_edge_idx, self.op_edge_idx), dim=0).contiguous()
        data['m', 'to', 'm'].edge_index     = torch.cat((self.m_m_edge_idx, self.m_m_edge_idx), dim=0).contiguous()

        return data, self.op_unfinished, self.job_srpt
       
    def add_job(self, job):
        src, tar = self.fully_connect(self.op_num, job.op_num)
        self.op_op_edge_src_idx = torch.cat((self.op_op_edge_src_idx, src.unsqueeze(0)), dim=1)
        self.op_op_edge_tar_idx = torch.cat((self.op_op_edge_tar_idx, tar.unsqueeze(0)), dim=1)
        self.current_op.append(0)

       for i in range(job.op_num):
            job.operations[i].node_id = self.op_num # set index of an op in the graph
            op = job.operations[i]
            self.op_edge_idx    = torch.cat((self.op_edge_idx,  torch.tensor([[self.op_num for _ in range(len(op.machine_and_processtime))]])), dim=1)
            self.m_edge_idx     = torch.cat((self.m_edge_idx,   torch.tensor([[machine_and_processtime[0] for machine_and_processtime in op.machine_and_processtime]])), dim=1)
            self.edge_x         = torch.cat((self.edge_x,       torch.tensor([[machine_and_processtime[1] for machine_and_processtime in op.machine_and_processtime]])), dim=1)

            self.op_unfinished.append(self.op_num)
            self.op_num += 1


    def update_feature(self, jobs, machines, current_time, delete_node_only=False):
        # delete node
        self.update_graph()
        if self.args.delete_node == True:
            for i in range(len(jobs)):
                cur = self.current_op[i]
                for j in range(cur, len(jobs[i].operations)):
                    op = jobs[i].operations[j]

                    status = op.get_status(current_time)

                    # if status == COMPLETE:
                    if status == PROCESSED or status == COMPLETE:
                        idx = bisect.bisect_left(self.op_unfinished, op.node_id)
        
                        if idx == len(self.op_unfinished) or self.op_unfinished[idx] != op.node_id:
                            raise "abnormal idx"

                        self.update_graph(idx)
                        self.op_unfinished.remove(op.node_id)
                        self.current_op[i] += 1
                    else:
                        break
        if delete_node_only:
            return

        self.op_x, self.m_x, self.job_srpt = [], [], []
        self.max_process_time = self.get_max_process_time()

        for i in range(len(jobs)):
            job = jobs[i]
            for j in range(self.current_op[i], len(job.operations)):
                op = job.operations[j]

                status = op.get_status(current_time)

                # status
                if self.args.delete_node == True:
                    feat = [0] * 2
                    feat[status // 2] = 1
                else:
                    feat = [0] * 4
                    feat[status] = 1

                feat.append(op.expected_process_time / self.max_process_time)

                # # 1013_3 try waiting time?
                if status == AVAILABLE:
                    feat.append((current_time - op.avai_time) / self.max_process_time)
                else:
                    feat.append(0)

                #MWKR
                feat.append(job.acc_expected_process_time[op.op_id] / job.acc_expected_process_time[0])

                self.op_x.append(feat) 
        # machine feature
        for m in machines:
            feat = [0] * 2
            # status : [is_AVAIABE, is_PROCESSED]
            status = m.get_status(current_time)
            feat[status] = 1
            # time to available
            if status == AVAILABLE:
                feat.append(0)
            elif status == BREAKDOWN: # for reschedule, if append, use avai time.
                feat.append((m.breakdown_time + m.repair_dist) / self.max_process_time)
            else:
                feat.append((m.avai_time() - current_time) / self.max_process_time)

            # waiting time
            if status == AVAILABLE:
                feat.append((current_time - m.avai_time()) / self.max_process_time)
            else:
                feat.append(0)
            
            self.m_x.append(feat)

        # SRPT slack time
        for i in range(len(jobs)):
            if jobs[i].current_op_id == -1:
                self.job_srpt.append(0)
                continue
            rpt = jobs[i].acc_expected_process_time[jobs[i].current_op_id]
            self.job_srpt.append((jobs[i].due_date - current_time - rpt) / (rpt * self.args.max_process_time))

        self.job_srpt = torch.Tensor(self.job_srpt)

    def update_graph(self, idx=None):
        if idx is not None:
            src_idxs = np.where(self.op_op_edge_src_idx == idx)
            self.op_op_edge_src_idx = np.delete(self.op_op_edge_src_idx, src_idxs)
            self.op_op_edge_tar_idx = np.delete(self.op_op_edge_tar_idx, src_idxs)

            tar_idxs = np.where(self.op_op_edge_tar_idx == idx)
            self.op_op_edge_src_idx = np.delete(self.op_op_edge_src_idx, tar_idxs)
            self.op_op_edge_tar_idx = np.delete(self.op_op_edge_tar_idx, tar_idxs)

            #op-m, m-op
            idxs = np.where(self.op_edge_idx == idx)
            self.op_edge_idx = np.delete(self.op_edge_idx, idxs)
            self.m_edge_idx = np.delete(self.m_edge_idx, idxs)
            self.edge_x = np.delete(self.edge_x, idxs)

        _, self.op_edge_idx = np.unique(self.op_edge_idx, return_inverse=True)
        _, self.op_op_edge_src_idx = np.unique(self.op_op_edge_src_idx, return_inverse=True)
        _, self.op_op_edge_tar_idx = np.unique(self.op_op_edge_tar_idx, return_inverse=True)


    def fully_connect(self, begin, size):
        adj_matrix = np.ones((size, size),)
        idxs = np.where(adj_matrix > 0)
        edge_index = np.stack((idxs[0] + begin, idxs[1] + begin))
        return edge_index[0], edge_index[1]

    def get_max_process_time(self):
        # return np.max(self.edge_x.numpy())
        return np.max(self.edge_x)

    def add_node(self, node_id, connect_op, connect_m_and_pt):
        idx = bisect.bisect_left(self.op_unfinished, node_id)
        self.op_unfinished.insert(idx, node_id)
        connect_op = [bisect.bisect_left(self.op_unfinished, _id) for _id in connect_op]

        for i in range(len(self.op_edge_idx)):
            if self.op_edge_idx[i] >= idx:
                self.op_edge_idx[i] += 1

        for i in range(len(self.op_op_edge_src_idx)):
            if self.op_op_edge_src_idx[i] >= idx:
                self.op_op_edge_src_idx[i] += 1
            if self.op_op_edge_tar_idx[i] >= idx:
                self.op_op_edge_tar_idx[i] += 1
        
        self.op_edge_idx = np.append(self.op_edge_idx, [idx] * len(connect_m_and_pt))
        self.m_edge_idx = np.append(self.m_edge_idx, [pair[0] for pair in connect_m_and_pt])
        self.edge_x = np.append(self.edge_x, [pair[1] for pair in connect_m_and_pt])

        self.op_op_edge_src_idx = np.append(self.op_op_edge_src_idx, [idx] * (len(connect_op) - 1))
        self.op_op_edge_src_idx = np.append(self.op_op_edge_src_idx, connect_op)
        self.op_op_edge_tar_idx = np.append(self.op_op_edge_tar_idx, connect_op)
        self.op_op_edge_tar_idx = np.append(self.op_op_edge_tar_idx, [idx] * (len(connect_op) - 1))
