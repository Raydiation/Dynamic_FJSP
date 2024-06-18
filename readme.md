# Machine breakdown
While resetting the environment, each machine get 2 parameters: failure_dist ~ uniform(MTBF) & repair_dist ~ uniform(MTTR)

Each machine will determine the next time stamp of breakdown and the repair time (model will not know this information). While there are some operation be executed and the process overlap the breakdown interval, it will increase the process time by the "repair time", then this machine will determine the next time stamp of breakdown and the repair time.