import numpy as np
from lib.Data import Data
from lib.configs import config_dict
from lib.utils import Model
from lib.Constants import POLICY_UPDATE_INTERVAL, WARMUP_TIME_SECONDS, T_TOTAL_SECONDS, INT_ASSIGN, \
    ANALYSIS_TIME_SECONDS, DEMAND_UPDATE_INTERVAL
import time
data_instance = Data.init_from_config_dic(config_dict)
m = Model(data_instance)

print('Fleet size is {f}'.format(f=data_instance.FLEET_SIZE))

stime = time.time()

# # dispatch the system for T_TOTAL seconds, at the interval of INT_ASSIGN
for T in range(data_instance.WARMUP_TIME_SECONDS,
               data_instance.T_TOTAL_SECONDS,
               data_instance.INT_ASSIGN):
    m.dispatch_at_time(T)

# end time
etime = time.time()
# run time of this simulation
runtime = etime - stime
print("The run time was {runtime} minutes ".format(runtime=runtime / 60))

m.save_zonal_stats("../performance_stats/")