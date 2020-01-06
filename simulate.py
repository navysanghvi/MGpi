# sim_interact for static scenario, sim_interact_dyn for dynamic scenario
from sim_interact_dyn import GroupInteract
import agent_sim
from ipy_progressbar import ProgressBarTerminal as PB
from joblib import Parallel, delayed
from distutils import dir_util as DU
import numpy as np
import cPickle

# Group pose and gaze data for simulation
# data_path = 'interaction_data/interaction_data/synth_data.npy'
# data_path = 'interaction_data/interaction_data/cocktailparty.npy'
data_path = 'interaction_data/interaction_data/coffee_data.npy'
syn_data = np.load(data_path)

# Calibration for data
# calib = 1
# calib_path = 'interaction_data/interaction_data/calib_cocktail.npy'
calib_path = 'interaction_data/interaction_data/calib_coffee.npy'
calib = np.load(calib_path)


# Make save directory
save_path = 'simulations_dyn/'
DU.mkpath(save_path)

# Number of interaction steps to simulate
grp_steps = 300

# Number of datasets
n_syn = syn_data.shape[0]

# Simulate interaction from one synthetic dataset
def sim_data(i_syn):
    syn = syn_data[i_syn]
    gInt = GroupInteract(start_pos=syn[:,:2]*calib, start_gaze=syn[:,2:4], gID=syn[:,-1])
    gInt.run_all(epi_len=grp_steps)
    cPickle.dump(gInt, open(save_path+'group_%03d.p' % i_syn, 'wb'))

# Simulate interactions from all datasets in parallel
Parallel(n_jobs=4)(delayed(sim_data)(i_syn) 
                   for i_syn in PB(range(n_syn)))

# Random dataset to simulate and visualize
# syn = syn_data[10]
# gInt = GroupInteract(start_pos=syn[:,:2], start_gaze=syn[:,2:4], gID=syn[:,-1])
# gInt.run_all(epi_len=grp_steps)
# gInt.visualize()