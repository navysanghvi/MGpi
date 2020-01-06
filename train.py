from train_model import CommModels
from glob import glob
import numpy as np
import random
import cPickle
import sys
import os


# In `group' in each file:
# [pos]   : positions' matrix size        (number of agents x space dimension (i.e., 2) x time)
# [gaze]  : gaze directions' matrix size  (number of agents x space dimension (i.e., 2) x time)
# [state] : comm. modes' matrix size      (number of agents x modes dimension (i.e., 6) x time)



### adds noise to data for groups from each file
def data_noise(group, pos_noise = 0, gaze_noise = 0):
    pos = group.pos; gaze = group.gaze; modes = group.mode                      # original non-verbal data
    pos += np.random.normal(scale = pos_noise, size = pos.shape)                # noise in position data
    gaze += np.random.normal(scale = gaze_noise, size = gaze.shape)             # noise in gaze data
    gaze /= np.linalg.norm(gaze, axis = 1)[:, np.newaxis, :]                    # renormalize noisy gaze data
    return pos, gaze, modes



### Computes and prepares data for each agent
def data_per_agt(pos, gaze, modes, agt, num_agents, 
                 n_neighbors, agt_subset = 'pos_neigh'):
    
    rel_pos_all = (pos - pos[agt])                                              # all relative poses
    
    # Differrent neighbor choice strategies
    if(agt_subset == 'pos_neigh'):
        mean_dists = np.linalg.norm(rel_pos_all, axis=1).mean(axis=1)           # mean relative distances
        neighs = np.argsort(mean_dists)[1:n_neighbors+1]                        # select neighbors

    rel_pos_neigh = rel_pos_all[neighs]                                         # neighbor rel. poses
    gaze_neigh = gaze[neighs]                                                   # neighbor gazes

    rel_pos_neigh.shape = (len(neighs),) + pos.shape[1:]                        # make sure of correct shapes..
    gaze_neigh.shape = (len(neighs),) + gaze.shape[1:]                          # ..in case of single neighbor

    th = np.arctan2(gaze[agt,0], gaze[agt,1])                                   # agt gaze's rotation angle
    rot = np.array([[np.cos(th), -np.sin(th)], 
                    [np.sin(th), np.cos(th)]])                                  # CCW rotation matrix
    rot_pos = np.einsum('ijt,kjt->kit', rot, rel_pos_neigh)                     # neigh. rotated rel. poses
    rel_gaze = np.einsum('ijt,kjt->kit', rot, gaze_neigh)                       # neigh. relative gazes
    modes_neigh = modes[neighs]                                                 # neighs' mode histories
    modes_self = modes[agt]                                                     # agt's mode history

    modes_neigh.shape = (len(neighs),) + modes.shape[1:]                        # ..in case of single neighbor

    return rot_pos, rel_gaze, modes_neigh, modes_self    


def noisy(modes, mode_noise):
    if(not mode_noise): return modes
    res = modes.copy()
    all_m = np.arange(len(modes))
    n_enc = modes.shape[1]
    for n in range(n_enc):
        if(np.random.rand() > 0.1 + (n*0.9/(n_enc-1))):
            m = np.where(modes[:,n])[0][0]; res[m,n] = 0
            res[random.choice(np.setdiff1d(all_m,m)),n] = 1
    return res

### Construct inputs for all observation module
def obs_module_input(num_agents, n_neighbors, epi_len, n_enc, n_kpm, n_vis,
                     pos_in, gaze_in, phist_in, ghist_in, modeN_in, rot_pos, 
                     rel_gaze, modes_neigh, mode_noise, model_name):
    
    if(model_name in ['HistKPM','HistNoSelf']): n_add = n_kpm
    elif(model_name in ['AllHist','All-SocPool','All-NoSelfState','All-NoKPMGate']): n_add = n_vis
    else: n_add = n_enc
    n_end = max(n_enc,n_add)

    # construct neighbor list considered
    if(n_neighbors < num_agents): 
        neigh_list = np.arange(n_neighbors)
    else:
        neigh_list = np.concatenate([np.arange(num_agents-1),
            np.random.choice(num_agents-1, n_neighbors - num_agents+1)])

    # compute inputs for each obs module
    if(('All' in model_name) or ('Hist' in model_name)):
        for i,n in enumerate(neigh_list):
            pos_hists = np.array([rot_pos[n,:,t:t+n_add] 
                     for t in range(0, epi_len - n_end)])
            gaze_hists = np.array([rel_gaze[n,:,t:t+n_add] 
                     for t in range(0, epi_len - n_end)])
            phist_in[i].append(pos_hists.transpose(0,2,1))
            ghist_in[i].append(gaze_hists.transpose(0,2,1))
    if(('All-' in model_name) or ('Hist' in model_name) or (
        'SocPool' in model_name) or (model_name in ['NoSelfState','MAGDAM'])):
        for i,n in enumerate(neigh_list):
            pos_in[i].append( rot_pos[n, :, n_enc:].T )
            gaze_in[i].append( rel_gaze[n, :, n_enc:].T )
    for i,n in enumerate(neigh_list):
        mode_hists = np.array( [noisy(modes_neigh[n,:,t:(t + n_enc)], mode_noise)
                                for t in range(0, epi_len - n_end)] )
        mode_hists.shape = mode_hists.shape[:2] + (n_enc,)
        modeN_in[i].append( mode_hists.transpose(0,2,1) )



### Construct input for self-state encoder
def self_enc_input(epi_len, n_enc, n_end, modes_self, modeS_in, mode_noise):
    mode_hists = np.array( [noisy(modes_self[:,t:(t + n_enc)], mode_noise)
                            for t in range(0, epi_len - n_end)] )
    mode_hists.shape = mode_hists.shape[:2] + (n_enc,)
    modeS_in.append( mode_hists.transpose(0,2,1) )



### stacks and augments data from lists for network training
def stack_data(pos_in, gaze_in, phist_in, ghist_in,
               modeN_in, modeS_in, mode_out, augment = False):
    if(len(pos_in[0])):  pos_in  = [np.vstack(x) for x in pos_in]
    if(len(gaze_in[0])): gaze_in = [np.vstack(x) for x in gaze_in]
    if(len(phist_in[0])): phist_in = [np.vstack(x) for x in phist_in]
    if(len(ghist_in[0])): ghist_in = [np.vstack(x) for x in ghist_in]
    modeN_in = [np.vstack(x) for x in modeN_in]
    modeS_in = np.vstack(modeS_in)
    mode_out = np.vstack(mode_out)
    
    # data augmentation - same data reflected in y-axis added
    if(augment):
        if(len(pos_in[0])):  pos_in  = [np.vstack((x, x * [-1, 1])) for x in pos_in]
        if(len(gaze_in[0])): gaze_in = [np.vstack((x, x * [-1, 1])) for x in gaze_in]
        if(len(phist_in[0])): phist_in = [np.vstack((x, x * [-1, 1])) for x in phist_in]
        if(len(ghist_in[0])): ghist_in = [np.vstack((x, x * [-1, 1])) for x in ghist_in]
        modeN_in = [np.vstack((x, x)) for x in modeN_in]
        modeS_in = [np.vstack((modeS_in, modeS_in))]
        mode_out = [np.vstack((mode_out, mode_out))]
    else: modeS_in = [modeS_in]; mode_out = [mode_out]
    
    return pos_in, gaze_in, phist_in, ghist_in, modeN_in, modeS_in, mode_out



### Converts data to input for the MAGDAM network.
def data_to_input(path_pattern, agt_subset = 'pos_neigh', n_neighbors = 8, 
        n_enc = 15, n_kpm = 1, n_vis = 15, pos_noise = 0, gaze_noise = 0, 
        mode_noise = False, augment = False, model_name = 'MAGDAM'):

    if(n_enc == 1): mode_noise = False
    n_end = max(n_enc,n_kpm)

    # initialize empty lists for inputs and outputs to train network
    n_ind = range(n_neighbors)
    pos_in  = [[] for _ in n_ind]; gaze_in = [[] for _ in n_ind]
    phist_in = [[] for _ in n_ind]; ghist_in = [[] for _ in n_ind]
    modeN_in = [[] for _ in n_ind]; modeS_in = []
    mode_out = []

    # compute input and output data from all data files (simulated)
    data_files = glob(path_pattern)
    for file in data_files:

        group = cPickle.load(open(file))                                        # group data per file
        pos, gaze, modes = data_noise(group, pos_noise, gaze_noise)             # add noise
        num_agents = pos.shape[0]; epi_len = pos.shape[2]                       # number of agts; steps                  
        if(num_agents < 2): continue;
        # construct inputs and outputs for all agents
        for agt in np.arange(num_agents):
            # computes data for each agent
            rot_pos, rel_gaze, modes_neigh, modes_self = data_per_agt(
                                   pos, gaze, modes, agt, num_agents, 
                                   n_neighbors, agt_subset=agt_subset)

            # construct inputs for observation modules
            obs_module_input(num_agents, n_neighbors, epi_len, n_enc, n_kpm,
                             n_vis, pos_in, gaze_in, phist_in, ghist_in,
                             modeN_in, rot_pos, rel_gaze, modes_neigh, 
                             mode_noise, model_name)

            # construct input for self-state encoder
            self_enc_input(epi_len, n_enc, n_end, modes_self, 
                           modeS_in, mode_noise)

            
            # construct output (expert demonstration) for each agent
            mode_out.append( modes_self[:,n_enc:n_enc+epi_len-n_end].T )

    # Final stacked inputs and outputs for network
    return (stack_data(pos_in, gaze_in, phist_in, ghist_in, 
            modeN_in, modeS_in, mode_out, augment=augment))


## Gets indicator weights on neighbor data 
## for social pooling baseline
def getSocPoolWts(pos_in, d_socPool, gsize):
    rnge = np.arange(-d_socPool/2,d_socPool/2)
    mesh = np.stack(np.meshgrid(rnge,rnge)).reshape(2,d_socPool**2).T
    weights = [[] for _ in range(len(mesh))]
    for t in range(pos_in[0].shape[0]):
        neighs = np.stack([np.floor(p[t]/(gsize*d_socPool/2))
                           for p in pos_in])
        for sqr in range(len(mesh)):
            wts = np.zeros(len(neighs))
            wts[np.where((neighs==mesh[sqr]).all(axis=1))] = 1
            weights[sqr].append(wts)
    weights = [np.vstack(w) for w in weights]
    return weights


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"]="3"
    path_pattern = sys.argv[1]
    model_name = sys.argv[2]
    n_set = [2,4,8,12]; n_enc = 15; n_kpm = 1; n_vis = 15
    pos_noise = 1e0; gaze_noise = 1e-1; mode_noise = False
    epochs = np.int(sys.argv[3]); batch_size = 4096
    d_socPool = 4; gsize = 50
    n_states = 6;

    for i,n_neighbors in enumerate(n_set):
        if(len(sys.argv) > 4):
            model_noself = CommModels(model_name='NoSelfState', 
                                      n_neighbors=n_neighbors,
                                      n_enc=n_enc, n_states=n_states)
            model_noself.model.load_weights(sys.argv[4+i])
            obs_wts = model_noself.obs_mod.get_weights()
        else: obs_wts = []

        print('*** No. of neighbors: '+str(n_neighbors)+' ***')
        print('*** Starting preprocessing ***')

        (pos_in, gaze_in, phist_in, ghist_in, 
         modeN_in, modeS_in, mode_out) = data_to_input(path_pattern, 
                agt_subset = 'pos_neigh', n_neighbors = n_neighbors, 
                n_enc = n_enc, n_kpm = n_kpm, n_vis = n_vis, 
                pos_noise = pos_noise, gaze_noise = gaze_noise, 
                mode_noise = mode_noise, augment = True, 
                model_name = model_name)
        if('SocPool' in model_name): pool_wts = getSocPoolWts(pos_in, d_socPool, gsize)
        else: pool_wts = []
        
        print('*** Preprocessing done ***')

        model = CommModels(model_name=model_name, 
             # Change to required log directory
             logdir='logs/logs_hist'+str(n_enc)+'/logs_[0-4]/',
             logfile = 'result_'+model_name+str(n_neighbors)+'.csv', 
             check='model_'+model_name+str(n_neighbors), n_states=n_states, n_enc = n_enc, 
             n_kpm = n_kpm, n_vis = n_vis, n_neighbors=n_neighbors)
        model.fitModel(pos_in, gaze_in, phist_in, ghist_in, 
             modeN_in, modeS_in, mode_out, pool_wts=pool_wts, 
             epochs=epochs, batch_size=batch_size, obs_wts = obs_wts)
