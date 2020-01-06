from sim_interact import GroupInteract
from train_model import CommModels
from model_gru import GPOM
from sklearn.cluster import DBSCAN
from scipy.spatial import distance_matrix
import numpy as np
import time

n_enc = 15; n_kpm = 1; n_vis = 15
model_name = 'AllHist'
n_neighbors = 12
epoch = '020'
modes = 7   # 6 for static case, 7 for dynamic case

# Change to required log directory
logdir = 'logs_dyn_mix/logs_hist'+str(n_enc)+'/logs_10/'
weight_file = logdir+'model_'+model_name+str(n_neighbors)+'.'+epoch+'.hdf5'
model = CommModels(model_name=model_name, n_neighbors=n_neighbors, n_states=modes, n_enc = n_enc, n_kpm = n_kpm, n_vis = n_vis)
model.model.load_weights(weight_file)
gate = model.kpm_gate

if(model_name == 'HistKPM'): visCue = True; n_tile = n_kpm
elif(model_name == 'AllEnc'): visCue = True; n_tile = n_enc
else: visCue = False; n_tile = n_enc

data_dir = 'interaction_data/interaction_data/'
d_names = ['synth_data.npy', 'coffee_data.npy', 'cocktailparty.npy']
c_names = ['', 'calib_coffee.npy', 'calib_cocktail.npy']
db_eps = []
precs = []; recs = []; fmeases =[]; times=[];

for d_name,c_name in zip(d_names,c_names):
    data = np.load(data_dir+d_name)
    if(c_name == ''): cal = 1
    else: cal = np.load(data_dir+c_name)
    pose = [d[:,:2]*cal for d in data if(len(d)>1)]
    gaze = [d[:,2:4] for d in data if (len(d)>1)]
    gid = [d[:,-1] for d in data if (len(d)>1)]

    b_prec = 0; b_rec = 0; b_fmeas = 0
    c_prec = 0; c_rec = 0; c_fmeas = 0
    b_eps = 0.01; b_time = 0.0

    for eps in np.arange(0.01, 0.8, 0.01):
        prec = []; rec = []; g_times = [];
        for ps, gs, ids in zip(pose, gaze, gid):
            start = time.time()
            if(not np.any(ids > 0)): continue
            n = len(ps); dist = np.zeros((n,n))
            for agt in range(n):
                th = np.arctan2(gs[agt,0], gs[agt,1])
                rot = np.array([[np.cos(th), -np.sin(th)], 
                                [np.sin(th), np.cos(th)]])
                rot_pos = np.einsum('ij,kj->ki', rot, ps - ps[agt])
                rel_gaze = np.einsum('ij,kj->ki', rot, gs)
                if(visCue):
                    rot_pos.shape = rot_pos.shape + (1,)
                    rot_pos = np.tile(rot_pos,n_tile).transpose(0,2,1)
                    rel_gaze.shape = rel_gaze.shape + (1,)
                    rel_gaze = np.tile(rel_gaze,n_tile).transpose(0,2,1)
                gate_pred = gate.predict([rot_pos, rel_gaze])
                dist[agt] = gate_pred.reshape((gate_pred.shape[0],))
                dist[agt] /= dist[agt,agt]
            #dist = np.around(1 - ((dist+dist.T)/2),decimals=3)
            dist = distance_matrix(ps,ps); dist /= np.max(dist);
            p_ids = DBSCAN(eps=eps, min_samples=2, metric='precomputed').fit_predict(dist)
            p_ids[p_ids < 0] = np.arange(1,np.sum(p_ids < 0)+1)*-1
            true_g = [(ids == i)*1. for i in np.unique(ids) if np.sum(ids == i) > 1]
            pred_g = [(p_ids == i)*1. for i in np.unique(p_ids) if i >= 0]
            if(len(pred_g) == 0): 
                prec.append(0.0); rec.append(0.0)
                continue
            conf_g = [[np.abs(true-pred).sum() == 0 for true in true_g] for pred in pred_g]
            #conf_g = [[np.abs(true-pred).sum() <= np.floor(0.34*np.sum(true)) for true in true_g] for pred in pred_g]
            TP = np.any(conf_g, axis=1).sum()
            p = (TP/float(len(pred_g))); r = (TP/float(len(true_g)))

            end = time.time()
            
            prec.append(p); rec.append(r); g_times.append(end-start);
        c_prec = np.sum(prec)/float(len(prec))
        c_rec = np.sum(rec)/float(len(rec))
        c_fmeas = 2*c_prec*c_rec/(c_prec+c_rec)
        if(c_fmeas > b_fmeas):
            b_prec = c_prec
            b_rec = c_rec
            b_fmeas = c_fmeas
            b_eps = eps
            b_time = np.mean(g_times)
    precs.append(b_prec)
    recs.append(b_rec)
    fmeases.append(b_fmeas)
    db_eps.append(b_eps)
    times.append(b_time)

print('Precisions: ', precs)
print('Recalls: ', recs)
print('F-measures: ', fmeases)
print('Times: ', times)
print('model, n_enc, n_kpm: ', model_name, n_enc, n_kpm)
print('neighbors, epoch: ', n_neighbors, epoch)
print('DB eps: ', db_eps)

