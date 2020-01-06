import numpy as np
import tensorflow as tf
from train_model import CommModels
import train
import cPickle

d_socPool = 4; gsize = 50
epoch = '030'
n_enc = 15; n_kpm = 1; n_vis = 15
modes = 6
summarize = False
mode_noise = False
path_pattern = "simulations/05_09/*"
#logdir = 'logs_dyn/logs_hist'+str(n_enc)+'/logs_[5-9]/'
logdir = 'logs1/logs_hist'+str(n_enc)+'/logs_[0-4]/'
model_names = ['NoNeighMod'];
#model_names = ['All-NoSelfState', 'NoNeighMod', 'All-NoKPMGate', 'All-SocPool', 'AllHist'];

for model_name in model_names:
    
    save_path = logdir+'test/'+model_name+'_test.p'
    losses = []; accs = []; mAPs = []; confs = [];
    if(model_name in ['NoNeighMod']): n_set = [2]
    else: n_set = [2,4,8,12]

    for n_neighbors in n_set:
        weight_file = logdir+'model_'+model_name+str(n_neighbors)+'.'+epoch+'.hdf5'
        model = CommModels(model_name=model_name, metrics = ['categorical_accuracy'],
             logdir=logdir,logfile = 'result_'+model_name+str(n_neighbors)+'.csv', 
             check='model_'+model_name+str(n_neighbors), n_states = modes, n_enc = n_enc, 
             n_kpm = n_kpm, n_vis = n_vis, n_neighbors=n_neighbors,
             summarize = summarize)
        model.model.load_weights(weight_file)

        print('Starting preprocessing')
        (pos_in, gaze_in, phist_in, ghist_in, 
         modeN_in, modeS_in, mode_out) = train.data_to_input(path_pattern, 
                agt_subset = 'pos_neigh', n_neighbors = n_neighbors, 
                n_enc = n_enc, n_kpm = n_kpm, n_vis = n_vis, 
                mode_noise=mode_noise, augment=True, model_name=model_name)
        if('SocPool' in model_name): pool_wts = train.getSocPoolWts(pos_in, d_socPool, gsize)
        else: pool_wts = []

        if(model_name in ['MAGDAM']): 
            mod_in = pos_in+gaze_in+modeN_in+modeS_in
        elif(model_name in ['NoSelfState']): 
            mod_in = pos_in+gaze_in+modeN_in
        elif(model_name in ['NoNeighMod']): 
            mod_in = modeS_in
        elif(model_name in ['NoKPMGate']): 
            mod_in = modeN_in+modeS_in
        elif(model_name in ['SocPool']): 
            mod_in = pool_wts+modeN_in+modeS_in
        elif(model_name in ['AllEnc','HistKPM']):
            mod_in = phist_in+ghist_in+modeN_in+modeS_in
        elif(model_name in ['HistNoSelf']): 
            mod_in = phist_in+ghist_in+modeN_in
        elif(model_name in ['AllHist']):
            mod_in = pos_in+gaze_in+phist_in+ghist_in+modeN_in+modeS_in
        elif(model_name in ['All-NoSelfState']):
            mod_in = pos_in+gaze_in+phist_in+ghist_in+modeN_in
        elif(model_name in ['All-NoKPMGate']):
            mod_in = phist_in+ghist_in+modeN_in+modeS_in
        elif(model_name in ['All-SocPool']):
            mod_in = pool_wts+phist_in+ghist_in+modeN_in+modeS_in

        print('Starting evaluation')
        scores = model.model.evaluate(mod_in, mode_out, verbose=0)
        print('Test Loss: ', scores[0]); print('Categorical accuracy: ', scores[1])

        predictions = model.model.predict(mod_in)
        tf_cm = tf.to_float(tf.confusion_matrix(tf.argmax(mode_out[0],-1), 
        	                                    tf.argmax(predictions,-1)))
        init_op = tf.global_variables_initializer()
        with tf.Session() as sess:
        	sess.run(init_op)
        	conf_matrix = tf_cm.eval()
        	mAP = (conf_matrix.diagonal()/conf_matrix.sum(-1)).mean()
        	print('Confusion Matrix: ')
        	print(conf_matrix)
        	print('Mean Average Precision: ', mAP)

        losses.append(scores[0]); accs.append(scores[1])
        mAPs.append(mAP); confs.append(conf_matrix)
    
    if(model_name in ['NoNeighMod']):
        n_set = [2,4,8,12];
        for _ in range(len(n_set)-1):
            losses.append(scores[0]); accs.append(scores[1])
            mAPs.append(mAP); confs.append(conf_matrix)

    cPickle.dump([n_set, losses, accs, mAPs, confs], open(save_path,'wb'))
