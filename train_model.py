import numpy as np
from distutils import dir_util as DU
import keras
from keras.models import Model, Input
from keras.layers import Dense, RepeatVector, Flatten, GRU, Reshape
from keras.layers.merge import Multiply, Average, Dot
from keras.layers.merge import Concatenate as Cat
from keras.engine.topology import Container
from keras.callbacks import ModelCheckpoint, CSVLogger
from keras.backend import int_shape
from sklearn.utils import compute_class_weight
from scipy.spatial.distance import cdist


class CommModels:

    def __init__(self, model_name='MAGDAM', n_neighbors=8, n_enc=15, 
                 n_kpm = 1, n_vis = 15, n_states=6, d_socPool=4, metrics=[], 
                 logdir='logs/', logfile = 'result.csv', check='model', 
                 summarize = False):
        self.n_states = n_states;             self.n_neighbors = n_neighbors
        self.n_enc = n_enc;                   self.n_kpm = n_kpm
        self.n_vis = n_vis;                   self.d_socPool = d_socPool
        self.model_name = model_name;         self.metrics = metrics
        self.makeModel()      
        if(summarize): self.model.summary()
        
        DU.mkpath(logdir);                    self.callbacks = []
        self.callbacks.append(ModelCheckpoint(filepath=(logdir+check+
            '.{epoch:03d}.hdf5'), verbose=1, period=10))
        self.callbacks.append(CSVLogger(logdir+logfile))

    def kpmGate(self):
        if(self.n_kpm > 1):
            self.pose = Input(shape=(self.n_kpm, 2), name='pose')
            self.gaze = Input(shape=(self.n_kpm, 2), name='gaze')
            vis_out = GRU(64, name='KPM-gru')(Cat()([self.pose, self.gaze]))
        else:
            self.pose = Input(shape=(2,), name='pose')
            self.gaze = Input(shape=(2,), name='gaze')
            vis_out = Dense(64, activation='elu', name='KPM-elu')(
                      Cat()([self.pose, self.gaze]))
        sig_out = Dense(1, activation='hard_sigmoid', name='KPM-sig', 
            kernel_initializer='zero')(vis_out)
        rep_out = Flatten()(RepeatVector(64)(sig_out))
        self.kpm_gate = Model([self.pose, self.gaze], sig_out) 
        self.kpm_mod = Container([self.pose, self.gaze], rep_out, 
                       name='KPM-Gate')
    
    def neighVisEncoder(self):
        self.visP = Input(shape=(self.n_vis, 2), name='visP')
        self.visG = Input(shape=(self.n_vis, 2), name='visG')
        gru_out = GRU(64, name='VisEnc-gru')(Cat()([self.visP, self.visG]))
        elu_out = Dense(64, activation='elu', name='VisEnc-elu')(gru_out)
        self.nvis_mod = Container([self.visP, self.visG], elu_out, name='VisEnc')

    def neighStateEncoder(self, vStt = False):
        self.modeN = Input(shape=(self.n_enc, self.n_states), name='mode')
        if(vStt):
            self.pose = Input(shape=(self.n_enc, 2), name='pose')
            self.gaze = Input(shape=(self.n_enc, 2), name='gaze')
            gru_out = GRU(64, name='StateEnc-gru')(Cat()([self.pose, 
                                               self.gaze, self.modeN]))
            elu_out = Dense(64, activation='elu', name='StateEnc-elu')(gru_out)
            self.nstate_mod = Container([self.pose, self.gaze, self.modeN], 
                                         elu_out, name='StateEnc')
        else:
            gru_out = GRU(64, name='StateEnc-gru')(self.modeN)
            elu_out = Dense(64, activation='elu', name='StateEnc-elu')(gru_out)
            self.nstate_mod = Container([self.modeN], elu_out, name='StateEnc')

    def selfStateEncoder(self):
        self.modeS = Input(shape=(self.n_enc, self.n_states), name='self_mode')
        gru_out = GRU(64, name='SelfStateEnc-gru')(self.modeS)
        self.sstate_mod = Container([self.modeS], gru_out, name='SelfStateEnc')

    def obsModule(self,kpm = True,neighEnc = True,visEnc = False,vStt = False):
        if(neighEnc): self.neighStateEncoder(vStt=vStt)
        if(visEnc): self.neighVisEncoder()
        if(kpm):
            self.kpmGate(); mod_inp = []; mod_out = []
            if(neighEnc):
                mod_inp += [self.pose, self.gaze, self.modeN]
                mod_out = Multiply()([self.kpm_mod([self.pose, self.gaze]),
                                      self.nstate_mod([self.modeN])])
            if(visEnc):
                mult_out = Multiply()([self.kpm_mod([self.pose, self.gaze]),
                                       self.nvis_mod([self.visP, self.visG])])
                if(mod_out != []): mod_out = Cat()([mod_out, mult_out])
                else: mod_out = mult_out
                mod_inp += [self.visP, self.visG]
            self.obs_mod = Container(mod_inp, mod_out)
        else:
            if(neighEnc and not visEnc): self.obs_mod = self.nstate_mod
            elif(not neighEnc and visEnc): self.obs_mod = self.nvis_mod
            elif(neighEnc and visEnc):
                mod_out      = Cat()([self.nstate_mod([self.modeN]),
                               self.nvis_mod([self.visP, self.visG])])
                self.obs_mod = Container([self.modeN, self.visP, self.visG],mod_out)        
        self.obs_mod.name = 'ObsMod'

    def neighModule(self, kpm = True, neighEnc = True, visEnc = False,
                    socPool = False, vStt = False):
        self.obsModule(kpm=kpm, neighEnc=neighEnc, visEnc=visEnc, vStt=vStt)
        self.weight = [];   self.modeN_all = []
        self.pose_all = []; self.gaze_all = []
        self.pose_stt = []; self.gaze_stt = []
        self.visP_all = []; self.visG_all = []
        
        observations = []
        for n in range(self.n_neighbors):
            obs_input = []
            if(kpm):
                shape = (self.n_kpm, 2) if(self.n_kpm > 1) else (2,)
                self.pose_all.append(Input(shape=shape, name='pose_%d' % n))
                self.gaze_all.append(Input(shape=shape, name='gaze_%d' % n))
                obs_input.append(self.pose_all[n]); obs_input.append(self.gaze_all[n])
            if(vStt):
                self.pose_stt.append(Input(shape=(self.n_enc, 2), name='pStt_%d' % n))
                self.gaze_stt.append(Input(shape=(self.n_enc, 2), name='gStt_%d' % n))
                obs_input.append(self.pose_stt[n]); obs_input.append(self.gaze_stt[n])
            if(neighEnc):
                self.modeN_all.append(Input(shape=(self.n_enc, self.n_states), 
                                                            name='mode_%d' % n))
                obs_input.append(self.modeN_all[n])
            if(visEnc):
                self.visP_all.append(Input(shape=(self.n_vis, 2), name='visP_%d' % n))
                self.visG_all.append(Input(shape=(self.n_vis, 2), name='visG_%d' % n))
                obs_input.append(self.visP_all[n]); obs_input.append(self.visG_all[n])
            observations.append(self.obs_mod(obs_input))
        
        if(socPool):
            observations = Cat(axis=2)([Reshape((int_shape(observations[i])[1],1))(
                observations[i]) for i in range(len(observations))])
            self.weight = [Input(shape=(self.n_neighbors,)) 
                          for _ in range(self.d_socPool**2)]
            neigh_avg = Cat()([Flatten()(Dot(2)([observations,
                        Reshape((1,self.n_neighbors))(self.weight[i])]))
                        for i in range(self.d_socPool**2)])
        else: neigh_avg = Average()(observations)

        self.nmod_input = (self.weight+self.pose_stt+self.gaze_stt+
                                       self.pose_all+self.gaze_all+
                                       self.visP_all+self.visG_all+
                                       self.modeN_all)
        self.neigh_mod = Container(self.nmod_input, neigh_avg)

    def policyModule(self, kpm = True, neighEnc = True, visEnc = False, 
                     selfEnc = True, socPool = False, vStt = False):
        if(selfEnc):
            self.selfStateEncoder()
            elu_in = self.sstate_mod([self.modeS])
            self.pmod_input = [self.modeS]
        if(kpm or neighEnc or visEnc):
            self.neighModule(kpm=kpm, neighEnc=neighEnc, visEnc=visEnc, 
                             socPool=socPool, vStt=vStt)
            elu_in = self.neigh_mod(self.nmod_input)
            self.pmod_input = self.nmod_input
            if(selfEnc):
                elu_in = Cat()([self.neigh_mod(self.nmod_input), 
                                self.sstate_mod([self.modeS])])
                self.pmod_input = self.nmod_input + [self.modeS]
   
        elu_out = Dense(64, activation='elu', name='PolMod-elu')(elu_in)
        smax_out = Dense(self.n_states, activation='softmax', 
                   name='PolMod-smax')(elu_out)
        self.pmod_out = smax_out

    def makeModel(self):
        print('*** Compiling model ***')
        if(self.model_name in ['MAGDAM']):
            self.policyModule(kpm=1, neighEnc=1, selfEnc=1)
        elif(self.model_name in ['HistKPM']):
            self.policyModule(kpm=1, neighEnc=1, selfEnc=1, vStt=1)
        elif(self.model_name in ['NoSelfState']):
            self.policyModule(kpm=1, neighEnc=1, selfEnc=0)
        elif(self.model_name in ['HistNoSelf']):
            self.policyModule(kpm=1, neighEnc=1, selfEnc=0, vStt=1)
        elif(self.model_name in ['NoNeighMod']):
            self.policyModule(kpm=0, neighEnc=0, selfEnc=1)
        elif(self.model_name in ['NoKPMGate']):
            self.policyModule(kpm=0, neighEnc=1, selfEnc=1)
        elif(self.model_name in ['SocPool']):
            self.policyModule(kpm=0, neighEnc=1, selfEnc=1, socPool=1)
        elif(self.model_name in ['AllEnc']):
            self.policyModule(kpm=0, neighEnc=1, selfEnc=1, vStt=1)
        elif(self.model_name in ['AllHist']):
            self.policyModule(kpm=1, neighEnc=1, visEnc=1, selfEnc=1)
        elif(self.model_name in ['All-NoSelfState']):
            self.policyModule(kpm=1, neighEnc=1, visEnc=1, selfEnc=0)
        elif(self.model_name in ['All-NoKPMGate']):
            self.policyModule(kpm=0, neighEnc=1, visEnc=1, selfEnc=1)
        elif(self.model_name in ['All-SocPool']):
            self.policyModule(kpm=0, neighEnc=1, selfEnc=1, visEnc=1, socPool=1)

        self.model = Model(self.pmod_input, outputs=[self.pmod_out])
        self.model.compile(optimizer='adam', metrics = self.metrics,
            loss=['categorical_crossentropy'])
        print('*** Model compiled **')

    def fitModel(self, pos_in, gaze_in, phist_in, ghist_in, modeN_in, modeS_in, 
                 mode_out, pool_wts=[], epochs=50, batch_size=64, obs_wts=[]):

        print('*** Training model ***')
        if(obs_wts != []): self.obs_mod.set_weights(obs_wts)
        max_mode = np.argmax(mode_out[0], axis=1)
        classes = np.unique(max_mode)
        cw = compute_class_weight('balanced', classes, max_mode)
        class_weight = {i:w for i, w in enumerate(cw)}
        if(self.model_name in ['MAGDAM']): 
            mod_in = pos_in+gaze_in+modeN_in+modeS_in
        elif(self.model_name in ['NoSelfState']): 
            mod_in = pos_in+gaze_in+modeN_in
        elif(self.model_name in ['NoNeighMod']): 
            mod_in = modeS_in
        elif(self.model_name in ['NoKPMGate']): 
            mod_in = modeN_in+modeS_in
        elif(self.model_name in ['SocPool']): 
            mod_in = pool_wts+modeN_in+modeS_in
        elif(self.model_name in ['AllEnc','HistKPM']):
            mod_in = phist_in+ghist_in+modeN_in+modeS_in
        elif(self.model_name in ['HistNoSelf']): 
            mod_in = phist_in+ghist_in+modeN_in
        elif(self.model_name in ['AllHist']):
            mod_in = pos_in+gaze_in+phist_in+ghist_in+modeN_in+modeS_in
        elif(self.model_name in ['All-NoSelfState']):
            mod_in = pos_in+gaze_in+phist_in+ghist_in+modeN_in
        elif(self.model_name in ['All-NoKPMGate']):
            mod_in = phist_in+ghist_in+modeN_in+modeS_in
        elif(self.model_name in ['All-SocPool']):
            mod_in = pool_wts+phist_in+ghist_in+modeN_in+modeS_in

        self.model.fit(mod_in, mode_out, epochs=epochs, class_weight=class_weight, 
            batch_size=batch_size, verbose=1, callbacks=self.callbacks)
