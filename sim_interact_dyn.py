import cPickle
import numpy as np
import numpy.random as np_r
import numpy.linalg as LA
import matplotlib
#matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Arrow
from matplotlib.collections import PatchCollection as PC
import matplotlib.animation as animation
import scipy
import scipy.spatial.distance as dist

class GroupInteract():
    def __init__(self, start_pos=[], start_gaze=[], gID=[], mode_n = 7, cdist = 50,
                 dur_spk = 10, dur_res = 4, dur_dis = 5, dur_noise = 1e-1, 
                 p_mv = 0.1, p_act = 0.1, p_dis = 0.05, d_shift = 0.2):

        self.N = len(start_pos)                         # total no. of agents
        assert len(start_gaze) == self.N
        self.N_aug = self.N                             # augmented no. of agents 
                                                        # (grps of 1 become of grps of 2)
        self.start_pos = start_pos                      # start poses
        self.start_gaze = start_gaze                    # start gazes
        self.gID = gID                                  # group ID vector
        if(len(self.gID) != self.N):
            assert(len(self.gID) == 1)                  # if single group..
            self.gID = np.ones(self.N) * gID            # ..create vector 
        
        _, u = np.unique(self.gID, return_index=True)
        if(not self.N): self.groups = []  
        else: self.groups = self.gID[np.sort(u)]        # group indicators
        if(len(self.groups==0)): print('wtf')
        self.mode_n = mode_n                            # no. of modes

        self.cdist = cdist                              # threshold distance ..
                                                        # ..for moving agent
        self.dur_spk = dur_spk                          # speaker duration
        self.dur_res = dur_res                          # response duration
        self.dur_dis = dur_dis                          # distracted duration
        self.dur_noise = dur_noise                      # noise to add
        self.p_mv = p_mv                                # prob. of movement
        self.p_act = p_act                              # prob. of reset
        self.p_dis = p_dis                              # prob. of distraction
        self.d_shift = d_shift                          # degree of gaze shift

        self.rep_rad = 20
        self.m_step = 5

        self.init_memory()                              # initialize memory


    # Initialize memory
    def init_memory(self):

        # Augment groups of only one agent
        for grp in self.groups:
            i_grp = np.where(self.gID==grp)[0]
            if len(i_grp) == 1: 
                self.N_aug = self.N_aug + 1
                self.gID = np.append(self.gID, [self.gID[i_grp[0]]], axis=0)
                self.start_pos = np.append(self.start_pos, 
                    [self.start_pos[i_grp[0]] + 
                    np_r.rand(self.start_pos.shape[1])], axis=0)
                self.start_gaze = np.append(self.start_gaze, 
                    [self.start_gaze[i_grp[0]] + 
                    np_r.rand(self.start_gaze.shape[1])], axis=0)


        self.memory ={'mode':[[] for _ in range(self.N)], 
                       'pos':[[] for _ in range(self.N)],
                      'gaze':[[] for _ in range(self.N)],
                       'gID':[]}                               # memory of interactions
        self.now   = {'cent':np.zeros((len(self.groups),2)),
                      'pose':np.zeros((self.N_aug,2)),
                      'gaze':np.zeros((self.N_aug,2)),
                      'mode':np.ones(self.N_aug, dtype=int),
                       'spk':np.zeros(self.N_aug),
                       'now':np.zeros(self.N_aug),
                       'res':np.zeros(self.N_aug),
                       'dis':np.zeros(self.N_aug),
                      'look':np.zeros((self.N_aug,2)),
                        'to':np.zeros(self.N_aug, dtype=int),
                       'new':np.ones(self.N_aug)*np.Inf}

        self.now['pose'] = self.start_pos.copy()
        self.now['gaze'] = self.start_gaze.copy()
        self.now['look'] = self.start_pos + self.start_gaze

        for g,grp in enumerate(self.groups):
            i_grp = np.where(self.gID[:self.N]==grp)[0]
            n_grp = len(i_grp)

            if n_grp == 1: n_grp = n_grp+1
            mode = np.ones(n_grp)
            mode[np_r.choice(n_grp)] = 0
            
            for i,ind in enumerate(i_grp):
                self.memorize(self.start_pos[ind,:], 
                              self.start_gaze[ind,:],
                              int(mode[i]), ind)


            i_grp_aug = np.where(self.gID==grp)[0]
            self.now['cent'][g] = np.mean(self.start_pos[i_grp_aug], 
                            axis=0, dtype=np.float64)
            self.now['mode'][i_grp_aug] = mode
            self.now['spk'][i_grp_aug[np.where(mode==0)[0]]] = (
                                      self.d_rand(self.dur_spk))
            self.now['dis'][i_grp_aug[np.where(mode==1)[0]]] = (
                                      self.dur_dis)

        self.memory['gID'].append(self.gID[:self.N])


    # Store interactions
    def memorize(self, pos, gaze, mode, ind):
        mode_m = np.zeros(self.mode_n)
        mode_m[int(mode)] = 1
        self.memory['mode'][ind].append([mode_m])
        self.memory['pos'][ind].append([pos])
        self.memory['gaze'][ind].append([gaze])

    def d_rand(self, mean):
        return(np.max((np.ceil(np_r.normal(mean, scale=self.dur_noise)), 1)))


    # Run interaction rules for all groups for `epi_len' time
    def run_all(self, epi_len = 300):
    	gID_old = self.gID.copy()
        for stp in range(epi_len):
            gID_new = self.gID.copy()
            pdists = dist.squareform(dist.pdist(self.pos[:,:,-1]))
            for g,grp in enumerate(self.groups):
                i_grp = np.where(self.gID==grp)[0]
                n_grp = len(i_grp)

                s = i_grp[np.where([self.now['mode'][i_grp]==j for j in [0,3,4]])[1][0]]
                d = i_grp[np.where(self.now['mode'][i_grp]==2)[0]]
                r = i_grp[np.where(self.now['mode'][i_grp]==5)[0]]
                for i in i_grp:
                    m = i_grp[np.where(self.now['mode'][i_grp]==6)[0]]
                    movejoin = m[np.where(gID_old[m]!=self.gID[m])[0]]
                    #### Speaking mode
                    if self.now['mode'][i] == 0:

                        # If the next chosen speaker is responding,
                        # start listening
                        if len(r): 
                            self.now['mode'][i] = 1
                            self.now['spk'][i] = 0
                            self.now['res'][i] = 0
                            self.now['dis'][i_grp] = self.dur_dis

                        # If duration of speaking isn't done,
                        # keep speaking
                        elif self.now['spk'][i] > 0: 
                            self.now['new'][i_grp] = np.Inf
                            self.now['spk'][i] -= 1

                        # -- If a moving agent has joined the group,
                        #    assign him as next speaker 
                        #    (and relieve any other next assigned)
                        # -- Else, if a new speaker hasn't been assigned, 
                        #    assign one
                        # -- Switch to weakly or strongly addressing 
                        #    while waiting for response
                        else:
                            if(len(movejoin) and self.now['spk'][movejoin] > 0): 
                                if(self.now['new'][i] not in [movejoin]):
                                    if(self.now['new'][i] != np.Inf):
                                        self.now['spk'][self.now['new'][i]] = 0
                                        self.now['res'][self.now['new'][i]] = 0
                                    self.now['look'][i] = self.now['pose'][movejoin]
                                    self.now['new'][i_grp] = movejoin
                            elif self.now['new'][i] == np.Inf:
                                snew = np_r.choice(np.setdiff1d(i_grp,np.union1d([i],m)))
                                self.now['spk'][snew] = self.d_rand(self.dur_spk)
                                self.now['res'][snew] = self.d_rand(self.dur_res)
                                self.now['look'][i] = self.now['pose'][snew]
                                self.now['new'][i_grp] = snew
                            self.now['mode'][i] = 3 if(len(d)) else 4


                    #### Listening mode
                    elif self.now['mode'][i] == 1:

                        # If chosen as next speaker,
                        # respond after duration
                        if self.now['spk'][i] > 0:
                            self.now['look'][i] = self.now['pose'][s]
                            self.now['res'][i] -= 1
                            if self.now['res'][i] <= 0: self.now['mode'][i] = 5
                        
                        # Otherwise be distracted after some time 
                        # with a certain probability
                        else:
                            self.now['dis'][i] -= 1
                            if self.now['dis'][i] <= 0 and np_r.rand() < self.p_dis:
                                self.now['mode'][i] = 2

                    
                    #### Distracted mode
                    elif self.now['mode'][i] == 2:

                        # If chosen as next speaker,
                        # respond after duration
                        if self.now['spk'][i] > 0:
                            self.now['look'][i] = self.now['pose'][s]
                            self.now['res'][i] -= 1
                            if self.now['res'][i] <= 0: self.now['mode'][i] = 5

                        # Otherwise start moving
                        # with a certain probability
                        else:
                            if(i < self.N and len(m) == 0 and n_grp != 2 and len(self.groups) > 1 and np_r.rand() < self.p_mv):
                                self.now['mode'][i] = 6
                                self.now['to'][i] = np_r.choice(np.setdiff1d(
                                                  range(len(self.groups)),g))
                                self.now['look'][i] = self.now['cent'][self.now['to'][i]]


                    #### Strongly or Weakly Addressing
                    elif self.now['mode'][i] in [3,4]: 
                        if len(r): 
                            self.now['mode'][i] = 1
                            self.now['spk'][i] = 0
                            self.now['res'][i] = 0
                            self.now['dis'][i_grp] = self.dur_dis
                        else:
                            self.now['spk'][i] = self.d_rand(self.dur_spk)
                            self.now['mode'][i] = 0

                    #### Responding mode
                    elif self.now['mode'][i] == 5: 
                        self.now['mode'][i] = 0
                        if(np_r.rand() > self.p_act):
                            self.now['look'][i_grp] += (np_r.rand(2) - .5) * 100

                    #### Moving mode
                    elif self.now['mode'][i] == 6:
                        if self.now['spk'][i] > 0: self.now['mode'][i] = 1
                        else:
                            cdir = self.now['cent'][self.now['to'][i]] - self.now['pose'][i]
                            if(LA.norm(cdir) <= self.cdist):
                                gID_new[i] = self.groups[self.now['to'][i]]
                                self.now['spk'][i] = self.d_rand(self.dur_spk)
                                self.now['res'][i] = self.d_rand(self.dur_res)
                            else:
                                if i < self.N:
                                    near = self.now['pose'][i]- self.now['pose'][np.setdiff1d(np.where(pdists[i]<self.rep_rad)[0],i)]
                                    if(near.shape[0]):
                                        fin = np.sum(near/(LA.norm(near,axis=1).reshape(near.shape[0],1)),axis=0) + cdir/LA.norm(cdir)
                                    else:
                                        fin = cdir/LA.norm(cdir)
                                else:
                                    fin = cdir/LA.norm(cdir)
                                self.now['pose'][i] = self.now['pose'][i] + self.m_step*fin/LA.norm(fin)
                                #raw_input("Press enter")

                    dg = self.now['look'][i] - self.now['pose'][i]
                    ng = self.d_shift*dg/LA.norm(dg) + (
                        (1-self.d_shift)*self.now['gaze'][i])
                    self.now['gaze'][i] = ng/LA.norm(ng)
                    if i < self.N:
                        self.memorize(self.now['pose'][i].copy(), self.now['gaze'][i].copy(), 
                                      self.now['mode'][i].copy(), i)
                self.now['cent'][g] = np.mean(self.now['pose'][np.setdiff1d(i_grp,m)], 
                            axis=0, dtype=np.float64)

            gID_old = self.gID.copy()
            self.gID = gID_new.copy()
            self.memory['gID'].append(self.gID[:self.N])




    # Visualize interactions:
    #    either from loaded simulation file or self 
    #    for any subset of groups (vis_grps)
    #    at steps in episodes at intervals (stp_skip)
    def visualize(self, sim_file = '', vis_grps = 'all', stp_skip = 1):

        vis_obj = cPickle.load(open(sim_file)) if(sim_file != '') else self
        vis_obj.gID = vis_obj.gID[:vis_obj.N]
        # Select subset of self properties acc to vis_grps
        if(vis_grps == 'all'): vis_grps = vis_obj.groups
        vis_sub = np.array([vis_obj.gID == i 
            for i in list(np.unique(vis_grps))]).any(axis=0)
        pos  =  vis_obj.pos[vis_sub]
        gaze = vis_obj.gaze[vis_sub]
        mode = vis_obj.mode[vis_sub]
        gID = vis_obj.group_id[vis_sub]
        _, u = np.unique(gID[:,0], return_index=True)
        groups = list(gID[np.sort(u),0])
        n_agts = pos.shape[0]


        # Figure creation
        fig, ax = plt.subplots(figsize=(10,10))
        ax.set_xlim((np.min(pos) - 50., np.max(pos) + 50.))
        ax.set_ylim((np.min(pos) - 50., np.max(pos) + 50.))
        ax.set_axis_bgcolor((0.9,0.9,0.9))
        # Patch objects to represent poses, gazes, modes
        pos_patch  = [[] for _ in range(n_agts)] 
        mode_patch = [[] for _ in range(n_agts)] 
        gaze_patch = [[] for _ in range(n_agts)]

        #Writer = animation.writers['ffmpeg']
        #writer = Writer(fps=15, metadata=dict(artist='NavyataS'), bitrate=1800)

        # Patch object properties (radii and colors)
        rad_p = 15; rad_m = 8; col_grp = []
        col_mode = ['r','g','y','r','r','r', 'b']
        for i in range(len(groups)):
            col = (1,1,1)
            if(sum(gID[:,0]==groups[i]) != 1):
                col = np_r.choice(2,3)
                col[np_r.choice(3)] = np_r.rand()
            col_grp.append(tuple(col))

        # Update figure across time
        def update_plot(t):
            for i in range(n_agts):
                pos_patch[i]  = Circle(pos[i,:,t], radius=rad_p, 
                    fc=col_grp[groups.index(gID[i,t])], alpha = 0.4)
                mode_patch[i] = Circle(pos[i,:,t], radius=rad_m, 
                    fc=col_mode[list(mode[i,:,t]).index(1)])
                gaze_patch[i] = Arrow(pos[i,0,t], pos[i,1,t], 
                    25*gaze[i,0,t], 25*gaze[i,1,t], width=30, fc = 'k', ec='k')
            print(t)
            ax.clear()
            p_coll = PC(pos_patch+mode_patch+gaze_patch, match_original=True)
            #p_coll = PC(pos_patch+gaze_patch, match_original=True)
            ax.add_collection(p_coll)

        anim = animation.FuncAnimation(fig, update_plot, 
                        frames=range(0,pos.shape[2],stp_skip),interval=1)
        #anim.save('simulation.mp4',writer=writer)
        plt.show()

    @property
    def mode(self):
        # return np.hstack(self.memory['mode']).transpose(
        #     1,2,0)[self.orig_agts]
        return np.hstack(self.memory['mode']).transpose(
            1,2,0)
    
    @property
    def pos(self):
        # return np.hstack(self.memory['pos']).transpose(
        #     1,2,0)[self.orig_agts]
        return np.hstack(self.memory['pos']).transpose(
            1,2,0)
    
    @property
    def gaze(self):
        # return np.hstack(self.memory['gaze']).transpose(
        #     1,2,0)[self.orig_agts]
        return np.hstack(self.memory['gaze']).transpose(
            1,2,0)

    @property
    def group_id(self):
        # return np.hstack(self.memory['gaze']).transpose(
        #     1,2,0)[self.orig_agts]
        return np.array(self.memory['gID']).T