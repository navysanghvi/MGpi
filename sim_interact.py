import cPickle
import numpy as np
import numpy.random as np_r
import matplotlib
#matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Arrow
from matplotlib.collections import PatchCollection as PC
import matplotlib.animation as animation

class GroupInteract():
    def __init__(self, start_pos=[], start_gaze=[], gID=[], mode_n = 6, 
                 mode_spk=[0,3,4,5], dur_spk = 10, dur_res = 4, dur_dis = 5, 
                 dur_noise = 1e-1, p_act = 0.5, p_dis = 0.05, d_shift = 0.5):

        self.N = len(start_pos)                         # total no. of agents
        assert len(start_gaze) == self.N
        self.start_pos = start_pos                      # start poses
        self.start_gaze = start_gaze                    # start gazes
        self.gID = gID                                  # group ID vector
        if(len(self.gID) != self.N):
            assert(len(self.gID) == 1)                  # if single group..
            self.gID = np.ones(self.N) * gID            # ..create vector 
        
        _, u = np.unique(self.gID, return_index=True)
        if(not self.N): self.groups = []  
        else: self.groups = self.gID[np.sort(u)]        # group indicators

        self.mode_n = mode_n                            # no. of modes
        self.mode_spk = mode_spk                        # talking modes

        self.init_memory()                              # initialize memory

        self.dur_spk = dur_spk                          # speaker duration
        self.dur_res = dur_res                          # response duration
        self.dur_dis = dur_dis                          # distracted duration
        self.dur_noise = dur_noise                      # noise to add
        self.p_act = p_act                              #
        self.p_dis = p_dis                              # prob. of distraction
        self.d_shift = d_shift                          # degree of gaze shift


    # Initialize memory
    def init_memory(self):
        self.orig_agts = []
        self.memory={'mode':[[] for _ in self.groups], 
                      'pos':[[] for _ in self.groups],
                     'gaze':[[] for _ in self.groups]}  # memory of interactions
        
        for i,g in enumerate(self.groups):
            i_grp=self.gID==g; n_grp=sum(i_grp)
            pos  =  self.start_pos[i_grp,:]             # group start poses
            gaze = self.start_gaze[i_grp,:]             # group start gazes
            self.orig_agts.append([True]*n_grp)         # non-expanded agents
            if(n_grp == 1):                             # expand group to at least two
                self.orig_agts.append(False)
                dp =  np_r.rand(pos.shape[1])
                dg = np_r.rand(gaze.shape[1])
                pos  =   np.concatenate([pos, pos+dp])
                gaze = np.concatenate([gaze, gaze+dg])
                n_grp += 1
            mode = np.ones(n_grp)
            mode[np_r.choice(n_grp)] = 0                # one speaker, rest listeners
            self.memorize(pos, gaze, mode, i)           # memorize group starts

        if(self.orig_agts != []):
            self.orig_agts = np.hstack(self.orig_agts)


    # Gaze shift rules
    def shift_gaze(self, curr_pos, curr_gaze, look_at):
        dg = look_at - curr_pos
        dir_gaze = dg/np.linalg.norm(dg,axis=1)[:,np.newaxis]
        ng = self.d_shift*dir_gaze + (1-self.d_shift)*curr_gaze
        next_gaze = ng/np.linalg.norm(ng,axis=1)[:,np.newaxis]
        return next_gaze
    

    # Store interactions
    def memorize(self, pos, gaze, mode, ind):
        mode_m = np.zeros((len(mode), self.mode_n))
        for i, m in enumerate(mode):
            mode_m[i, np.int(m)] = 1
        self.memory['mode'][ind].append(mode_m)
        self.memory['pos'][ind].append(pos)
        self.memory['gaze'][ind].append(gaze)


    # Run interaction rules for a particular group
    def run_grp(self, grp_pos, grp_gaze, grp_mode, 
                dynamic, grp_ind, epi_len):

        n_grp = grp_pos.shape[0]                                        # no. of group agents
        agt_spk = np.where([grp_mode==g for g in self.mode_spk])        # talking agent (expect one)

        # Initialize
        curr_pos  =  grp_pos.copy();   next_pos  =  curr_pos.copy()
        curr_gaze = grp_gaze.copy();   next_gaze = curr_gaze.copy()
        curr_mode = grp_mode.copy();   next_mode = curr_mode.copy()
        curr_spk = agt_spk[1][0];      next_spk  = curr_spk
        look_at = curr_pos+curr_gaze;  dur_noise = self.dur_noise
        n_res = 0; n_spk = np_r.randint(0, self.dur_spk/2)
        dur_spk = np.max((np_r.normal(self.dur_spk, scale=dur_noise), 1))
        dur_res = np.max((np_r.normal(self.dur_res, scale=dur_noise), 1))

        # Run interaction for length of episode specified
        for stp in range(epi_len):
            for agt in range(n_grp):

                # Speaking mode
                if(curr_mode[agt] == 0):
                    if(agt != curr_spk):                                # not the assigned speaker
                        next_mode[agt] = 1; continue                    # start listening
                    n_spk += 1
                    if(n_spk < dur_spk):                                # within speaking duration
                        next_mode[agt] = 0; continue                    # keep speaking
                    while(next_spk == curr_spk): 
                        next_spk = np_r.choice(n_grp)                   # choose next speaker
                    next_mode[agt] = 3 if(np.any(curr_mode==2)) else 4  # strongly/weakly address                    
                    look_at[agt] = curr_pos[next_spk]                   # look at new speaker
                    
                    dur_res = np.max((np_r.normal(self.dur_res, 
                                      scale=dur_noise), 1))
                    n_spk, n_res = 0, 0
                
                # Listening mode
                elif(curr_mode[agt] == 1):
                    next_mode[agt] = 2 if(n_spk > self.dur_dis and 
                     np_r.rand() < self.p_dis) else 1                   # distracted with some probab
                    if(agt == next_spk):                                # if chosen next speaker..
                        n_res += 1
                        look_at[agt] = curr_pos[curr_spk]
                        if(n_res > dur_res): next_mode[agt] = 5         # ..respond after duration

                # Distracted mode
                elif(curr_mode[agt] == 2):
                    next_mode[agt] = 2
                    if(agt == next_spk):                                # if chosen next speaker..
                        n_res += 1
                        look_at[agt] = curr_pos[curr_spk]
                        if(n_res > dur_res): next_mode[agt] = 5         # ..respond after duration

                # Strongly Addressing mode
                elif(curr_mode[agt] == 3): next_mode[agt] = 0

                # Weakly Addressing mode
                elif(curr_mode[agt] == 4): next_mode[agt] = 0

                # Responding mode
                elif(curr_mode[agt] == 5):
                    curr_spk = agt
                    next_mode[agt] = 0                                  # speak immediately
                    n_spk, n_res = 0, 0
                    dur_spk = np.max((np_r.normal(self.dur_spk, 
                                      scale=dur_noise), 1))                    
                    look_at = curr_pos.copy()                           ## unsure of 'grp_gaze' part
                    look_at += grp_gaze if(np_r.rand()
                     < self.p_act) else (np_r.rand(2) - .5) * 2
            
            # Update and store
            next_gaze = self.shift_gaze(curr_pos, curr_gaze, look_at)
            curr_mode = next_mode.copy()
            curr_pos = next_pos.copy()
            curr_gaze = next_gaze.copy()
            self.memorize(curr_pos, curr_gaze, curr_mode, grp_ind)


    # Run interaction rules for all groups:
    #    `epi_start' chooses from where on in the
    #    remembered interactions to start (or restart) 
    #    episodes of length `epi_len' of interactions
    #    `dynamic' decides if members can move about the room 
    def run_all(self, epi_start = 'end', dynamic = False, epi_len = 300):
        for i, g in enumerate(self.groups):
            if(epi_start != 'end'):
                assert(type(epi_start) == int)
                del self.memory['pos'][i][epi_start+1:]
                del self.memory['gaze'][i][epi_start+1:]
                del self.memory['mode'][i][epi_start+1:]
            grp_pos  = self.memory['pos'][i][-1]
            grp_gaze = self.memory['gaze'][i][-1]
            grp_mode = np.where((self.memory['mode'][i])[-1])[1]
            self.run_grp(grp_pos, grp_gaze, grp_mode, dynamic,
                         i, epi_len)


    # Visualize interactions:
    #    either from loaded simulation file or self 
    #    for any subset of groups (vis_grps)
    #    at steps in episodes at intervals (stp_skip)
    def visualize(self, sim_file = '', vis_grps = 'all', stp_skip = 1):

        vis_obj = cPickle.load(open(sim_file)) if(sim_file != '') else self

        # Select subset of self properties acc to vis_grps
        if(vis_grps == 'all'): vis_grps = vis_obj.groups
        vis_sub = np.array([vis_obj.gID == i 
            for i in list(np.unique(vis_grps))]).any(axis=0)
        pos  =  vis_obj.pos[vis_sub]
        gaze = vis_obj.gaze[vis_sub]
        mode = vis_obj.mode[vis_sub]
        gID = vis_obj.gID[vis_sub]
        _, u = np.unique(gID, return_index=True)
        groups = list(gID[np.sort(u)])
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
        col_mode = ['r','g','y','r','r','black']
        for i in range(len(groups)):
            col = (1,1,1)
            if(sum(gID==groups[i]) != 1):
                col = np_r.choice(2,3)
                col[np_r.choice(3)] = np_r.rand()
            col_grp.append(tuple(col))

        # Update figure across time
        def update_plot(t):
            for i in range(n_agts):
                pos_patch[i]  = Circle(pos[i,:,t], radius=rad_p, 
                    fc=col_grp[groups.index(gID[i])], alpha = 0.4)
                mode_patch[i] = Circle(pos[i,:,t], radius=rad_m, 
                    fc=col_mode[list(mode[i,:,t]).index(1)])
                gaze_patch[i] = Arrow(pos[i,0,t], pos[i,1,t], 
                    25*gaze[i,0,t], 25*gaze[i,1,t], width=30, fc = 'k', ec='k')
            ax.clear()
            p_coll = PC(pos_patch+mode_patch+gaze_patch, match_original=True)
            #p_coll = PC(pos_patch+gaze_patch, match_original=True)
            ax.add_collection(p_coll)

        anim = animation.FuncAnimation(fig, update_plot, 
                        frames=range(0,pos.shape[2],stp_skip))
        #anim.save('simulation.mp4',writer=writer)
        plt.show()

    @property
    def mode(self):
        return np.hstack(self.memory['mode']).transpose(
            1,2,0)[self.orig_agts]
    
    @property
    def pos(self):
        return np.hstack(self.memory['pos']).transpose(
            1,2,0)[self.orig_agts]
    
    @property
    def gaze(self):
        return np.hstack(self.memory['gaze']).transpose(
            1,2,0)[self.orig_agts]