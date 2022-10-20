import numpy as np
import os, torch
from typing import Dict
from copy import deepcopy
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

def get_angle(hist_traj: np.ndarray):
    '''
    Get the traveling angle of an agent
    Args:
        hist_traj: historical trajectory of the agent with the length of t_h
    Return:
        angle: traveling angle of the agent considering the last two frames 
    '''
    angle1 = hist_traj[1:] - hist_traj[:-1]
    angle1 = angle1[-2:]
    angle = np.mean(angle1, axis = 0)
    
    return np.arctan2(angle[1], angle[0])

def discard_pos_outside(pose: np.ndarray, extent = [-2, 2, 0, 4]):
    '''
    Discard positions outside the neighborhood range within 1.6 meters (assumed)
    Args:
        1. pose: the most recent position of a neighbor
        2. extent: the range of interest, which can be revised
    Return:
        boolean variable True: within the range; and False: outside of the range 
    '''
    
    if extent[0] <= pose[0] <= extent[1] and extent[2] <= pose[1] <= extent[3]:
        return True
    else:
        return False
    
def get_normalized_traj(traj_instance: np.ndarray, origin: np.ndarray, angle: float):
    '''
    Get the normalized trajectory of an agent
    Args:
        1. traj_instance: historical trajectory of the agent with the length of t_h
        2. origin: central coordinates of the agent
        3. angle: traveling angle of the agent
    Return:
        angle: traveling angle of the agent considering the last two frames 
    '''
    local_traj = traj_instance - origin
    local_x, local_y = local_traj
    
    return rotate(local_x, local_y, -angle + np.radians(90))

def rotate(x, y, angle):
    '''
    Convert a global position into local
    Args: 
        
    '''
    res_x = x * np.cos(angle) - y * np.sin(angle)
    res_y = x * np.sin(angle) + y * np.cos(angle)
    return res_x, res_y
    
    
def read_file(file_dir: str):
    data = []
    with open(file_dir, "r") as f:
        for line in f:
            line = line.strip().split("\t")
            data.append(line)
    
    return np.array(data).astype(float)

def get_hist_traj(ped_id: int, last_obs_frame: int, frame_feats: Dict, t_h = 8):
    '''
    Get fully 8-frame historical trajectory for a agent
    Args:
        1. ped_id: unique pedestrian id number for consideration
        2. last_obs_frames: the last frame number
        3. frame_feats: dictionary recording frame-level features
        4. t_h: total number of historical frames, default = 8
        
    Return:
        1. Fully 8-frame historical trajectory of target agent
    '''
    
    ped_id_hist_feats = []
    for time in range(last_obs_frame, last_obs_frame + t_h):
        curr_feats = frame_feats[time]
        ped_curr_feat = curr_feats[curr_feats[:, 1] == ped_id]
        assert len(ped_curr_feat) == 1, "double check your dataset"
        ped_id_hist_feats.append(ped_curr_feat.reshape(-1, ))
    
    return np.array(ped_id_hist_feats)[:, 2:]

def get_fut_traj(ped_id: int, ref_frame: int, frame_feats: Dict, t_f = 8):
    '''
    Get fully 8-frame future trajectory for a agent
    Args:
        1. ped_id: unique pedestrian id number for consideration
        2. ref_frame: the reference frame number
        3. frame_feats: dictionary recording frame-level features
        4. t_h: total number of historical frames, default = 8
        
    Return:
        1. Fully 8-frame historical trajectory of target agent
        2. Masks indicate a valid future trajectory (1) or not (0)
    '''
    
    ped_id_future = []
    # masks_future  = np.zeros(t_f)
    for time in range(ref_frame, ref_frame + t_f):
        # stop the loop if the agent is not in frame features
        if time not in frame_feats:
            break
        curr_feats = frame_feats[time]            
        ped_curr_feat = curr_feats[curr_feats[:, 1] == ped_id]
        
        # stop the loop if the agent is not in frame features
        if len(ped_curr_feat) == 0:
            break
        ped_id_future.append(ped_curr_feat.reshape(-1, ))
    
    # make masks based on the length of ped future trajectory
    # masks_future[: len(ped_id_future)] = 1
    return np.array(ped_id_future)[:, 2:] if len(ped_id_future) > 0 else np.array(ped_id_future)


def get_traj_instance(sample_file: np.ndarray, t_h = 8, t_f = 8):
    '''
    Function to extract historical and future trajectory instances
    Args:
        sample_file: processed text file in np.ndarray format
    Return:
        1. all_ped_ids: all ped ids in the last observed frame
        2. all_ped_feats: all ped historical trajectories in [t, t + t_h - 1]
        3. all_ped_future: all ped future trajectories in [t + t_h, t + t_f]
        4. all_masks: all ped masks indicate a valid future trajectory (1) or not (0)
    '''

    # record unique frames in the sample file
    frames = np.unique(sample_file[:, 0])

    # prepare a reference dictionary to record frame-level features
    frame_feats = {}
    for frame in frames:
        curr_feats = sample_file[sample_file[:, 0] == frame]
        frame_feats[frame] = curr_feats
    
    # local total time and global time
    t_total = t_h + t_f
    keys = list(frame_feats.keys())
    start_frame = int(min(keys))
    frame_total = int(max(keys))
    
    # feature dictionaries
    #all_masks  = {}
    #all_ped_ids = {}
    all_ped_feats = {}
    all_ped_future = {}
    
    # loop over plausible time frames 
    for idx in range(start_frame, frame_total - t_total + 1):
        # ped_ids in the last observed frame
        obs_last_ped_ids = sample_file[sample_file[:, 0] == idx][:, 1]
        # ped_ids in the most recent observed frame
        curr_ped_ids     = sample_file[sample_file[:, 0] == idx + t_h - 1][:, 1]
        # filter out ped_ids in consideration with fully 8 frames of observed trajectory
        filter_ped_ids   = np.intersect1d(obs_last_ped_ids, curr_ped_ids)
        #all_ped_ids[idx] = filter_ped_ids

        # filter out frames with no peds included
        if len(filter_ped_ids) == 0:
            continue

        # record all ped features in one_frame
        all_ped_feats_in_one_frame = {}
        all_ped_future_in_one_frame = {}
        #masks_in_one_frame          = {}

        for ped_id in filter_ped_ids:

            # get_hist_feats of the target
            ped_id_feats  = get_hist_traj(ped_id, idx, frame_feats)
            ped_id_future = get_fut_traj(ped_id, idx + t_h, frame_feats)

            # stack features in frame
            all_ped_feats_in_one_frame[ped_id] = ped_id_feats
            all_ped_future_in_one_frame[ped_id] = ped_id_future
            # masks_in_one_frame[ped_id] = masks

        # wrap up frame features together
        all_ped_feats[idx]  = all_ped_feats_in_one_frame
        all_ped_future[idx] = all_ped_future_in_one_frame
        #all_masks[idx]      = masks_in_one_frame

    return preprocess(all_ped_feats, all_ped_future)


def preprocess(ped_hist_traj: Dict, ped_fut_traj: Dict, t_f = 8):
    
    '''
    Preprocess the dataset to extract agent features and neighborhood representations.
    Args:
        1. ped_hist_traj: historical trajectory in t_h frames
        2. ped_fut_traj:  future trajectory in t_f frames
        3. ped_fut_masks: future trajectory masks in t_f frames
    Return:
        mapping: dictionary including essencial representations of the agent
    '''
    
    
    # dictionary to record input features of Social LSTM
    mapping = {}
    i = 0 
    
    for time, values in ped_hist_traj.items():

        idx_set = set(values.keys())             # record the set of ids
        
        # input instance of social lstm
        
        for ped_id, ped_features in values.items():
            
            temp = {}
            mapping[i] = temp
            # record the central coordinates 
            temp['cent_coord'] = ped_features[-1]

            # record the traveling angle of the agent
            traveling_angle = get_angle(ped_features)
            temp['angle']   = traveling_angle
            
            # current historical trajectory of the agent
            curr_traj   = deepcopy(ped_features)
            future_traj = deepcopy(ped_fut_traj[time][ped_id])
            
            # normalize neighbors' trajectories
            norm_instance = [get_normalized_traj(instance, temp['cent_coord'], traveling_angle) 
                             for instance in curr_traj]
            norm_future   = [get_normalized_traj(instance, temp['cent_coord'], traveling_angle) 
                             for instance in future_traj]
            
            #norm_future   = np.zeros([t_f, 2]) if len(norm_future) == 0 else np.vstack([norm_future, np.zeros([t_f-norm_future.shape[0], 2])])
            
            temp['agent_traj_hist']   = np.array(norm_instance)
            temp['agent_traj_future'] = np.array(norm_future)
            # temp['agent_traj_masks']  = ped_fut_masks[time][ped_id]
            
            # temporarily remove agent ped_id to record his/her neigbor(s)
            idx_set.remove(ped_id)
            neighbor_lst = []
            #neighbor_idx = []

            for idx in idx_set:

                # normalized neighbor historical trajectories
                
                nbr_ins        = ped_hist_traj[time][idx]
                norm_nbr_ins   = deepcopy(nbr_ins)
                norm_nbrs_ins  = np.array([get_normalized_traj(instance, temp['cent_coord'], traveling_angle) 
                                           for instance in norm_nbr_ins])
                
                # discard positions outside of the neighborhood grid
                if not discard_pos_outside(norm_nbrs_ins[-1]):
                    continue
                    
                neighbor_lst.append(norm_nbrs_ins)
                # neighbor_idx.append(idx)
            
            # resume the idx_set
            idx_set.add(ped_id)
            
            # record neighbors of the agent
            temp['neighbor_feats'] = np.array(neighbor_lst)
            #temp['neighbor_ids'] = neighbor_idx
            
            i += 1

    return mapping
    
    
class pedDataset(Dataset):
    
    '''
    Dataloader for pedestrian model
    '''
    
    def __init__(self, data_dir, data_loc, t_h = 8, t_f = 8, grid = [20, 20], enc_size = 64):
    
        # data_loc           = os.listdir(data_dir)
        self.t_h           = t_h
        self.t_f           = t_f
        sample_file        = read_file(os.path.join(data_dir, data_loc))                ### change it later for loop file extraction
        sample_file[:, 0] /= 10
        self.mapping       = get_traj_instance(sample_file)
        self.grid          = grid
        self.enc_size      = enc_size
        
    def __getitem__(self, item):
        inputs = self.mapping[item]
        return inputs['agent_traj_hist'], inputs['agent_traj_future'], inputs['neighbor_feats']
    
    def __len__(self):
        return len(self.mapping)
    
    def collate_fn(self, samples):
        
        batch_nbrs_size = 0
        for _, _, nbrs in samples:
            batch_nbrs_size += sum([len(nbr) != 0 for nbr in nbrs])
        
        maxlen = self.t_h
        nbrs_batch = torch.zeros(maxlen, batch_nbrs_size, 2)
                
        pos = [0, 0]
        mask_batch = torch.zeros(len(samples), self.grid[0], self.grid[1], self.enc_size)
        mask_batch = mask_batch.byte()


        # Initialize history, history lengths, future, output mask, lateral maneuver and longitudinal maneuver batches:
        hist_batch = torch.zeros(maxlen, len(samples), 2)
        fut_batch = torch.zeros(self.t_f, len(samples), 2)
        op_mask_batch = torch.zeros(self.t_f, len(samples), 2)        
        
        
        for sampleId, (hist, fut, nbrs) in enumerate(samples):
            
            hist_batch[:len(hist), sampleId, 0] = torch.from_numpy(hist[:, 0]).float()
            hist_batch[:len(hist), sampleId, 1] = torch.from_numpy(hist[:, 1]).float()
            
            if len(fut) > 0:
                fut_batch[:len(fut), sampleId, 0]   = torch.from_numpy(fut[:, 0]).float()
                fut_batch[:len(fut), sampleId, 1]   = torch.from_numpy(fut[:, 1]).float()

            op_mask_batch[:len(fut), sampleId, :]   = 1
            
            count = 0
            for nbr_feat in nbrs:
                
                if len(nbr_feat) > 0:
                    nbrs_batch[:len(nbr_feat), count, 0]    = torch.from_numpy(nbr_feat[:, 0]).float()
                    nbrs_batch[:len(nbr_feat), count, 1]    = torch.from_numpy(nbr_feat[:, 1]).float()
                    
                    pos_x, pos_y = int(nbr_feat[-1, 0]//0.2), int(nbr_feat[-1, 1]//0.2)
                    pos[0] = pos_x + 10 if pos_x >= 0 else abs(pos_x) - 9
                    pos[1] = pos_y
                    mask_batch[sampleId, pos[0], pos[1], :] = torch.ones(self.enc_size)
                    
                    count += 1
            
        return hist_batch, fut_batch, op_mask_batch, nbrs_batch, mask_batch
             
    def collate_fn_omap(self, samples):
        
        maxlen = self.t_h
                        
        pos = [0, 0]

        # Initialize history, history lengths, future, output mask, lateral maneuver and longitudinal maneuver batches:
        hist_batch = torch.zeros(maxlen, len(samples), 2)
        fut_batch = torch.zeros(self.t_f, len(samples), 2)
        op_mask_batch = torch.zeros(self.t_f, len(samples), 2)
        
        mask_batch = torch.zeros(maxlen, len(samples), self.grid[0], self.grid[1], 2)
        #mask_batch = mask_batch.byte()
        
        for sampleId, (hist, fut, nbrs) in enumerate(samples):
            
            hist_batch[:len(hist), sampleId, 0] = torch.from_numpy(hist[:, 0]).float()
            hist_batch[:len(hist), sampleId, 1] = torch.from_numpy(hist[:, 1]).float()
            
            if len(fut) > 0:
            
                fut_batch[:len(fut), sampleId, 0]   = torch.from_numpy(fut[:, 0]).float()
                fut_batch[:len(fut), sampleId, 1]   = torch.from_numpy(fut[:, 1]).float()

            op_mask_batch[:len(fut), sampleId, :]   = 1
            
            count = 0
            for nbr_feat in nbrs:
                
                if len(nbr_feat) > 0:
                    
                    pos_x, pos_y = int(nbr_feat[-1, 0]//0.1), int(nbr_feat[-1, 1]//0.1)
                    pos[0] = pos_x + 15 if pos_x > 0 else abs(pos_x)
                    pos[1] = pos_y + 15 if pos_y > 0 else abs(pos_y)
                    mask_batch[:len(nbr_feat), sampleId, pos[0], pos[1], 0] = torch.from_numpy(nbr_feat[:, 0]).float()
                    mask_batch[:len(nbr_feat), sampleId, pos[0], pos[1], 1] = torch.from_numpy(nbr_feat[:, 1]).float()
                    
                    count += 1
            
        return hist_batch, fut_batch, op_mask_batch, mask_batch
        
          

