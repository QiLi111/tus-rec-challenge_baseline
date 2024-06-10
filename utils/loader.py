
import random
import json,os

import h5py
import numpy as np


class Dataset():

    def __init__(self, data_path,num_samples, sample_range, indices_in_use=None):

        """
        :param data_path, file path
        :param indices_in_use: 
            case 1: a list of tuples (idx_subject, idx_scans), indexing self.num_frames[indices_in_use[idx]]
            case 2: a list of two lists, [indices_subjects] and [indices_scans], meshgrid to get indices
            case 3: None (default), use all available in the file
        
        Sampling parameters
        :param num_samples: type int, number of (model input) frames, > 1. However, when num_samples=-1, sample all in the scan
        :param sample_range: type int, range of sampling frames, default is num_samples; should not larger than the number of frames in a scan
        """
        self.data_path=data_path
        self.subs = [f for f in os.listdir(self.data_path) if os.path.isdir(os.path.join(self.data_path, f))]
        self.subs = sorted(self.subs)
        # scans in each folder have the same set of names
        self.scans = [f for f in os.listdir(os.path.join(self.data_path, self.subs[0])) if f.endswith(".h5")]
        
        if indices_in_use is None:
            # use all the scans
            self.indices_in_use = [(i_sub,i_scn) for i_sub in range(len(self.subs)) for i_scn in range(len(self.scans))]                    

        # use selected scans
        elif all([isinstance(t,tuple) for t in indices_in_use]):
            self.indices_in_use = indices_in_use
        elif isinstance(indices_in_use[0],list) and isinstance(indices_in_use[1],list):
            self.indices_in_use = [(i_sub,i_scn) for i_sub in indices_in_use[0] for i_scn in indices_in_use[1]]            
        else:
            raise("indices_in_use should be a list of tuples (idx_subject, idx_scans) of two lists, [indices_subjects] and [indices_scans].")
        
        if len(set(self.indices_in_use)) != len(self.indices_in_use):
            print("WARNING: Replicated indices are found - not removed.")
        
        self.indices_in_use.sort()
        self.num_indices = len(self.indices_in_use)

        # sampling parameters
        if num_samples < 2:
            if num_samples == -1:
                if sample_range is not None:
                    sample_range = None
                    print("Sampling all frames. sample_range is ignored.")
            else:
                raise('num_samples should be greater than or equal to 2, or -1 for sampling all frames.')
        self.num_samples = num_samples
        
        if sample_range is None:
            self.sample_range = self.num_samples
        else:
            self.sample_range = sample_range

    def partition_by_ratio(self, ratios, randomise=False):
        # partition the dataset into train, val, and test sets
        """
        :param ratios, the ratio for train, val, and test sets
        :param randomise: suffer the dataset or not

        """
        
        num_sets = len(ratios)
        ratios = [ratios[i]/sum(ratios) for i in range(num_sets)]
        print("Partitioning into %d sets with a normalised ratios %s," %
              (num_sets, ratios))

        # subject-level split
        set_sizes = [int(len(self.subs)*r) for r in ratios]
        sub_ind = list(range(len(self.subs)))
        for ii in range(len(self.subs)-sum(set_sizes)): 
            set_sizes[ii]+=1
        if randomise:
            random.Random(4).shuffle(sub_ind)

        # indices_in_use_list = [list(ele) for ele in self.indices_in_use]
        # indices_sets_sub = [sub_ind[n0:n0+n1] for (n0, n1) in zip([sum(set_sizes[:ii]) for ii in range(num_sets)], set_sizes)]  # get the index tuples for all sets
        
        # first element of self.indices_in_use
        fir_ele = [i_idx[0] for i_idx in self.indices_in_use]
        idx_all = [np.where(np.array(fir_ele)==sub_ind[i])[0] for i in range(len(sub_ind))]
        idx_all = [l.tolist() for l in idx_all]
        indices_sets= None
        for (n0, n1) in zip([sum(set_sizes[:ii]) for ii in range(num_sets)], set_sizes):  # get the index tuples for all sets
            idx_list = [x for xs in idx_all[n0:n0+n1] for x in xs]
            if indices_sets==None:
                indices_sets = [[self.indices_in_use[idx_list_i] for idx_list_i in idx_list]]
            else:
                indices_sets.append([self.indices_in_use[idx_list_i] for idx_list_i in idx_list])

        print("at subject-level, with %s subjects." % (set_sizes))

        return [Dataset(data_path=self.data_path, num_samples=self.num_samples, sample_range=self.sample_range, indices_in_use=idx_list) for idx_list in indices_sets]




    def __add__(self, other):
        # add two dataset
        if self.data_path != other.data_path:
            raise('Currently different file combining is not supported.')
        if self.num_samples != other.num_samples:
            print('WARNING: found different num_samples - the first is used.')
        if self.sample_range != other.sample_range:
            print('WARNING: found different sample_range - the first is used.')
        indices_combined = self.indices_in_use + other.indices_in_use
        return Dataset(data_path = self.data_path, num_samples=self.num_samples, sample_range=self.sample_range, indices_in_use=indices_combined)
    

    def __len__(self):
        return self.num_indices
    

    def __getitem__(self, idx):

        indices = self.indices_in_use[idx]

        scans = [f for f in os.listdir(os.path.join(self.data_path, self.subs[indices[0]])) if f.endswith(".h5")]
        if len(scans) != 24:
            raise("Each subjects should have 24 scans")
        fn_mha=sorted(scans)

        h5file = h5py.File(os.path.join(self.data_path,self.subs[indices[0]],fn_mha[indices[1]]), 'r')
        frames = h5file['frames']
        tforms = h5file['tforms']
        scan_name = fn_mha[indices[1]][:-3]
        
        if self.num_samples == -1:  # sample all available frames, for validation
            return frames[()],tforms[()],indices,scan_name

        else:
            # sample a sequence of frames
            i_frames = self.frame_sampler(len(frames))
            return frames[i_frames],tforms[i_frames],indices,scan_name

    def frame_sampler(self, n):
        """
        sample sequence from scan
        :param n, the length of the population that to be sampled

        """
        n0 = random.randint(0,n-self.sample_range)  # sample the start index for the range
        idx_frames = random.sample(range(n0,n0+self.sample_range), self.num_samples)   # sample indices
        idx_frames.sort()
        return idx_frames
    

    def write_json(self, jason_filename):
        # write the dataset information to a json file
        with open(jason_filename, 'w', encoding='utf-8') as f:
            json.dump({
                "data_path" : self.data_path, 
                "indices_in_use" : self.indices_in_use,
                "num_samples": self.num_samples,
                "sample_range": self.sample_range
                }, f, ensure_ascii=False, indent=4)
        print("%s written." % jason_filename)
    
    
    @staticmethod
    def read_json(jason_filename,num_samples = None):
        # read the dataset information from a json file
        with open(jason_filename, 'r', encoding='utf-8') as f:
            obj = json.load(f)
            if num_samples is None:
                num_samples = obj['num_samples']
            return Dataset(
                data_path=obj['data_path'],
                num_samples=num_samples, 
                sample_range=obj['sample_range'],
                indices_in_use = [tuple(ids) for ids in obj['indices_in_use']], # convert to tuples from json string
                )
   
   
   

   
   
        
            