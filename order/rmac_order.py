import h5py
from . import utils
import numpy as np
from .order_base import OrderBase

class RMACOrder(OrderBase):
    def __init__(self, rmac_file, rmac_order_filename, normalize=False):
        super().__init__()
        print('Loading RMAC features...')
        self.rmac_feats = self.load_rmac_features(rmac_file, rmac_order_filename)
        if normalize:
            self.rmac_feats = utils.normalized(self.rmac_feats, 1)

    def load_rmac_features(self, feat_filename, feat_order_filename):
        features = h5py.File(feat_filename, 'r')['/rmac']
        img_names = open(feat_order_filename, 'r').readlines()
        
        assert len(features) == len(img_names)

        #takes only val features
        filtered = [feat for feat, name in zip(features, img_names) if 'val' in name]
        filtered = np.vstack(filtered)
        return filtered

    def compute_distances(self, query_img_index):
        query_feat = self.rmac_feats[query_img_index]
        distances = [utils.dot_dist(query_feat, f) for f in self.rmac_feats]

        return distances

    def get_name(self):
        return 'RMAC'

    def length(self):
        return len(self.rmac_feats)

#simple test
import os
if __name__ == "__main__":
    clevr_dir = '../features'
    idx = 6

    filename = os.path.join(clevr_dir,'clevr_rmac_features.h5')
    order_filename = os.path.join(clevr_dir,'clevr_rmac_features_order.txt')
    
    s = RMACOrder(filename, order_filename)
    print(s.get(idx))
        
