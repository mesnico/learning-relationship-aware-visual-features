import pickle
import os
import numpy as np
import pdb
import networkx as nx
from scipy.stats import spearmanr, kendalltau
import math
import time
from order import rn_order, rmac_order, graphs_order, states_order, graphs_approx_order
from order import utils
import argparse
import metrics
#import networkx.algorithms.similarity

def recall_at(gt_list, c_list, k=10):
    if len(gt_list) != len(c_list):
        raise ValueError('Dimension mismatch: the two lists should have same length')
    if k > len(gt_list):
        k = len(gt_list)
    gt_set = set(gt_list[0:k])
    c_set = set(c_list[0:k])
    
    diff = gt_set.intersection(c_set)
    return len(diff)/k

def compute_ranks(feat_orders, gt_order, query_img_index, include_query=False):
    stat_indexes = []
    max_feats_len, min_feats_len = utils.max_min_length(feat_orders + [gt_order])

    #prepare the axis for computing log-scale recall-at-k.
    log_base = 1.3
    logscale_ub = np.floor(math.log(max_feats_len,log_base))
    k_logscale_axis = np.floor(np.power(log_base,np.array(range(int(logscale_ub)))))
    k_logscale_axis = np.unique(k_logscale_axis)
    k_logscale_axis = k_logscale_axis.astype(int)

    feat_distances = utils.build_feat_dict(feat_orders, query_img_index, min_length=min_feats_len, include_query=include_query)
    gt_distance = utils.build_feat_dict([gt_order], query_img_index, min_length=min_feats_len, include_query=include_query)

    assert len(gt_distance) == 1, 'More than one Ground-Truth!'
    dist_gt, _, perm_gt = list(gt_distance.values())[0]

    for name, (dist, _, permut) in feat_distances.items():
        #calculate stats for every feature
        k_logscale = {k:recall_at(permut, perm_gt, k) for k in k_logscale_axis}
        
        norm_gt_similarities = 1 - (dist_gt / max(dist_gt))
        norm_similarities = 1 - (dist / max(dist))
        
        stat_indexes.append({'label': name,
                    'kendall-tau': kendalltau(dist, dist_gt)[0],
                    'spearmanr': spearmanr(dist, dist_gt)[0],
                    'nDCG': metrics.ndcg_score(norm_gt_similarities, norm_similarities, 5),
                    'recall-at-10': recall_at(permut, perm_gt, 10),
                    'recall-at-100': recall_at(permut, perm_gt, 100),
                    'recall-at-1000': recall_at(permut, perm_gt, 1000),
                    'recall-at-k': dict(k_logscale)})

    return stat_indexes

def print_stats(stats, gt, idx):
    for stat in stats:
        print('## Query idx: {} ## - Correlation among {} and actual GT: {}\n\tspearman-rho: {}\n\tkendall-tau: {}\n\tnDCG: {}\n\trecall-at-10: {}\n\trecall-at-100: {}\n\trecall-at-1000: {}'.format(
            idx,
            stat['label'],
            gt,
            stat['spearmanr'],
            stat['kendall-tau'],
            stat['nDCG'],
            stat['recall-at-10'],
            stat['recall-at-100'],
            stat['recall-at-1000']))
        
#show images only if we are threating only one image    
   
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Similarity search and stats recording')
    parser.add_argument('--query-img-index', type=int, default=0,
                        help='index of the image to use as query')
    parser.add_argument('--until-img-index', type=int, default=10,
                        help='index of the last image to use as query (for stats recording)')
    parser.add_argument('--normalize', action='store_true', default=False,
                        help='enables features normalization')
    parser.add_argument('--graph-ground-truth', type=str, choices=['proportional','atleastone'], default='proportional',
                        help='ground truth if graph GT is used')
    parser.add_argument('--ground-truth', type=str, choices=['graph','graph-approx','states'], default='graph',
                        help='which GT to use')
    parser.add_argument('--clevr-dir', type=str, default='.',
                        help='CLEVR dataset base dir')
    parser.add_argument('--cpus', type=int, default=8,
                        help='how many CPUs to use for graph distance calculation')
    parser.add_argument('--skip-missing', action='store_true', default=False,
                        help='skip missing cached distances')
    parser.add_argument('--set', type=str, choices=['train','test'], default='test', help='Which set use among training and test')
    parser.add_argument('--verbose', action='store_true', default=False,
                        help='Verbose output')
    args = parser.parse_args()

    feats_dir = './features'    

    #prepare cache folder
    cache_dir='./dist_cache'
    try:
        os.makedirs(cache_dir)
    except:
        print('directory {} already exists'.format(cache_dir))

    #prepare the stats directory
    stats_dir = './stats'
    try:
        os.makedirs(stats_dir)
    except:
        print('directory {} already exists'.format(stats_dir))

    stats_out = {}

    #initialize orders objects
    print('Initializing feature order objects...')
    feats_orders = []
    how_many=15000

    if args.set == 'test':
        #TODO: check arguments to all functions in test
        ### FROM TEST SET - SD ###
        '''feats_orders.append(rn_order.RNOrder(os.path.join(feats_dir,'max_features_conv_sd.pickle'), 'conv\nmax sd', args.normalize))
        feats_orders.append(rn_order.RNOrder(os.path.join(feats_dir,'avg_features_conv_sd.pickle'), 'conv\navg sd', args.normalize))
        feats_orders.append(rn_order.RNOrder(os.path.join(feats_dir,'test_gfc1_avg_features_sd.pickle'), 'g_fc1\navg sd', args.normalize))
        feats_orders.append(rn_order.RNOrder(os.path.join(feats_dir,'test_gfc1_max_features_sd.pickle'), 'g_fc1\nmax sd', args.normalize))
        feats_orders.append(rn_order.RNOrder(os.path.join(feats_dir,'test_gfc2_avg_features_sd.pickle'), 'g_fc2\navg sd', args.normalize))
        feats_orders.append(rn_order.RNOrder(os.path.join(feats_dir,'test_gfc2_max_features_sd.pickle'), 'g_fc2\nmax sd', args.normalize))
        feats_orders.append(rn_order.RNOrder(os.path.join(feats_dir,'afteraggr_features_sd.pickle'), 'afteraggr\nprenorm\nsd', args.normalize))
        feats_orders.append(rn_order.RNOrder(os.path.join(feats_dir,'afteraggr-no-prenorm_features_sd.pickle'), 'afteraggr\nno-prenorm\nsd', args.normalize))'''

        ### FROM TEST SET - FP ###
        #feats_orders.append(rn_order.RNOrder(os.path.join(feats_dir,'test_maxconv_features_fp.pickle'), 'conv\nmax fp', args.normalize))
        feats_orders.append(rn_order.RNOrder(os.path.join(feats_dir,'test_avgconv_features_fp.pickle'), 'conv\navg fp', args.normalize))

        '''feats_orders.append(rn_order.RNOrder(os.path.join(feats_dir,'test_gfc1_avg_features_sd.pickle'), 'g_fc1\navg sd', args.normalize))
        feats_orders.append(rn_order.RNOrder(os.path.join(feats_dir,'test_gfc1_max_features_sd.pickle'), 'g_fc1\nmax fp', args.normalize))'''
        feats_orders.append(rn_order.RNOrder(os.path.join(feats_dir,'gfc2_avg_features_fp.pickle'), 'g_fc2\navg fp', args.normalize))
        #feats_orders.append(rn_order.RNOrder(os.path.join(feats_dir,'gfc2_max_features_fp.pickle'), 'g_fc2\nmax fp', args.normalize))
        
        #feats_orders.append(rn_order.RNOrder(os.path.join(feats_dir,'gfc4_innetaggreg_qinj4_aggr4_1024dim_bidir1_features_fp.pickle'), 'afteraggr\nbidir1 fp', args.normalize))
        feats_orders.append(rn_order.RNOrder(os.path.join(feats_dir,'gfc4_innetaggreg_qinj5_aggr4_512dim_bidir2_features_fp.pickle'), 'afteraggr\nbidir2 fp', args.normalize))
        #feats_orders.append(rn_order.RNOrder(os.path.join(feats_dir,'gfc4_innetaggreg_qinj5_aggr4_256dim_bidir3_features_fp.pickle'), 'afteraggr\nbidir3 fp', args.normalize))
        #feats_orders.append(rn_order.RNOrder(os.path.join(feats_dir,'gfc4_innetaggreg_qinj5_aggr4_1024dim_bidir4_features_fp.pickle'), 'afteraggr\nbidir4 fp', args.normalize))
        #feats_orders.append(rn_order.RNOrder(os.path.join(feats_dir,'gfc4_innetaggreg_qinj5_aggr4_1024dim_bidir7_features_fp.pickle'), 'afteraggr\nbidir7 fp', args.normalize))
        #feats_orders.append(rn_order.RNOrder(os.path.join(feats_dir,'gfc3_innetaggreg_qinj4_aggr3_512dim_bidir8_features_fp.pickle'), 'afteraggr\nbidir8 fp', args.normalize))
        #feats_orders.append(rn_order.RNOrder(os.path.join(feats_dir,'gfc4_innetaggreg_qinj5_aggr4_512dim_bidir9_features_fp.pickle'), 'afteraggr\nbidir9 fp', args.normalize))
        #feats_orders.append(rn_order.RNOrder(os.path.join(feats_dir,'gfc4_innetaggreg_qinj5_aggr4_2048dim_bidir10_features_fp.pickle'), 'afteraggr\nbidir10 fp', args.normalize))
        #feats_orders.append(rn_order.RNOrder(os.path.join(feats_dir,'gfc4_innetaggreg_qinj5_aggr4_4096dim_bidir11_features_fp.pickle'), 'afteraggr\nbidir11 fp', args.normalize))
        feats_orders.append(rn_order.RNOrder(os.path.join(feats_dir,'gfc4_innetaggreg_qinj5_aggr4_512dim_bidir2_transferlearn_weightedsum_features_fp.pickle'), 'afteraggr\nbidir2\nTL-WS\nfp', args.normalize))
        feats_orders.append(rn_order.RNOrder(os.path.join(feats_dir,'gfc4_innetaggreg_qinj5_aggr4_512dim_bidir2_transferlearn_weightedsum_epoch795_features_fp.pickle'), 'afteraggr\nbidir2\nTL-WS\nep795 fp', args.normalize))
        feats_orders.append(rn_order.RNOrder(os.path.join(feats_dir,'G2V_bs32_features_test.pickle'), 'G2V', args.normalize))
        feats_orders.append(rn_order.RNOrder(os.path.join(feats_dir,'G2V_node_dropout_bs64_features_test.pickle'), 'G2V\nnode\ndropout', args.normalize))
        feats_orders.append(rn_order.RNOrder(os.path.join(feats_dir,'G2V_node_dropout_bs64_contrastive_features_test.pickle'), 'G2V\nnode\ndropout\ncontrastive', args.normalize))
        feats_orders.append(rn_order.RNOrder(os.path.join(feats_dir,'G2V_bs64_features_test.pickle'), 'G2V\nbs64\nself-loops', args.normalize))
        feats_orders.append(rn_order.RNOrder(os.path.join(feats_dir,'G2V_contrastive_selfatt_mpnn1.pickle'), 'G2V\nself-att\nMPNN', args.normalize))
        feats_orders.append(rn_order.RNOrder(os.path.join(feats_dir,'G2V_contrastive_mpnn1_selfatt_belilovskyloss.pickle'), 'G2V\nself-att\nMPNN\nbelilovsky', args.normalize))
        feats_orders.append(rn_order.RNOrder(os.path.join(feats_dir,'G2V_contrastive_mpnn1_selfatt_mean.pickle'), 'G2V\nself-att\nMPNN\nmean', args.normalize))
        feats_orders.append(rn_order.RNOrder(os.path.join(feats_dir,'G2V_contrastive_mpnn1_selfatt_norm.pickle'), 'G2V\nself-att\nMPNN\nnorm', args.normalize))
        feats_orders.append(rn_order.RNOrder(os.path.join(feats_dir,'G2V_contrastive_mpnn_GRUCell_no-selfatt.pickle'), 'G2V\nMPNN\nGRUCell\nnorm', args.normalize))        

        feats_orders.append(rn_order.RNOrder(os.path.join(feats_dir,'random_baseline.pickle'), 'RAND', args.normalize))

        #WITH PCA
        '''feats_orders.append(rn_order.RNOrder(os.path.join(feats_dir,'gfc1_avg_features_sd.pickle'), 'g_fc1\navg sd', args.normalize, how_many, args.set, 'pca', pca_dims=16))
        feats_orders.append(rn_order.RNOrder(os.path.join(feats_dir,'gfc1_max_features_sd.pickle'), 'g_fc1\nmax sd', args.normalize, how_many, args.set, 'pca', pca_dims=16))
        feats_orders.append(rn_order.RNOrder(os.path.join(feats_dir,'gfc2_avg_features_sd.pickle'), 'g_fc2\navg sd', args.normalize, how_many, args.set, 'pca', pca_dims=16))
        feats_orders.append(rn_order.RNOrder(os.path.join(feats_dir,'gfc2_max_features_sd.pickle'), 'g_fc2\nmax sd', args.normalize, how_many, args.set, 'pca', pca_dims=16))
        feats_orders.append(rn_order.RNOrder(os.path.join(feats_dir,'test_gfc1_avg_features_sd.pickle'), 'g_fc1\navg sd', args.normalize, how_many, args.set, 'pca', pca_dims=16))
        feats_orders.append(rn_order.RNOrder(os.path.join(feats_dir,'test_gfc1_max_features_sd.pickle'), 'g_fc1\nmax sd', args.normalize, how_many, args.set, 'pca', pca_dims=16))
        feats_orders.append(rn_order.RNOrder(os.path.join(feats_dir,'test_gfc2_avg_features_sd.pickle'), 'g_fc2\navg sd', args.normalize, how_many, args.set, 'pca', pca_dims=16))
        feats_orders.append(rn_order.RNOrder(os.path.join(feats_dir,'test_gfc2_max_features_sd.pickle'), 'g_fc2\nmax sd', args.normalize, how_many, args.set, 'pca', pca_dims=16))
        feats_orders.append(rn_order.RNOrder(os.path.join(feats_dir,'afteraggr_features_sd.pickle'), 'afteraggr\nprenorm\nsd', args.normalize, how_many, args.set, 'pca', pca_dims=256))
        feats_orders.append(rn_order.RNOrder(os.path.join(feats_dir,'afteraggr-no-prenorm_features_sd.pickle'), 'afteraggr\nno-prenorm\nsd', args.normalize, how_many, args.set, 'pca', pca_dims=256))'''

        


    '''elif args.set == 'train':
        ### FROM TRAIN ###
        feats_orders.append(rn_order.RNOrder(os.path.join(feats_dir,'train_gfc2_avg_features_fp.pickle'), 'g_fc2\navg fp', args.normalize, how_many, args.set))
        feats_orders.append(rn_order.RNOrder(os.path.join(feats_dir,'train_gfc2_max_features_fp.pickle'), 'g_fc2\nmax fp', args.normalize, how_many, args.set))
        feats_orders.append(rn_order.RNOrder(os.path.join(feats_dir,'train_gfc1_avg_features_fp.pickle'), 'g_fc1\navg fp', args.normalize, how_many, args.set))
        feats_orders.append(rn_order.RNOrder(os.path.join(feats_dir,'train_gfc1_max_features_fp.pickle'), 'g_fc1\nmax fp', args.normalize, how_many, args.set))
        feats_orders.append(rn_order.RNOrder(os.path.join(feats_dir,'train_gfc0_avg_features_fp_orig.pickle'), 'conv\navg fp\noriginal', args.normalize, how_many, args.set))
        feats_orders.append(rn_order.RNOrder(os.path.join(feats_dir,'train_gfc0_max_features_fp_orig.pickle'), 'conv\nmax fp\noriginal', args.normalize, how_many, args.set))

        # WITH PCA#
        feats_orders.append(rn_order.RNOrder(os.path.join(feats_dir,'train_gfc2_avg_features_fp.pickle'), 'g_fc2\navg fp', args.normalize, how_many, args.set, 'pca'))
        feats_orders.append(rn_order.RNOrder(os.path.join(feats_dir,'train_gfc2_max_features_fp.pickle'), 'g_fc2\nmax fp', args.normalize, how_many, args.set, 'pca'))
        feats_orders.append(rn_order.RNOrder(os.path.join(feats_dir,'train_gfc1_avg_features_fp.pickle'), 'g_fc1\navg fp', args.normalize, how_many, args.set, 'pca'))
        feats_orders.append(rn_order.RNOrder(os.path.join(feats_dir,'train_gfc1_max_features_fp.pickle'), 'g_fc1\nmax fp', args.normalize, how_many, args.set, 'pca'))
        feats_orders.append(rn_order.RNOrder(os.path.join(feats_dir,'train_gfc0_avg_features_fp_orig.pickle'), 'conv\navg fp\noriginal', args.normalize, how_many, args.set, 'pca'))
        feats_orders.append(rn_order.RNOrder(os.path.join(feats_dir,'train_gfc0_max_features_fp_orig.pickle'), 'conv\nmax fp\noriginal', args.normalize, how_many, args.set, 'pca'))

        feats_orders.append(rn_order.RNOrder(os.path.join(feats_dir,'train_gfc2_avg_features_fp.pickle'), 'g_fc2\navg fp', args.normalize, how_many, args.set, 'pca', pca_dims=32))
        feats_orders.append(rn_order.RNOrder(os.path.join(feats_dir,'train_gfc2_max_features_fp.pickle'), 'g_fc2\nmax fp', args.normalize, how_many, args.set, 'pca', pca_dims=32))
        feats_orders.append(rn_order.RNOrder(os.path.join(feats_dir,'train_gfc1_avg_features_fp.pickle'), 'g_fc1\navg fp', args.normalize, how_many, args.set, 'pca', pca_dims=32))
        feats_orders.append(rn_order.RNOrder(os.path.join(feats_dir,'train_gfc1_max_features_fp.pickle'), 'g_fc1\nmax fp', args.normalize, how_many, args.set, 'pca', pca_dims=32))
        feats_orders.append(rn_order.RNOrder(os.path.join(feats_dir,'train_gfc0_avg_features_fp_orig.pickle'), 'conv\navg fp\noriginal', args.normalize, how_many, args.set, 'pca', pca_dims=16))
        feats_orders.append(rn_order.RNOrder(os.path.join(feats_dir,'train_gfc0_max_features_fp_orig.pickle'), 'conv\nmax fp\noriginal', args.normalize, how_many, args.set, 'pca', pca_dims=16))

        feats_orders.append(rn_order.RNOrder(os.path.join(feats_dir,'train_gfc2_avg_features_fp.pickle'), 'g_fc2\navg fp', args.normalize, how_many, args.set, 'pca', pca_dims=16))
        feats_orders.append(rn_order.RNOrder(os.path.join(feats_dir,'train_gfc2_max_features_fp.pickle'), 'g_fc2\nmax fp', args.normalize, how_many, args.set, 'pca', pca_dims=16))
        feats_orders.append(rn_order.RNOrder(os.path.join(feats_dir,'train_gfc1_avg_features_fp.pickle'), 'g_fc1\navg fp', args.normalize, how_many, args.set, 'pca', pca_dims=16))
        feats_orders.append(rn_order.RNOrder(os.path.join(feats_dir,'train_gfc1_max_features_fp.pickle'), 'g_fc1\nmax fp', args.normalize, how_many, args.set, 'pca', pca_dims=16))
        feats_orders.append(rn_order.RNOrder(os.path.join(feats_dir,'train_gfc0_avg_features_fp_orig.pickle'), 'conv\navg fp\noriginal', args.normalize, how_many, args.set, 'pca', pca_dims=8))
        feats_orders.append(rn_order.RNOrder(os.path.join(feats_dir,'train_gfc0_max_features_fp_orig.pickle'), 'conv\nmax fp\noriginal', args.normalize, how_many, args.set, 'pca', pca_dims=8))

        feats_orders.append(rn_order.RNOrder(os.path.join(feats_dir,'train_gfc2_avg_features_fp.pickle'), 'g_fc2\navg fp', args.normalize, how_many, args.set, 'pca', pca_dims=8))
        feats_orders.append(rn_order.RNOrder(os.path.join(feats_dir,'train_gfc2_max_features_fp.pickle'), 'g_fc2\nmax fp', args.normalize, how_many, args.set, 'pca', pca_dims=8))
        feats_orders.append(rn_order.RNOrder(os.path.join(feats_dir,'train_gfc1_avg_features_fp.pickle'), 'g_fc1\navg fp', args.normalize, how_many, args.set, 'pca', pca_dims=8))
        feats_orders.append(rn_order.RNOrder(os.path.join(feats_dir,'train_gfc1_max_features_fp.pickle'), 'g_fc1\nmax fp', args.normalize, how_many, args.set, 'pca', pca_dims=8))
        feats_orders.append(rn_order.RNOrder(os.path.join(feats_dir,'train_gfc0_avg_features_fp_orig.pickle'), 'conv\navg fp\noriginal', args.normalize, how_many, args.set, 'pca', pca_dims=4))
        feats_orders.append(rn_order.RNOrder(os.path.join(feats_dir,'train_gfc0_max_features_fp_orig.pickle'), 'conv\nmax fp\noriginal', args.normalize, how_many, args.set, 'pca', pca_dims=4))'''

    # RMAC #
    feats_orders.append(rmac_order.RMACOrder(os.path.join(feats_dir,'clevr_rmac_features.h5'),
        os.path.join(feats_dir,'clevr_rmac_features_order.txt'), args.normalize, how_many, args.set))

    # RMAC WITH PCA #
    '''feats_orders.append(rmac_order.RMACOrder(os.path.join(feats_dir,'clevr_rmac_features.h5'),
        os.path.join(feats_dir,'clevr_rmac_features_order.txt'), args.normalize, how_many, args.set, 'pca'))

    feats_orders.append(rmac_order.RMACOrder(os.path.join(feats_dir,'clevr_rmac_features.h5'),
        os.path.join(feats_dir,'clevr_rmac_features_order.txt'), args.normalize, how_many, args.set, 'pca', pca_dims=512))

    feats_orders.append(rmac_order.RMACOrder(os.path.join(feats_dir,'clevr_rmac_features.h5'),
        os.path.join(feats_dir,'clevr_rmac_features_order.txt'), args.normalize, how_many, args.set, 'pca', pca_dims=256))'''

    # Make sure that there are not duplicate identifiers, otherwise strange things happen
    all_ids = [x.get_identifier() for x in feats_orders]
    assert len(set(all_ids))==len(all_ids), "Duplicate identifiers!"
    
    #initialize ground truth
    print('Initializing ground truth...')
    gt_orders = {}
    #gt_orders['graph'] = graphs_order.GraphsOrder(os.path.join(args.clevr_dir,'scenes','CLEVR_val_scenes.json'), args.graph_ground_truth, args.cpus)
    gt_orders['graph-approx'] = graphs_approx_order.GraphsApproxOrder(args.clevr_dir, args.graph_ground_truth, how_many, args.set, args.cpus)
    #gt_orders['states'] = states_order.StatesOrder(scene_json_filename, mode='fuzzy', ncpu=args.cpus)
    
    found_gt = False
    for k in list(gt_orders.keys()):
        if args.ground_truth == k:
            actual_gt_order = gt_orders[k]
            del gt_orders[k]
            found_gt = True
    assert found_gt, 'Unknown ground truth!'

    #add remaining ground truths as features
    feats_orders = feats_orders + list(gt_orders.values())

    for idx in range(args.query_img_index, args.until_img_index+1):
        #if some cached distance is missing, possibly ignore it
        dist_files_exist = [os.path.isfile(os.path.join(cache_dir,d,'d_{}.npy'.format(idx))) for d in os.listdir(cache_dir)]
        if args.skip_missing and not all(dist_files_exist):
            continue

        stat_indexes = compute_ranks(feats_orders, actual_gt_order, idx)
        if args.verbose:
            print_stats(stat_indexes,actual_gt_order.get_name(),idx)

        stats = list(stat_indexes[0].keys())
        #take only the statistical indexes and leave the label apart
        stats = [s for s in stats if 'label' not in s]
        for stat in stats:
            if stat not in stats_out:
                stats_out[stat] = []
            row_to_write = {e['label']: e[stat] for e in stat_indexes}
            #add the number of items of the current query image                
            #row_to_write['n_items'] = n_items
            row_to_write['image_id'] = idx
            stats_out[stat].append(row_to_write)
    
    #dump stats on file
    if args.normalize:
        normalized_str = 'normalized'
    else:
        normalized_str = 'no-normalized'
    gt = '{}-{}'.format(args.ground_truth, args.graph_ground_truth) if 'graph' in args.ground_truth else args.ground_truth 
    filename = os.path.join(stats_dir,'stats_{}_{}-gt_{}.pickle'.format(normalized_str, gt,args.set))
    outf = open(filename, 'wb')
    pickle.dump(stats_out, outf)
