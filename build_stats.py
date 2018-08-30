import pickle
import os
import numpy as np
import pdb
import networkx as nx
from scipy.stats import spearmanr, kendalltau
import math
import time
from order import rn_order, rmac_order, graphs_approx_order
from order import utils
import argparse
from tqdm import tqdm
#import networkx.algorithms.similarity

def compute_ranks(feat_orders, gt_order, query_img_index, include_query=False):
    stat_indexes = []
    max_feats_len, min_feats_len = utils.max_min_length(feat_orders + [gt_order])

    feat_distances = utils.build_feat_dict(feat_orders, query_img_index, min_length=min_feats_len, include_query=include_query)
    gt_distance = utils.build_feat_dict([gt_order], query_img_index, min_length=min_feats_len, include_query=include_query)

    assert len(gt_distance) == 1, 'More than one Ground-Truth!'
    dist_gt, _, perm_gt = list(gt_distance.values())[0]

    for name, (dist, _, permut) in feat_distances.items():
        #calculate stats for every feature
        
        stat_indexes.append({'label': name,
                    'spearmanr': spearmanr(dist, dist_gt)[0]
                    })

    return stat_indexes

def print_stats(stats, gt, idx):
    for stat in stats:
        print('## Query idx: {} ## - Correlation among {} and actual GT: {}\n\tspearman-rho: {}'.format(
            idx,
            stat['label'],
            gt,
            stat['spearmanr']
            ))
        
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
    parser.add_argument('--ground-truth', type=str, choices=['graph','graph-approx','states'], default='graph-approx',
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

    if args.query_img_index > args.until_img_index:
        raise ValueError('Start query index should be less than end index')
        

    rn_feats_dir = os.path.join('RelationNetworks-CLEVR','features')
    rmac_feats_dir = 'rmac_features'

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

    for f in os.listdir(rn_feats_dir):
        nf = f.replace('_','\n').split('.')[0]
        feats_orders.append(rn_order.RNOrder(os.path.join(rn_feats_dir,f), nf, args.normalize))

    # RMAC #
    feats_orders.append(rmac_order.RMACOrder(os.path.join(rmac_feats_dir,'clevr_rmac_features.h5'),
        os.path.join(rmac_feats_dir,'clevr_rmac_features_order.txt'), args.normalize, how_many, args.set))

    # Make sure that there are not duplicate identifiers, otherwise strange things happen
    all_ids = [x.get_identifier() for x in feats_orders]
    assert len(set(all_ids))==len(all_ids), "Duplicate identifiers!"
    
    #initialize ground truth
    print('Initializing ground truth...')
    gt_orders = {}
    #gt_orders['graph'] = graphs_order.GraphsOrder(scene_json_filename, args.graph_ground_truth, args.cpus)
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

    for idx in tqdm(range(args.query_img_index, args.until_img_index+1)):
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
