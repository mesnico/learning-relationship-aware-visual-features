import pickle
import os
import numpy as np
import pdb
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import networkx as nx
from scipy.stats import spearmanr 
import multiprocessing
import csv
import math
import time
#import networkx.algorithms.similarity

def spearman_rho_distance(perm1, perm2):
    perms = np.stack((perm1, perm2), axis=1)
    perms = perms.T
    inv_perms = np.empty(perms.shape)
    for i in range(2):
        for idx, val in enumerate(perms[i]):     
            inv_perms[i][val] = idx
    diff = np.diff(inv_perms, axis=0)
    sum = np.sum(pow(diff,2))
    return sum

def recall_at(gt_list, c_list, k=10):
    if len(gt_list) != len(c_list):
        raise ValueError('Dimension mismatch: the two lists should have same length')
    if k > len(gt_list):
        k = len(gt_list)
    gt_set = set(gt_list[0:k])
    c_set = set(c_list[0:k])
    
    diff = gt_set.intersection(c_set)
    return len(diff)/k

def build_figure(title, images, numcols, query_img):
    num_images = len(images)
    numrows = num_images // numcols

    plt.figure(title)
    gs = gridspec.GridSpec(numrows+1, numcols)
    #fig, axs = plt.subplots(numrows, numcols, sharex=True, sharey=True)
    query_axs = plt.subplot(gs[0, numcols//2])
    #query_swim = np.swapaxes(query_img,0,2)
    query_axs.set_title('Query Image')
    query_axs.imshow(query_img)
    i = 0
    for image in images:
        #swim = np.swapaxes(image,0,2)
        axs = plt.subplot(gs[(i//numcols)+1, i%numcols])
        axs.set_title('#' + str(i+1))
        axs.set_yticklabels([])
        axs.set_xticklabels([])
        axs.imshow(image)
        i = i+1

'''
Calculates graph edit distance.
If node_weight_mode == 'proportional', node substitution weights 1/n for every of the attributes not matching (n is total number of attributes)
If node_weight_mode == 'atleastone', node substitution weights 1 is even only one attribute does not match
'''
def ged(g1,g2,node_weight_mode='proportional'):
    #need to incorporate edges attributes in order for ged edge costs to work correctly
    for e, attr in g1.edges.items():
    #pdb.set_trace()
        attr['nodes'] = '{}-{}'.format(e[0],e[1])
    for e, attr in g2.edges.items():
        attr['nodes'] = '{}-{}'.format(e[0],e[1])

    def edge_subst_cost(gattr, hattr):
        if (gattr['relation'] == hattr['relation']) and (gattr['nodes'] == hattr['nodes']):
            return 0
        else:
            return 1

    def node_subst_cost_proportional(uattr, vattr):
        cost = 0
        attributes = list(uattr.keys())
        unit_cost = 1/len(attributes)
        for attr in attributes:
            if(uattr[attr] != vattr[attr]): cost=cost+unit_cost
        return cost

    def node_subst_cost_atleastone(uattr, vattr):
        cost = 0
        attributes = list(uattr.keys())
        for attr in attributes:
            if(uattr[attr] != vattr[attr]): 
                cost=1
                break
        return cost

    if node_weight_mode == 'proportional':
        node_subst_cost = node_subst_cost_proportional
    elif node_weight_mode == 'atleastone':
        node_subst_cost = node_subst_cost_atleastone
    else:
        raise ValueError('Node weight mode {} not known'.format(node_weight_mode))

    return nx.graph_edit_distance(g1, g2,
         edge_subst_cost = edge_subst_cost,
         node_subst_cost = node_subst_cost)
    
def load_features(filename):
    features = []

    f = open(filename, 'rb')
    while 1:
        try:
            this_feat = pickle.load(f)
            features.append(this_feat[1])
            bs = this_feat[1].shape[0]
            #print('batch #{} (size: {}) loaded'.format(this_feat[0], bs))
        except EOFError:
            break
    #pdb.set_trace()
    #print('features loaded from {}'.format(filename))
    return features

def ged_parallel_worker(query_img_index, idx, node_weight_mode):
    if ged_parallel_worker.distances[idx] < 0:
        query_img_graphs = ged_parallel_worker.graphs[query_img_index]
        g = ged_parallel_worker.graphs[idx]
        start = time.time()
        dist = ged(query_img_graphs,g, node_weight_mode)
        end = time.time()
        print('## Query idx: {}, mode {} ## - #edges: {}; sample#{}/{} ({} s)'.format(query_img_index, node_weight_mode, len(g.edges), idx, len(ged_parallel_worker.graphs), end-start))
        ged_parallel_worker.distances[idx] = dist
    else:
        print('## Query idx: {}, mode {} ## ------------- sample#{}/{} SKIP'.format(query_img_index, node_weight_mode, idx, len(ged_parallel_worker.graphs)))        

'''
used to inizialize workers context with the queue
'''
def ged_parallel_worker_init(distances, graphs):
    ged_parallel_worker.distances = distances
    ged_parallel_worker.graphs = graphs

def cache_ged_distances(graphs, query_img_index, node_weight_mode,cpus):
    query_img_graphs = graphs[query_img_index]
    filename = os.path.join('./cache','graph_distances_queryidx{}_{}.npy'.format(query_img_index,node_weight_mode))
    n_graphs = len(graphs)
    if os.path.isfile(filename):
        print('Graph distances file existing for image {}, mode {}! Loading...'.format(query_img_index,node_weight_mode))
        distances = np.memmap(filename, dtype=np.float32, shape=(n_graphs,), mode='r+')
    else:
        distances = np.memmap(filename, dtype=np.float32, shape=(n_graphs,), mode='w+')
        distances[:] = -1
        
    print('Computing {} graph distances for image {}, mode {};...'.format(len(graphs),query_img_index,node_weight_mode))
    
    with multiprocessing.Pool(processes=cpus, initializer=ged_parallel_worker_init, initargs=(distances,graphs)) as pool:
        for idx in range(n_graphs):
            pool.apply_async(ged_parallel_worker, args=(query_img_index, idx, node_weight_mode))
        
        pool.close()
        pool.join()
    
    distances.flush()
    return distances

def compute_ranks(features, graphs, query_img_index, node_weight_mode, cpus):
    stat_indexes = []

    if len(features) != 0:
        max_feats_len = max(map(len, features.values()))

        #prepare the axis for computing log-scale recall-at-k.
        log_base = 1.3
        logscale_ub = np.floor(math.log(max_feats_len,log_base))
        k_logscale_axis = np.floor(np.power(log_base,np.array(range(int(logscale_ub)))))
        k_logscale_axis = np.unique(k_logscale_axis)
        k_logscale_axis = k_logscale_axis.astype(int)

    ''' GRAPH DISTANCE ORDERING (GROUND TRUTH)'''
    query_img_graphs = graphs[query_img_index]
    #cut to the same number of features  
    distances_graphs = cache_ged_distances(graphs, query_img_index, node_weight_mode, cpus)
    if len(features) != 0:
        distances_graphs = distances_graphs[0:max_feats_len]  

    dist_permutations_graphs = np.argsort(distances_graphs)

    ''' FEATURES ORDERING '''
    dist_permutations_feats = []
    for name, feat in features.items():
        query_img_feat = feat[query_img_index]
        dists = [np.linalg.norm(query_img_feat-i) for i in feat]
        #cut so that all feats have the same length
        dists = dists[0:max_feats_len]
        permuts = np.argsort(dists)
        dist_permutations_feats.append({'name':name,'permuts':permuts})

        #calculate stats for every conv feature
        k_logscale = {k:recall_at(permuts, dist_permutations_graphs,k) for k in k_logscale_axis}

        stat_indexes.append({'label': name,
                    'spearmanr': spearmanr(dists, distances_graphs)[0],
                    'recall-at-10': recall_at(permuts, dist_permutations_graphs,10),
                    'recall-at-100': recall_at(permuts, dist_permutations_graphs,100),
                    'recall-at-1000': recall_at(permuts, dist_permutations_graphs,1000),
                    'recall-at-k': dict(k_logscale)})

    number_of_items = len(query_img_graphs.nodes)
    return {'ranks':(dist_permutations_feats, dist_permutations_graphs), 
        'stat-indexes': stat_indexes, 'items': number_of_items}

def print_stats(stats, idx):
    for stat in stats:
        print('## Query idx: {} ## - Correlation among {} and graph:\n\tspearman-rho: {}\n\trecall-at-10: {}\n\trecall-at-100: {}\n\trecall-at-1000: {}'.format(
            idx,
            stat['label'],
            stat['spearmanr'],
            stat['recall-at-10'],
            stat['recall-at-100'],
            stat['recall-at-1000']))
        
#show images only if we are threating only one image
def start(images_loader, query_img_index, graphs, n_displ_images = 10, ground_truth = 'proportional', until_img_index = None, cpus = 8, features = {}, normalize = False):
    #prepare cache folder
    cache_dir='./cache'
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

    if not until_img_index:
        results = compute_ranks(features, graphs, query_img_index, ground_truth, cpus)
        print_stats(results['stat-indexes'],query_img_index)

        query_img = images_loader.get(query_img_index)

        '''sorted_images = [images[d] for d in results['ranks'][0]]
        #take only the first n
        sorted_images = sorted_images[0:args.N]
        build_figure('With g_fc4 features distance',sorted_images, 5, query_img)'''

        #sorted_conv_images = []
        for permuts in results['ranks'][0]:
            values = permuts['permuts']
            #cut to the number of image to show
            values = values[0:n_displ_images]
            name = permuts['name']
            sorted_images = [images_loader.get(d) for d in values]
            build_figure('With {} features distance; query idx {}'.format(name, query_img_index), sorted_images, 5, query_img)
        #among all the conv features, display the result for the best one in terms of spearman-rho correlation
        #spearman_conv = {c['label']: c['spearmanr'] for c in results['stat-indexes'] if 'Conv' in c['label']}
        #max_correlation = np.amax(spearman_conv.values())
        #max_correl_featidx = np.argmax(spearman_conv.values())
        #best_feat_label = list(spearman_conv.keys())[max_correl_featidx]
        
        values = results['ranks'][1][0:n_displ_images]
        sorted_images = [images_loader.get(d) for d in values]
        build_figure('With graph edit distance ({} ground-truth); query idx {}'.format(ground_truth, query_img_index),sorted_images, 5, query_img)

        plt.show()
    else:
        '''#first of all, cache all ged distances, if necessary
        if(os.path.isfile(os.path.join(cache_dir,'graph_distances_queryidx{}_{}.pickle'.format(until_img_index,ground_truth)))):
            print('All ged distances already cached! Skipping...')
        else:
            with multiprocessing.Pool(processes=8) as pool:
                for idx in range(query_img_index, until_img_index+1):
                    pool.apply_async(cache_ged_distances, args=(graphs, idx, ground_truth))

                pool.close()
                pool.join()
        '''
        #then, calculate actual statistics
        stats_out = {}

        for idx in range(query_img_index, until_img_index+1):
            results = compute_ranks(features, graphs, idx, ground_truth, cpus)
            print_stats(results['stat-indexes'],idx)
            stats = []
            if len(features) != 0:
                stats = list(results['stat-indexes'][0].keys())
                #take only the statistical indexes and leave the label apart
                stats = [s for s in stats if 'label' not in s]
                for stat in stats:
                    if stat not in stats_out:
                        stats_out[stat] = []
                    row_to_write = {e['label']: e[stat] for e in results['stat-indexes']}
                    #add the number of items of the current query image                
                    row_to_write['n_items'] = results['items']
                    row_to_write['image_id'] = idx
                    stats_out[stat].append(row_to_write)
        
        #dump stats on file
        if normalize:
            normalized_str = 'normalized'
        else:
            normalized_str = 'no-normalized'
        filename = os.path.join(stats_dir,'stats_{}_{}-gt.pickle'.format(normalized_str, ground_truth))
        outf = open(filename, 'wb')
        pickle.dump(stats_out, outf)
   
            
