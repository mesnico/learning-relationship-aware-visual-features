import pickle
import numpy as np
import pdb
import networkx as nx
import multiprocessing
import time
import json
from .order_base import OrderBase

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

def ged_parallel_worker(query_img_index, idx, node_weight_mode):
    if ged_parallel_worker.distances[idx] < 0:
        query_img_graphs = ged_parallel_worker.graphs[query_img_index]
        g = ged_parallel_worker.graphs[idx]
        start = time.time()
        dist = ged(query_img_graphs,g, node_weight_mode)
        end = time.time()
        print('## Query idx: {}, mode {} ## - #edges: {}; sample#{}/{} ({} s)'.format(query_img_index, node_weight_mode, len(g.edges), idx, len(ged_parallel_worker.graphs), end-start))
        ged_parallel_worker.distances[idx] = dist
    #else:
        #print('## Query idx: {}, mode {} ## ------------- sample#{}/{} SKIP'.format(query_img_index, node_weight_mode, idx, len(ged_parallel_worker.graphs)))        

'''
used to inizialize workers context with the queue
'''
def ged_parallel_worker_init(distances, graphs):
    ged_parallel_worker.distances = distances
    ged_parallel_worker.graphs = graphs


'''Ordering for distances extracted by graphs by means of graph edit distance'''
class GraphsOrder(OrderBase):
    def __init__(self, scene_file, gt='proportional', ncpu=4):
        super().__init__()
        print('Building graphs from JSON...')
        self.graphs = self.load_graphs(scene_file)
        self.gt = gt
        self.ncpu = ncpu

    def load_graphs(self,scene_file):
        clevr_scenes = json.load(open(scene_file))['scenes']
        graphs = []

        for scene in clevr_scenes:
            graph = nx.MultiDiGraph()
            #build graph nodes for every object
            objs = scene['objects']
            for idx, obj in enumerate(objs):
                graph.add_node(idx, color=obj['color'], shape=obj['shape'], material=obj['material'], size=obj['size'])
            
            relationships = scene['relationships']
            for name, rel in relationships.items():
                if name in ('right','front'):
                    for b_idx, row in enumerate(rel):
                        for a_idx in row:
                            graph.add_edge(a_idx, b_idx, relation=name)

            graphs.append(graph)
        return graphs

    def compute_distances(self, query_img_index):
        query_img_graphs = self.graphs[query_img_index]
        filename = os.path.join('./cache','graph_distances_queryidx{}_{}.npy'.format(query_img_index,self.gt))
        n_graphs = len(self.graphs)
        if os.path.isfile(filename):
            print('Graph distances file existing for image {}, mode {}! Loading...'.format(query_img_index,self.gt))
            distances = np.memmap(filename, dtype=np.float32, shape=(n_graphs,), mode='r+')
        else:
            distances = np.memmap(filename, dtype=np.float32, shape=(n_graphs,), mode='w+')
            distances[:] = -1
            
        print('Computing {} graph distances for image {}, mode {};...'.format(len(self.graphs),query_img_index,self.gt))
        
        with multiprocessing.Pool(processes=self.ncpu, initializer=ged_parallel_worker_init, initargs=(distances,self.graphs)) as pool:
            for idx in range(n_graphs):
                pool.apply_async(ged_parallel_worker, args=(query_img_index, idx, self.gt))
            
            pool.close()
            pool.join()

        distances.flush()
        return distances

    def get_name(self):
        return 'graph GT ({})'.format(self.gt)
    def length(self):
        return len(self.graphs)


#simple test
import os
if __name__ == "__main__":
    clevr_dir = '../../../../CLEVR_v1.0'
    idx = 6
    
    scene_json_filename = os.path.join(clevr_dir, 'scenes', 'CLEVR_val_scenes.json')
    s = GraphsOrder(scene_json_filename)
    print(s.get(idx))
