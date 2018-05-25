import matplotlib
matplotlib.use('Agg')
import networkx as nx
import pickle
import numpy as np
import pdb
import json
import os
import cv2
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from order import graphs_order
from order import utils
import random

class ClevrImageLoader():
    def __init__(self, images_dir):
        self.images_dir = images_dir

    def get(self,index):
        padded_index = str(index).rjust(6,'0')
        img_filename = os.path.join(self.images_dir, 'val', 'CLEVR_val_{}.png'.format(padded_index))
        image = cv2.imread(img_filename)
        return image / 255.

def load_graphs(scene_file):
    clevr_scenes = json.load(open(scene_file))['scenes']
    graphs = []

    for scene in clevr_scenes:
        graph = nx.MultiDiGraph()
        #build graph nodes for every object
        objs = scene['objects']
        idx_pool = list(range(len(objs)))
        random.shuffle(idx_pool)
        for idx, obj in enumerate(objs):
            graph.add_node(idx_pool[idx], color=obj['color'], shape=obj['shape'], material=obj['material'], size=obj['size'])
        
        relationships = scene['relationships']
        for name, rel in relationships.items():
            if name in ('right','front'):
                for b_idx, row in enumerate(rel):
                    for a_idx in row:
                        graph.add_edge(a_idx, b_idx, relation=name)

        graphs.append(graph)
    return graphs

'''
Calculates graph edit distance.
If node_weight_mode == 'proportional', node substitution weights 1/n for every of the attributes not matching (n is total number of attributes)
If node_weight_mode == 'atleastone', node substitution weights 1 is even only one attribute does not match
'''
def ged_paths(g1,g2,node_weight_mode='proportional'):
    #need to incorporate edges attributes in order for ged edge costs to work correctly
    for e, attr in g1.edges.items():
    #pdb.set_trace()
        attr['nodes'] = '{}-{}'.format(e[0],e[1])
    for e, attr in g2.edges.items():
        attr['nodes'] = '{}-{}'.format(e[0],e[1])

    def edge_subst_cost(gattr, hattr):
        if (gattr['relation'] == hattr['relation']): # and (gattr['nodes'] == hattr['nodes'])):
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

    return nx.optimal_edit_paths(g1, g2,
         edge_subst_cost = edge_subst_cost,
         node_subst_cost = node_subst_cost)

if __name__ == '__main__':
    clevr_scene = '../../../CLEVR_v1.0/scenes/CLEVR_val_scenes.json'
    clevr_image = '../../../CLEVR_v1.0/images'
    query_img = 6
    mode = 'proportional'

    order = graphs_order.GraphsOrder(clevr_scene, mode, 4)
    _, ordered_dist, permuts = list(utils.build_feat_dict([order], query_img, include_query=True).values())[0]
    print('First 10 distances: {}'.format(ordered_dist[:10]))
    print('First 10 permuts:   {}'.format(permuts[:10]))
    idx1 = permuts[0]
    idx2 = permuts[4]
    print('Idxs: {} and {}'.format(idx1, idx2))

    graphs = graphs_order.GraphsOrder.graphs
    #graphs = load_graphs(clevr_scene)
    '''graphs = []
    g1 = nx.MultiDiGraph()
    eidx1=21
    eidx2=7
    eidx3=9
    g1.add_node(eidx1, color='green', shape='cube', material='rubber', size='big')
    g1.add_node(eidx2, color='yellow', shape='cube', material='metal', size='big')
    g1.add_node(eidx3, color='gray', shape='sphere', material='rubber', size='small')
    g1.add_edge(eidx1, eidx2, relation='left')
    g1.add_edge(eidx1, eidx3, relation='left')
    g1.add_edge(eidx3, eidx2, relation='left')
    g1.add_edge(eidx1, eidx2, relation='front')
    g1.add_edge(eidx3, eidx2, relation='front')
    g1.add_edge(eidx3, eidx1, relation='front')
    graphs.append(g1)

    g2 = nx.MultiDiGraph()
    eidx1=44
    eidx2=45
    eidx3=46
    g2.add_node(eidx1, color='green', shape='cube', material='rubber', size='big')
    g2.add_node(eidx2, color='yellow', shape='cube', material='metal', size='big')
    g2.add_node(eidx3, color='gray', shape='sphere', material='rubber', size='small')
    g2.add_edge(eidx1, eidx2, relation='left')
    g2.add_edge(eidx1, eidx3, relation='left')
    g2.add_edge(eidx3, eidx2, relation='left')
    g2.add_edge(eidx1, eidx2, relation='front')
    g2.add_edge(eidx3, eidx2, relation='front')
    g2.add_edge(eidx3, eidx1, relation='front')
    graphs.append(g2)

    idx1 = 0
    idx2 = 1'''

    for _idx2 in permuts[:10]:
        print(ged_paths(graphs[idx1], graphs[_idx2], mode))
    #print(ged_paths(graphs[0], graphs[1], mode))

    #visualize the two images
    fig = plt.figure('Images Comparison', figsize=(10,10))
    gs = gridspec.GridSpec(2, 2)

    image_loader = ClevrImageLoader(clevr_image)
    
    img1_axs = plt.subplot(gs[0, 0])
    img1_axs.set_title('img1')
    img1_axs.imshow(image_loader.get(idx1))

    img1_axs = plt.subplot(gs[0, 1])
    img1_axs.set_title('img2')
    img1_axs.imshow(image_loader.get(idx2))
    
    graph1_axs = plt.subplot(gs[1, 0])
    nx.draw_networkx(graphs[idx1], ax=graph1_axs, font_size=8, labels=dict((n,'{}\n{}\n{}\n{}\n{}'.format(n,d['size'],d['color'],d['material'],d['shape'])) for n,d in graphs[idx1].nodes(data=True)))

    graph2_axs = plt.subplot(gs[1, 1])
    nx.draw_networkx(graphs[idx2], ax=graph2_axs, font_size=8, labels=dict((n,'{}\n{}\n{}\n{}\n{}'.format(n,d['size'],d['color'],d['material'],d['shape'])) for n,d in graphs[idx2].nodes(data=True)))

    plt.savefig('imgs.png')
    

    
    
    
