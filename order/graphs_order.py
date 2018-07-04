import pickle
import numpy as np
import pdb
import networkx as nx
import json
from .order_base import OrderBase
from .parallel_dist import parallel_distances

'''Ordering for distances extracted by graphs by means of graph edit distance'''
class GraphsOrder(OrderBase):
    graphs = None

    def __init__(self, gt='proportional', ncpu=4):
        super().__init__()
        if not GraphsOrder.graphs:
            print('Building graphs from JSON...')
            GraphsOrder.graphs = self.load_graphs('./')
        self.gt = gt
        self.ncpu = ncpu

    def load_graphs(self,basedir):
        print('loading only the images...')
        dirs = os.path.join(basedir,'data')
        filename = os.path.join(dirs,'sort-of-clevr.pickle')
        with open(filename, 'rb') as f:
          train_datasets, test_datasets = pickle.load(f)

        elems = []
        for elem in train_datasets:
            img = elem[0]
            img = np.swapaxes(img,0,2)

            #Append also the graph is present in the data
            if len(elem)==3:
                elems.append((img))
            else:
                elems.append((img,elem[3]))

        for elem in test_datasets:
            img = elem[0]
            img = np.swapaxes(img,0,2)

            #Append also the graph is present in the data
            if len(elem)==3:
                elems.append((img))
            else:
                elems.append((img,elem[3]))
        print('loaded {} images'.format(len(elems)))
        
        return [e[1] for e in elems]

    '''
    Calculates graph edit distance.
    If node_weight_mode == 'proportional', node substitution weights 1/n for every of the attributes not matching (n is total number of attributes)
    If node_weight_mode == 'atleastone', node substitution weights 1 is even only one attribute does not match
    '''
    def ged(self,g1,g2,node_weight_mode='proportional'):
        '''#need to incorporate edges attributes in order for ged edge costs to work correctly
        for e, attr in g1.edges.items():
        #pdb.set_trace()
            attr['nodes'] = '{}-{}'.format(e[0],e[1])
        for e, attr in g2.edges.items():
            attr['nodes'] = '{}-{}'.format(e[0],e[1])'''

        def edge_subst_cost(gattr, hattr):
            if (gattr['relation'] == hattr['relation']): #and (gattr['nodes'] == hattr['nodes']):
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

    def compute_distances(self, query_img_index):
        return parallel_distances('ged-{}'.format(self.gt), self.graphs, query_img_index, self.ged, kwargs={'node_weight_mode':self.gt}, ncpu=self.ncpu)

    def get_name(self):
        return 'graph GT\n({})'.format(self.gt)
    def length(self):
        return len(self.graphs)


#simple test
import os
if __name__ == "__main__":
    clevr_dir = '../../../CLEVR_v1.0'
    idx = 6
    
    scene_json_filename = os.path.join(clevr_dir, 'scenes', 'CLEVR_val_scenes.json')
    s = GraphsOrder(scene_json_filename)
    print(s.get(idx))
