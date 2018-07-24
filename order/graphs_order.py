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

    def __init__(self, scene_file, gt='proportional', ncpu=4):
        super().__init__()
        if not GraphsOrder.graphs:
            print('Building graphs from JSON...')
            GraphsOrder.graphs = self.load_graphs(scene_file)
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

    def get_identifier(self):
        return format(self.get_name().replace('\n','_').replace(' ','-'))

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
