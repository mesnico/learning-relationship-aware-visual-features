import pickle
import numpy as np
import pdb
import networkx as nx
from .AproximatedEditDistance import AproximatedEditDistance
import json
from .order_base import OrderBase
from .parallel_dist import parallel_distances

class ApproxGED(AproximatedEditDistance):
    """
        Vanilla Aproximated Edit distance, implements basic costs for substitution insertion and deletion.
    """
    def __init__(self, node_weight_mode='proportional'):
        self.node_weight_mode = node_weight_mode

    """
        Node edit operations
    """
    def node_substitution(self, g1, g2):
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
        """
            Node substitution costs
            :param g1, g2: Graphs whose nodes are being substituted
            :return: Matrix with the substitution costs
        """
        if self.node_weight_mode == 'proportional':
            node_subst_cost = node_subst_cost_proportional
        elif self.node_weight_mode == 'atleastone':
            node_subst_cost = node_subst_cost_atleastone

        m = len(g1.nodes)
        n = len(g2.nodes)
        c_mat = np.array([node_subst_cost(g1.nodes[u], g2.nodes[v])
            for u in g1.nodes for v in g2.nodes]).reshape(m, n)
        return c_mat

    def node_insertion(self, g):
        """
            Node Insertion costs
            :param g: Graphs whose nodes are being inserted
            :return: List with the insertion costs
        """
        values = [v for k, v in g.nodes(data=True)]
        return [1]*len(values)

    def node_deletion(self, g):
        """
            Node Deletion costs
            :param g: Graphs whose nodes are being deleted
            :return: List with the deletion costs
        """
        values = [v for k, v in g.nodes(data=True)]
        return [1] * len(values)

    """
        Edge edit operations
    """
    def edge_substitution(self, g1, g2):
        """
            Edge Substitution costs
            :param g1, g2: Adjacency list for particular nodes.
            :return: List of edge deletion costs
        """
        edge_dist = np.zeros((len(g1), len(g2)))
        for i_idx,i in enumerate(g1):
            for j_idx,j in enumerate(g2):   
                rel_to_i = set([k['relation'] for k in g1[i].values()])
                rel_to_j = set([k['relation'] for k in g2[j].values()])
        
                # calculate the amount of relations that differ from node i to node j
                diff = len(rel_to_i.union(rel_to_j))-len(rel_to_i.intersection(rel_to_j))
                edge_dist[i_idx,j_idx] = diff
                                      
        return edge_dist

    def edge_insertion(self, g):
        """
            Edge insertion costs
            :param g: Adjacency list.
            :return: List of edge insertion costs
        """
        insert_edges = [len(e) for e in g]
        return np.ones(len(insert_edges))

    def edge_deletion(self, g):
        """
            Edge Deletion costs
            :param g: Adjacency list.
            :return: List of edge deletion costs
        """
        del_edges = [len(e) for e in g]
        return np.ones(len(del_edges))

'''Ordering for distances extracted by graphs by means of approximated graph edit distance'''
class GraphsApproxOrder(OrderBase):
    graphs = None

    def __init__(self, clevr_dir, gt='proportional', how_many=15000, st='test', ncpu=4):
        super().__init__()

        s = 'val' if st=='test' else st
        scene_file = os.path.join(clevr_dir, 'scenes', 'CLEVR_{}_scenes.json'.format(s))
        if not GraphsApproxOrder.graphs:
            print('Building graphs from JSON from {}...'.format(s))
            GraphsApproxOrder.graphs = self.load_graphs(scene_file, how_many)
        self.gt = gt
        self.st = st
        self.ncpu = ncpu

    '''def load_graphs(self,scene_file, how_many):
        clevr_scenes = json.load(open(scene_file))['scenes']
        clevr_scenes = clevr_scenes[:how_many]
        graphs = []

        for scene in clevr_scenes:
            graph = {}
            graph['right'] = nx.DiGraph()
            graph['front'] = nx.DiGraph()
            #build graph nodes for every object
            objs = scene['objects']
            for idx, obj in enumerate(objs):
                for _,g in graph.items():
                    g.add_node(idx, color=obj['color'], shape=obj['shape'], material=obj['material'], size=obj['size'])
            
            relationships = scene['relationships']
            for name, rel in relationships.items():
                if name in ('right','front'):
                    for b_idx, row in enumerate(rel):
                        for a_idx in row:
                            graph[name].add_edge(a_idx, b_idx)

            graphs.append(graph)
        return graphs'''

    def load_graphs(self,scene_file,how_many):
        clevr_scenes = json.load(open(scene_file))['scenes']
        clevr_scenes = clevr_scenes[:how_many]
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
    Calculates approximated graph edit distance.
    '''
    def ged(self,g1,g2,node_weight_mode='proportional'):
        tot_cost = 0
        approx_ged = ApproxGED(self.gt)
        '''for rel in ['right','front']:
            c, _ = approx_ged.ged(g1[rel], g2[rel])
            tot_cost += c
        '''
        tot_cost, _ = approx_ged.ged(g1, g2)
        return tot_cost

    def compute_distances(self, query_img_index):
        return parallel_distances('ged-approx-{}-{}'.format(self.gt, self.st), self.graphs, query_img_index, self.ged, kwargs={'node_weight_mode':self.gt}, ncpu=self.ncpu)
        #query_graph = self.graphs[query_img_index]
        #return [self.ged(query_graph, g) for g in self.graphs]

    def get_name(self):
        return 'graph GT\n({})\napprox'.format(self.gt)

    def get_identifier(self):
        return '{}-set{}'.format(self.get_name().replace('\n','_').replace(' ','-'), self.st)

    def length(self):
        return len(self.graphs)


#simple test
import os
if __name__ == "__main__":
    clevr_dir = '/mnt1/CLEVR_v1.0/'
    idx = 6
    
    #scene_json_filename = os.path.join(clevr_dir, 'scenes', 'CLEVR_val_scenes.json')
    s = GraphsApproxOrder(clevr_dir, how_many=13000, st='train')
    print(s.ged(GraphsApproxOrder.graphs[44],GraphsApproxOrder.graphs[7674]))
