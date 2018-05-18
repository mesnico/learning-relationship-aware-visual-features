import itertools
import json
import pickle
import numpy as np
import pdb
from numpy.linalg import inv
from .order_base import OrderBase
from .parallel_dist import parallel_distances

'''Ordering for distances extracted from states'''
class StatesOrder(OrderBase):
    states = None

    def __init__(self, state_file, mode='fuzzy', ncpu=4):
        super().__init__()
        if not StatesOrder.states:
            print('Loading states from JSON...')
            StatesOrder.states = self.load_states(state_file)
        
        #attributes used as primary key for identifying an object
        self.all_permuts = None
        self.mode = mode
        self.ncpu = ncpu
        self.keying_attributes = ['color','size','shape','material']

    def load_states(self, state_file):
        cached_scenes = state_file.replace('.json', '.simil.pkl')

        '''if os.path.exists(cached_scenes):
            print('==> using cached scenes: {}'.format(cached_scenes))
            with open(cached_scenes, 'rb') as f:
                objects = pickle.load(f)
        else:'''
        objects = []
        with open(state_file, 'r') as json_file:
            scenes = json.load(json_file)['scenes']
            #print('caching all objects in all scenes...')
            for s in scenes:
                obj = s['objects']
                objects.append(obj)
            '''with open(cached_scenes, 'wb') as f:
                pickle.dump(all_scene_objs, f)'''

        return objects

    #transforms position space into an orthonormal one (clevr does not use orthonormal basis)
    #TODO: get transform matrix components from scene json
    def clevr_transform(self,vector):
        basis_transform_matrix = np.matrix([[0.6563112735748291, -0.754490315914154],
                                            [0.7544902563095093, 0.6563112735748291]])
        return np.squeeze(np.asarray(np.matmul(inv(basis_transform_matrix),vector)))

    def single_distance_fn_xor(self, scene1_dict, scene2_dict):
        classes = {
            'deltax':[True, False],
            'deltay':[True, False],
            'material1':['rubber','metal'],
            'color1':['cyan','blue','yellow','purple','red','green','gray','brown'],
            'shape1':['sphere','cube','cylinder'],
            'size1':['large','small'],
            'material2':['rubber','metal'],
            'color2':['cyan','blue','yellow','purple','red','green','gray','brown'],
            'shape2':['sphere','cube','cylinder'],
            'size2':['large','small'],
        }        
        if self.all_permuts == None:
            lists = list(classes.values())
            self.all_permuts = list(itertools.product(*lists))
        
        n = len(self.all_permuts)

        #calculate intersection between keys (permutation ids)
        intersection = set(scene1_dict.keys()).intersection(scene2_dict.keys())
        union = set(scene1_dict.keys()).union(scene2_dict.keys())
        
        #retrieve intersecting objects
        inters_objects_scene1 = [scene1_dict[k] for k in intersection]
        inters_objects_scene2 = [scene2_dict[k] for k in intersection]

        #calculate differences among pairs (we care about only x and y)
        deltapos_objs_scene1 = [self.clevr_transform(np.asarray(o1['3d_coords'][0:2]) - np.asarray(o2['3d_coords'][0:2])) for o1,o2 in inters_objects_scene1]
        deltapos_objs_scene2 = [self.clevr_transform(np.asarray(o1['3d_coords'][0:2]) - np.asarray(o2['3d_coords'][0:2])) for o1,o2 in inters_objects_scene2]
        
        sim = 0
        for dpos_obj_1, dpos_obj_2 in zip(deltapos_objs_scene1, deltapos_objs_scene2):
            signs = [(dpos_obj_1[i] > 0) == (dpos_obj_2[i] > 0) for i in range(2)]

            #distance chosen on the basis of coordinate signs match
            if signs[0] != signs[1]:          sim += 0
            elif signs[0] and signs[1]:         sim += 1.0
            elif not signs[0] and not signs[1]: sim += 0

        #jaccard distance
        dist = (n - sim)
        return dist

    #computes distance among relations (pairs) in two different scenes
    def single_distance_fn_fuzzy(self, scene1_dict, scene2_dict):

        def similarity(relations_scene1, relations_scene2):
            sim = 0
            for s1_obj1, s1_obj2 in relations_scene1:
                attr_sim_max = 0
                for s2_obj1, s2_obj2 in relations_scene2:
                    sim_sum = 0
                    #attributes similarity
                    for attr in self.keying_attributes:
                        if s1_obj1[attr] != s2_obj1[attr] and s1_obj2[attr] != s2_obj2[attr]:       sim_sum += 0
                        elif (s1_obj1[attr] == s2_obj1[attr]) != (s1_obj2[attr] == s2_obj2[attr]):  sim_sum += 1
                        elif s1_obj1[attr] == s2_obj1[attr] and s1_obj2[attr] == s2_obj2[attr]:     sim_sum += 2
                    sim_sum /= 2*len(self.keying_attributes)
                    #coordinate similarity
                    coord_sim = 0
                    dpos_obj_1 = self.clevr_transform(
                        np.asarray(s1_obj1['3d_coords'][0:2]) - np.asarray(s1_obj2['3d_coords'][0:2]))
                    dpos_obj_2 = self.clevr_transform(
                        np.asarray(s2_obj1['3d_coords'][0:2]) - np.asarray(s2_obj2['3d_coords'][0:2]))
                    signs = [(dpos_obj_1[i] > 0) == (dpos_obj_2[i] > 0) for i in range(2)]
                    if signs[0] != signs[1]:            coord_sim += 2/3
                    elif signs[0] and signs[1]:         coord_sim += 1
                    elif not signs[0] and not signs[1]: coord_sim += 1/3

                    attr_sim_max = max(attr_sim_max, sim_sum * coord_sim)
                sim += attr_sim_max
            return sim

        relations_scene1 = scene1_dict.values()
        relations_scene2 = scene2_dict.values()
        
        symm_sim = min(similarity(relations_scene1, relations_scene2), similarity(relations_scene2, relations_scene1))
        dist = 1 - symm_sim/(len(scene1_dict)+len(scene2_dict)-symm_sim)
        return dist

    #computes distance among relations (pairs) in two different scenes
    def single_distance_fn_jaccard(self, scene1_dict, scene2_dict):

        #calculate intersection between keys (permutation ids)
        intersection = set(scene1_dict.keys()).intersection(scene2_dict.keys())
        union = set(scene1_dict.keys()).union(scene2_dict.keys())
        
        #retrieve intersecting objects
        inters_objects_scene1 = [scene1_dict[k] for k in intersection]
        inters_objects_scene2 = [scene2_dict[k] for k in intersection]

        #calculate differences among pairs (we care about only x and y)
        deltapos_objs_scene1 = [self.clevr_transform(np.asarray(o1['3d_coords'][0:2]) - np.asarray(o2['3d_coords'][0:2])) for o1,o2 in inters_objects_scene1]
        deltapos_objs_scene2 = [self.clevr_transform(np.asarray(o1['3d_coords'][0:2]) - np.asarray(o2['3d_coords'][0:2])) for o1,o2 in inters_objects_scene2]
        
        sim = 0
        for dpos_obj_1, dpos_obj_2 in zip(deltapos_objs_scene1, deltapos_objs_scene2):
            signs = [(dpos_obj_1[i] > 0) == (dpos_obj_2[i] > 0) for i in range(2)]

            #distance chosen on the basis of coordinate signs match
            if signs[0] != signs[1]:          sim += 2/3
            elif signs[0] and signs[1]:         sim += 1
            elif not signs[0] and not signs[1]: sim += 1/3

        #jaccard distance
        dist = 1 - sim/(len(scene1_dict)+len(scene2_dict)-sim)
        return dist

    def compute_distances(self, query_img_index):
        return parallel_distances('states-{}'.format(self.mode), self.states, query_img_index, self._compute_distances, ncpu=self.ncpu, kwargs={'mode':self.mode})

    def _compute_distances(self, query_scene, curr_scene, mode):
        obj_dictionary = {}
        #returns unique ids for equal objects
        def get_object_id(obj_state):
            # keys for accessing obj dictionary exclude 3d_coords
            key = {k:v for k,v in obj_state.items() if k in self.keying_attributes}
            key = str(key)
            if key not in obj_dictionary:
                obj_dictionary[key] = len(obj_dictionary)+1
            return obj_dictionary[key]

        #create permutations
        #TODO: check for 'combinations' instead of permutations
        query_scene_permuts = list(itertools.permutations(query_scene, r=2))            
        curr_scene_permuts = list(itertools.permutations(curr_scene, r=2))

        #create permutations ids
        curr_scene_pairs = [(get_object_id(s[0]), get_object_id(s[1]))
            for s in curr_scene_permuts]
        query_scene_pairs = [(get_object_id(s[0]), get_object_id(s[1]))
            for s in query_scene_permuts]

        #create dictionaries from permutation ids to objects
        curr_scene_dict = {str(k):v for k,v in zip(curr_scene_pairs, curr_scene_permuts)}
        query_scene_dict = {str(k):v for k,v in zip(query_scene_pairs, query_scene_permuts)}
        
        if mode == 'fuzzy':
            d = self.single_distance_fn_fuzzy(curr_scene_dict, query_scene_dict)
        elif mode == 'xor':
            d = self.single_distance_fn_xor(curr_scene_dict, query_scene_dict)
        elif mode == 'jaccard':
            d = self.single_distance_fn_jaccard(curr_scene_dict, query_scene_dict)
        else:
            raise ValueError
        return d

    def get_name(self):
        return 'states GT'

    def length(self):
        return len(self.states)

#simple test
import os
if __name__ == "__main__":
    clevr_dir = '../../../CLEVR_v1.0'
    idx = 6
    
    scene_json_filename = os.path.join(clevr_dir, 'scenes', 'CLEVR_val_scenes.json')
    s = StatesOrder(scene_json_filename)
    print(s.get(idx))
