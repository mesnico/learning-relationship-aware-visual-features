import numpy as np

### MATH UTILS ###
def dot_dist(a,b):
    a_norm = normalized(a)
    b_norm = normalized(b)
    return 1-np.dot(a_norm, b_norm)
    
def l2_dist(a,b):
    return np.linalg.norm(a-b)

def normalized(a, axis=-1, order=2):
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2==0] = 1
    return np.squeeze(a / np.expand_dims(l2, axis))

### ORDERING UTILS ###
def max_min_length(orders_iterable):
    lengths = list(map(lambda o: o.length(), orders_iterable))
    return (max(lengths), min(lengths))

def build_feat_dict(orders_iterable, query_idx, min_length=0, include_query=False):
    orderings = [o.get(query_idx, include_query) for o in orders_iterable]
    names = [o.get_name() for o in orders_iterable]

    start_idx = 0 #if include_query else 1
    if min_length > 0:
        #cut in order to have the same number of elements for all the orderings
        orderings = [tuple(o[i][start_idx:min_length] for i in np.arange(3)) for o in orderings]
    else:
        orderings = [tuple(o[i][start_idx:] for i in range(3)) for o in orderings]
        
    d = {n:o for n,o in zip(names, orderings)}

    return d
