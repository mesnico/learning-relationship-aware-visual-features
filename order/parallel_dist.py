import multiprocessing
import os
import time
import numpy as np
from progressbar import ProgressBar, Percentage, Bar, AdaptiveETA
import threading


def parallel_worker(query_img_index, idx, worker_fn, kwargs):
    if parallel_worker.distances[idx] < 0:
        query = parallel_worker.elems[query_img_index]
        g = parallel_worker.elems[idx]
        #start = time.time()
        try:
            dist = worker_fn(query,g,**kwargs)
        except Exception as e:
            print(str(e))
        #end = time.time()
        #print('## Query idx: {} ## - sample#{}/{} ({} s)'.format(query_img_index, idx, len(parallel_worker.elems), end-start))
        parallel_worker.distances[idx] = dist
    #else:
        #print('## Query idx: {} ## ------------- sample#{}/{} SKIP'.format(query_img_index, idx, len(parallel_worker.elems)))        

'''
used to inizialize workers context with the queue
'''
def parallel_worker_init(distances, elems):
    parallel_worker.distances = distances
    parallel_worker.elems = elems

def parallel_distances(cache_name, elems, query_img_index, worker_fn, ncpu=4, kwargs={}):
    lock = threading.Lock()
    end_works = 0
    def increment_finished(x):
        nonlocal end_works
        with lock:
            end_works+=1
        
    cache_dir = os.path.join('./dist_cache',cache_name)
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    filename = os.path.join(cache_dir,'d_{}.npy'.format(query_img_index))
    n = len(elems)
    if os.path.isfile(filename):
        print('Graph distances file existing for image {}, cache {}! Loading...'.format(query_img_index, cache_name))
        distances = np.memmap(filename, dtype=np.float32, shape=(n,), mode='r+')
    else:
        distances = np.memmap(filename, dtype=np.float32, shape=(n,), mode='w+')
        distances[:] = -1
        
    print('Computing {} distances for image {}, cache {}...'.format(n,query_img_index,cache_name))
    
    pbar = ProgressBar(widgets=[Percentage(), Bar(), AdaptiveETA()], maxval=n).start()
    with multiprocessing.Pool(processes=ncpu, initializer=parallel_worker_init, initargs=(distances,elems)) as pool:
        for idx in range(n):
            pool.apply_async(parallel_worker, args=(query_img_index, idx, worker_fn, kwargs), callback=increment_finished)
        
        while end_works != n:
            pbar.update(end_works)
            time.sleep(1)

        pool.close()
        pool.join()

    distances.flush()
    return distances
