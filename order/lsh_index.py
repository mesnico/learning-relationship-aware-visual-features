import time
import faiss
import os
from datetime import timedelta

def build_lsh(xb, name, n_bits=None):

    cache_dir = 'LSH_index_cache'
    try:
        os.makedirs(cache_dir)
    except:
        print('{} already existing'.format(cache_dir))

    dim = xb.shape[1]

    if n_bits is None:
        n_bits = dim

    index_cache_fname = os.path.join(cache_dir,'index_{}_{}bits.idx'.format(name, n_bits))
    
    start_time = time.time()
    if os.path.isfile(index_cache_fname):
        print('Loading existing index...')
        cpuindex = faiss.read_index(index_cache_fname)

        res = faiss.StandardGpuResources() # use a single GPU
        index = faiss.index_cpu_to_gpu(res, 0, cpuindex)
    else:
            
        # rotate_data=True and train_thresholds=True
        index = faiss.IndexLSH(dim, n_bits, True, True)

        print('Training index with: {} ...'.format(xb.shape))
        index.train(xb)
        print('Adding ...')
        index.add(xb)           
        
        writable_index = faiss.index_gpu_to_cpu(index)
        faiss.write_index(writable_index, index_cache_fname)
  
    end_time = time.time()     
    print('Done in: {}'.format(str(timedelta(seconds=(end_time - start_time)))))
    
    return index
