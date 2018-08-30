# Learning Relationship-aware Visual Features
This repository contains the code for reproducing results from our paper: [link]

## Get ready
1.  Download and extract CLEVR_v1.0 dataset: http://cs.stanford.edu/people/jcjohns/clevr/

2. Download this repository and all submodules with 
    ```
    git clone --recursive https://github.com/mesnico/ImageRetrieval-CLEVR
    ```
    This will also download [RelationNetworks-CLEVR](https://github.com/mesnico/RelationNetworks-CLEVR) repository as a submodule in the cloned directory.
    
3. Move into the cloned repository and run 
    ```
    ./setup.sh path/to/CLEVR_v1.0
    ```
    substituting ```path/to/CLEVR_v1.0``` with the path to your CLEVR extracted folder. This script will download RMAC features for CLEVR dataset and precalculated GED distances (ground-truth). Then, it will extract features from 2S-RN using pretrained IR model.
 
    
## Results
### Spearman-Rho correlation
This section is aimed at reproducing Spearman-Rho correlation values for RMAC, RN and 2S-RN features against the generated GT.
```
./compute_results.sh -d path/to/CLEVR_v1.0
```
This script will setup a virtual environment for computing all RMAC, RN and 2S-RN distances for both soft and hard matches.
**NOTE**:The first time this script is run may take some time; once finished, results will be cached and final spearman-rho metrics will be immediately available at every successive run.

This script prints spearman-rho correlation values in the current terminal and creates a graphical visualization storing it in pdf files in the ```output``` folder.

In order to modify parameters such as *start* and *end* query indexes or number of processes used to compute GED distances, run 
```
./compute_results.sh -h
```

### Visual Feedback
It is possible to view the top relevant images using RMAC, RN and 2S-RN features, for a bunch of query images.
```
./compute_visual_results.sh -d path/to/CLEVR_v1.0
```
This script will create a pdf file in the main folder called ```visual_results.pdf``` showing retrieval results for every query image.
By default, only 10 query images are used. To change the range of query images to use you can specify parameters ```-s``` and ```-e```. For more informations, run
```
./compute_visual_results.sh -h
```

An interactive browsing tool has been released on our webpage [link]
