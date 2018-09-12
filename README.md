# Learning Relationship-aware Visual Features
This repository contains the code for reproducing results from our paper: **Learning Relationship-aware Visual Features**
![r-cbir](https://user-images.githubusercontent.com/25117311/45022838-7e171f80-b035-11e8-8b2c-2842582291c6.png)
## Get ready
1.  Download and extract CLEVR_v1.0 dataset: http://cs.stanford.edu/people/jcjohns/clevr/

2. Download this repository and all submodules with 
    ```
    git clone --recursive https://github.com/mesnico/learning-relationship-aware-visual-features
    ```
    This will also download [RelationNetworks-CLEVR](https://github.com/mesnico/RelationNetworks-CLEVR) repository as a submodule in the cloned directory.
    
3. Install virtualenv, if you haven't already:
    ```
    sudo pip3 install virtualenv 
    ```
4. Move into the cloned repository and run:
    ```
    ./setup.sh path/to/CLEVR_v1.0
    ```
    substituting ```path/to/CLEVR_v1.0``` with the path to your CLEVR extracted folder. This script will download RMAC features for CLEVR dataset and precalculated GED distances (ground-truth). Then, it will extract features from 2S-RN using pretrained IR model.
 
    
## Results
### Spearman-Rho correlation
In order to reproduce Spearman-Rho correlation values for RMAC, RN and 2S-RN features against the generated GT, run
```
./compute_results.sh -d path/to/CLEVR_v1.0
```
This script will compute distances, rankings and correlation values for both soft and hard matches.

**NOTE**:The first time this script is run may take some time; once finished, results are cached and final spearman-rho metrics will be immediately available at every successive run.

This script prints Spearman-Rho correlation values in the current terminal and creates a graphical visualization storing it in pdf files in the ```output``` folder.

In order to modify parameters such as *start* and *end* query indexes or number of processes used to compute GED distances, run 
```
./compute_results.sh -h
```

### Visual Feedback
It is possible to view the top relevant images using RMAC, RN and 2S-RN features, for a bunch of query images.
```
./compute_visual_results.sh -d path/to/CLEVR_v1.0
```
This script will create a pdf file in the ```output``` folder called ```visual_results.pdf``` showing retrieval results for every query image.
By default, only 10 query images are used. In order to change the range for query images you can specify parameters ```-s``` and ```-e```. For more informations, run
```
./compute_visual_results.sh -h
```
