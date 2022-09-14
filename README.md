# View and try the demos
We provide 3 options to view and try our demo.
## Option 1: docker (recommended).
Note 1: This container was tested on Ubuntu [CPU_only, P100, V100] and Windows 11 with [Docker Desktop](https://www.docker.com/products/docker-desktop/) [CPU_only, P100, V100, GeForce930M (laptop GPU)]. 
We do not support A100 as we use [MolecularTransformer](https://github.com/pschwllr/MolecularTransformer.git) as git submodule, and its cuda version is not supported.

Note 2: If you are runing on Windows with GPU, please follow the [GPU Support](https://docs.docker.com/desktop/windows/wsl/#gpu-support) of [Docker Desktop](https://www.docker.com/products/docker-desktop/)

    # the below command will start a jupyter container, click the 
    # link 'http://127.0.0.1:8888/?token=xxxxx' and open the 
    # 'demo.ipynb' notebook. You can try our demo with your own cases.    

    # cpu only
    docker run -p 8888:8888 leifa/demo:1    

    # gpu
    docker run --gpus 1  -p 8888:8888 leifa/demo:1
    # NOTE: Please UNCOMMENT '-gpu 1' in `demo_data/subseq2seq.sh` with jupyter to use the GPU for substructure sequence to sequence inference, you can also change the batch size if necessary.


    # kill the container
    docker ps # this will show CONTAINER_ID in the first column
    docker kill CONTAINER_ID


## Option 2: setup on your own machine
Note: the following setup was tested on Ubuntu [CPU_only (default), P100, V100].

### step 2-1: download the [code and model](https://bdmstorage.blob.core.windows.net/shared/demo.tar.gz) 
    wget https://bdmstorage.blob.core.windows.net/shared/demo.tar.gz 
    tar -xzvf demo.tar.gz 

### step 2-2: clone submodule in the code folder 
    cd MCB_SMILES/
    git clone https://github.com/pschwllr/MolecularTransformer.git MolecularTransformer
    git clone https://github.com/jcyk/copyisallyouneed.git RetrievalModel 
    # the directory layout:
    MCB_SMILES
    ├── ...
    ├── demo_data 
    ├── MolecularTransformer  
    ├── RetrievalModel   
    ├── demo.ipynb     
    └── ...  

### step 2-3: setup python environments in the code folder with [conda](https://conda.io/projects/conda/en/latest/user-guide/install/linux.html#installing-on-linux)
    # install conda
    wget https://repo.anaconda.com/miniconda/Miniconda3-py39_4.12.0-Linux-x86_64.sh
    # input yes when asked to run conda init
    bash Miniconda3-py39_4.12.0-Linux-x86_64.sh 
    # reopen the bash terminal

    # setup for reaction retrieval
    conda create -n retrieval python=3.6 -y
    conda run -n retrieval pip install -r RetrievalModel/requirements.txt -f https://download.pytorch.org/whl/torch_stable.html

    # setup for substructure extraction and substructure seq2seq
    conda create -n mol_transformer -c pytorch -c conda-forge -y ipykernel rdkit=2022.03.1 tqdm func_timeout pytorch=0.4.1 notebook future six tqdm pandas torchvision python=3.7
    cd MolecularTransformer && conda run -n mol_transformer pip install torchtext==0.3.1 && conda run -n mol_transformer pip install -e . && cd ..

### step 2-4: change code of submodule (reaction retrieval) to run on CPU in the code folder.
    sed -i 's/device =/# device =/g'  RetrievalModel/search_index.py
    sed -i 's/model.to(device)/# model.to(device)/g'  RetrievalModel/search_index.py
    sed -i 's/mips.to_gpu()/# mips.to_gpu()/g'  RetrievalModel/search_index.py
    sed -i 's/model.cuda()/# model.cuda()/g'  RetrievalModel/search_index.py
    sed -i "s/q = move_to_device(batch, torch.device('cuda')).t()/q = torch.from_numpy(batch).contiguous().t() /g"  RetrievalModel/search_index.py
    sed -i 's/model = torch.nn.Data/# model = torch.nn.Data/g'  RetrievalModel/search_index.py
    

### step 2-5: run `demo.ipynb` in the code folder, and try your own cases.
    conda activate mol_transformer && jupyter notebook
    # NOTE: Please UNCOMMENT '-gpu 1' in `demo_data/subseq2seq.sh` with jupyter to use the GPU for substructure sequence to sequence inference, you can also change the batch size if necessary.
    

## Option 3: view our prepared cases at https://github.com/fangleigit/demos .
