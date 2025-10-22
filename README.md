# Mesh-Splat

Mesh-Splat is a research implementation built upon the official ["GaMeS: Mesh-Based Adapting and Modification of Gaussian Splatting"](https://arxiv.org/abs/2402.01459).
This project extends the original [official codebase](https://waczjoan.github.io/gaussian-mesh-splatting/) with additional utilities and experimental workflows for mesh-driven Gaussian Splatting and 3D rendering.

# Installation

## 1. Create the Conda environment
```bash
conda create --name meshsplat python=3.8
conda activate meshsplat
```

## 2. Configure CUDA (tested with CUDA 11.7) and PyTorch
We suggest using this way to setup CUDA environment.  

Ensure CUDA 11.7 is already installed:
```
cat /usr/local/cuda-*
```

Set environment variables:
```
conda env config vars set CUDA_HOME=/usr/local/cuda-11.7
conda env config vars set PATH=/usr/local/cuda-11.7/bin:$PATH
conda env config vars set LD_LIBRARY_PATH=/usr/local/cuda-11.7/lib64:$LD_LIBRARY_PATH
conda deactivate
conda activate meshsplat
```

Verify installation:
```
nvcc -V
```

Install PyTorch:
```
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2
```

Check CUDA availability:
```
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available())"
```

source: [Cuda and PyTorch Setup Guide \| SYJINTW](https://syjintw.github.io/posts/cuda-and-pytorch/)

## 3. Install dependencies 
```
pip install -r requirements.txt
```

## 4. Setup submodules
```
pip install ./submodules/diff-gaussian-rasterization
pip install ./submodules/simple-knn
```

## 5. Build and install PyTorch3D
```
mkdir ext
cd ext
git clone https://github.com/facebookresearch/pytorch3d.git
cd pytorch3d
pip install -e .
```

# Usage Example
## Training
```bash
python train.py --eval \                                                                      
-s /mnt/data1/syjintw/NEU/dataset/hotdog \
-m output/hotdog_testing \
--gs_type gs_mesh -w --iteration 10 \
--texture_obj_path /mnt/data1/syjintw/NEU/dataset/hotdog/mesh.obj \
--debugging
```

## Rendering
```bash
python ./render_mesh_splat.py \                                                               
-m output/hotdog_testing \
--gs_type gs_mesh \
--texture_obj_path /mnt/data1/syjintw/NEU/dataset/hotdog/mesh.obj
```

## Evaluation
```bash
python metrics.py \
-m /mnt/data1/syjintw/NEU/mesh-splat/output/hotdog_testing \
--gs_type gs_mesh
```

