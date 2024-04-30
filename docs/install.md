# Installation

Follow these instructions to set up the LoG environment on your machine.

### Prerequisites

Ensure you have Anaconda or Miniconda installed to manage your environments and packages.

### Create and Activate Environment

```bash
conda create --name LoG python=3.10 -y
conda activate LoG
```

Install specific versions of PyTorch and torchvision for CUDA 11.8:

```bash
pip install torch==2.0.1+cu118 --extra-index-url https://download.pytorch.org/whl/cu118
pip install torchvision==0.15.2+cu118 --extra-index-url https://download.pytorch.org/whl/cu118
```

If you are familiar with Python and PyTorch, feel free to use your own versions of Python and torch that suit your needs.

Install other necessary packages from a requirements file:

```bash
git clone https://github.com/zju3dv/LoG.git
cd LoG
pip install -r requirements.txt
```

Clone and install the differential Gaussian rasterization library:

```bash
mkdir submodules && cd submodules
git clone https://github.com/graphdeco-inria/diff-gaussian-rasterization.git --recursive
pip install ./diff-gaussian-rasterization -v
# clone the modified gs
git clone https://github.com/chingswy/diff-gaussian-rasterization.git mydiffgaussian --recursive
cd mydiffgaussian
git checkout antialias
pip install . -v
cd ..
```

Install Simple-KNN for k-nearest neighbor searches:

```bash
git clone https://gitlab.inria.fr/bkerbl/simple-knn.git
pip install ./simple-knn -v
```

Finally, install the LoG package in editable mode to facilitate development:

```bash
cd ..
# installs packages in editable mode
pip install -e .
```