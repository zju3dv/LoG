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


## [MiDaS](https://github.com/isl-org/MiDaS)

```bash
git clone https://github.com/isl-org/MiDaS.git --depth=1
cd MiDaS/weights
wget -c https://github.com/isl-org/MiDaS/releases/download/v3_1/dpt_beit_large_512.pt
cd ..
cp ../../docs/external/run_midas.py ./
# install extra packages
pip install timm==0.6.12 imutils
```

### Interactive GUI Documentation

We provide an interactive GUI that utilizes imgui and OpenGL for rendering. To operate this GUI, you must set up a desktop environment with a display on a Linux system. We have tested this setup on an Ubuntu system. Follow these steps to install the required dependencies:

```bash
cd submodules
git clone https://github.com/zju3dv/EasyVolcap.git
cd EasyVolcap
pip install -v -e . --no-deps
pip install pdbr h5py PyGLM imgui-bundle addict yapf ujson scikit-image cuda-python ruamel.yaml
cd ..
cd ..
```

To verify that the GUI is functioning correctly, execute the following command:

```bash
python3 apps/check_gui.py
```

The test script initializes a sequence of GS points randomly. You can interact with the GUI using the mouse to drag and rotate the viewpoint, scroll to zoom, and hold the right mouse button to pan the view. Keyboard controls using 'W', 'A', 'S', 'D' allow for camera movement. This setup provides a comprehensive way to explore and interact with the graphical content dynamically.

https://github.com/chingswy/LoGvideos/assets/22812405/95c5c010-4e1f-4273-89bf-c96c3a990b06
