# Preprocess

We provide a minimal example dataset [here](https://forms.gle/E3Roi9zriu6Sk4557).

After downloading and extracting the dataset, place it in `data/feicuiwan_sample_folder`. You can verify its integrity by executing the following command:

```bash
python3 apps/test_dataset.py --cfg config/example_feicui/dataset.yml split dataset
```

To ensure that the point cloud in the dataset is correctly projected, we utilize Gaussian Splatting with the following command:

```bash
python3 apps/test_pointcloud.py --cfg config/example_feicui/dataset.yml split dataset radius 0.01
```

The output will be directed to the `debug/` directory.

Should everything proceed as expected, you can then advance to the model training phase.

## Prepare your own datasets

### 1. Image Preparation

Organize your images in the following structure:

```bash
<data>
├── images          
│   ├── DJI_0144.JPG
│   ├── DJI_0145.JPG
│   └── ...
```

If you capture the data with multiple cameras, you can organize them as follows:

```bash
<data>
└── images
    ├── h
    ├── q
    ├── x
    ├── y
    └── z
```

### 2. COLMAP Execution

Execute the colmap pipeline to obtain sparse reconstruction:

```bash
data=<path_to_your_dataset>
# single camera
colmap feature_extractor --database_path ${data}/database.db --image_path ${data}/images --ImageReader.camera_model OPENCV --ImageReader.single_camera 1  --SiftExtraction.use_gpu 0
# multiple cameras
colmap feature_extractor --database_path ${data}/database.db --image_path ${data}/images --ImageReader.camera_model OPENCV --ImageReader.single_camera_per_folder 1 --SiftExtraction.use_gpu 0
# matching and mapper
colmap exhaustive_matcher --database_path ${data}/database.db --SiftMatching.use_gpu 0 
mkdir -p ${data}/sparse
colmap mapper --database_path ${data}/database.db --image_path ${data}/images --output_path ${data}/sparse
```

### 3. Colmap Results Inspection

Inspect and validate the colmap results with the following command:

```bash
python3 apps/calibration/read_colmap.py ${data}/sparse/0 --min_views 2
```

### Output Folder Structure

Below is an example of the folder structure you can expect after running the COLMAP processes

```bash
<root>
├── database.db     
├── images
└── sparse
    └── 0
        ├── project.ini
        # Colmap-generated outputs
        ├── cameras.bin
        ├── images.bin
        ├── points3D.bin
        # Converted camera parameters
        ├── extri.yml
        ├── intri.yml
        # Converted point cloud data
        └── sparse.npz
```

**Note:** This step may be time-consuming.
While colmap is effective for a variety of scenarios, it can be inefficient for large-scale scene reconstruction.
We will release our modified version of colmap for large scene calibration in an upcoming version, which is a component of our work [Detector-free SfM](https://zju3dv.github.io/DetectorFreeSfM/).

### 4. Align with GPS info

Details on aligning with GPS information will be provided in future updates.
