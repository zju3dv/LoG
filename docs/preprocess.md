# Preprocess

We provide a minimal example dataset [here](https://forms.gle/E3Roi9zriu6Sk4557).

After downloading and extracting the dataset, place it in `data/feicuiwan_sample_folder`. You can verify its integrity by executing the following command:

```bash
python3 apps/test_dataset.py --cfg config/example/test/dataset.yml split dataset
```

To ensure that the point cloud in the dataset is correctly projected, we utilize Gaussian Splatting with the following command:

```bash
python3 apps/test_pointcloud.py --cfg config/example/test/dataset.yml split dataset radius 0.01
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

# Preprocess the Large Scenes

In this section, we demonstrate how to preprocess data from publicly available large scene datasets. 

### Hospital

We use oblique photography data `Hospital` from the paper [UrbanScene3D] as an example, which captures scenes spanning 500,000 square meters. Download the dataset from [UrbanScene3D-V1](https://github.com/Linxius/UrbanScene3D?tab=readme-ov-file#urbanscene3d-v1). The dataset consists of 2485 images obtained from 5 cameras. We follow the steps below to run COLMAP:

```bash
colmap feature_extractor --database_path ${data}/database.db --image_path ${data}/images --ImageReader.camera_model OPENCV --ImageReader.single_camera_per_folder 1 --SiftExtraction.use_gpu 0
colmap vocab_tree_matcher --database_path ${data}/database.db --VocabTreeMatching.vocab_tree_path ./vocab_tree_flickr100K_words1M.bin --VocabTreeMatching.num_images 100 --SiftMatching.use_gpu 0
mkdir ${data}/sparse
colmap mapper --database_path ${data}/database.db --image_path ${data}/images --output_path ${data}/sparse
```

This process takes approximately 2 days. Make sure to download `vocab_tree_flickr100K_words1M.bin` from the COLMAP website beforehand. To facilitate further processing, we align the calibrated coordinates to position the origin of the world coordinate system at the center of the scene, with the z-axis perpendicular to the ground. Since the `Hospital` dataset lacks GPS information, we cannot directly use GPS for alignment. Instead, we fit a plane using the positions of the capturing cameras and assume it is parallel to the ground for processing.

```bash
python3 apps/calibration/align_with_cam.py --colmap_path ${data}/sparse/0 --target_path ${data}/sparse_align
python3 apps/calibration/read_colmap.py ${data}/sparse_align --min_views 3
```

Output:

```bash
[Read Colmap] filter 2039201/2173461 points3D, min view = 3
H W set {(4000, 6000)}
num_cameras: 5/2485
num_images: 2485
num_points3D: 2039201
```

Check the dataset and undistort the images.

```bash
python3 apps/test_dataset.py --cfg config/example/Hospital/dataset.yml split dataset
python3 apps/calibration/run_midas.py data/Hospital/cache/8/images --multifolder
```

Render overlook:

```bash
python3 apps/train.py --cfg config/example/Hospital/train_wdepth.yml split demo_overlook
```

### Campus

UrbanScene3D also provides the `Campus` sequence, covering an area of 1,300,000 square meters. Download the dataset from [UrbanScene3D-V1](https://github.com/Linxius/UrbanScene3D?tab=readme-ov-file#urbanscene3d-v1). Since this dataset includes GPS information in the images, we can utilize GPS for image matching and final alignment.

```bash
colmap feature_extractor --database_path ${data}/database.db --image_path ${data}/images --ImageReader.camera_model OPENCV --ImageReader.single_camera_per_folder 1 --SiftExtraction.use_gpu 1
# matching use GPS info from images, max_distance=300
colmap spatial_matcher --database_path ${data}/database.db --SpatialMatching.max_num_neighbors 200 --SpatialMatching.max_distance 300 --SiftMatching.use_gpu 1
mkdir ${data}/sparse
colmap mapper --database_path ${data}/database.db --image_path ${data}/images --output_path ${data}/sparse
```

This process also takes approximately 2 days. After aligning with GPS information, the camera parameters are in units of 100 meters for easier handling.

```bash
# read the GPS info from exif data of image
python3 apps/calibration/read_gps_info.py --image_path ${data}/images --output_path ${data}/gps.npy --multifolder
# align with GPS info
python3 apps/calibration/align_with_gps.py --gps_path ${data}/gps.npy --colmap_path ${data}/sparse/0 --output_colmap_path ${data}/sparse_align
# read colmap
python3 apps/calibration/read_colmap.py ${data}/sparse_align --min_views 3
```

Output:

```bash
[Read Colmap] filter 3042288/3448793 points3D, min view = 3
H W set {(3648, 5472)}
num_cameras: 9/5871
num_images: 5871
num_points3D: 3042288
```

Check the dataset and undistort the images.

```bash
python3 apps/test_dataset.py --cfg config/example/Hospital/dataset.yml split dataset
python3 apps/calibration/run_midas.py data/Hospital/cache/8/images --multifolder
```

Render overlook:

```bash
python3 apps/train.py --cfg config/example/Hospital/train_wdepth.yml split demo_overlook
```