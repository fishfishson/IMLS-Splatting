## Installation
The code is tested under Python 3.7.13, CUDA 11.3, and PyTorch 1.12.1 .
```
git clone https://github.com/SilenKZYoung/IMLS-Splatting.git
cd IMLS-Splatting
conda env create --file environment.yml
conda activate imls_splatting
```

## Training 
Run point cloud and normal initialization 
```
cd gaussian-splatting
python3 train.py --eval --white_background --resolution 2 --expname $exp_name -s $data_path
# for example
python3 train.py --eval --white_background --resolution 2 --expname 0103 -s ./data/thuman2.1_mv/0103
```

Run IMLS-Splatting
```
cd ..
python3 train.py --eval --white_background --SSAA 1 --resolution 1 --expname $exp_name --meshscale $scene_scale  --start_checkpoint_ply $3DGS_ply_path  -s  $data_path
# for example
python3 train.py --eval --white_background --SSAA 1 --resolution 1 --expname lego --meshscale 2.0 --start_checkpoint_ply ./gaussian-splatting/output/0103/point_cloud/iteration_30000/point_cloud.ply -s ./data/thuman2.1_mv/0103
```
