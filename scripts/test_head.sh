#!/bin/bash
start_time=$SECONDS

# 检查参数
if [ $# -ne 3 ]; then
    echo "Usage: $0 expname data_path gt_path"
    exit 1
fi

expname=$1
data_path=$2
gt_path=$3

# 初始化 Conda
# 使用正确的 Miniconda 路径
CONDA_BASE=/home/yuzhiyuan/miniconda3
if [ ! -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
    echo "Conda initialization script not found at $CONDA_BASE/etc/profile.d/conda.sh"
    exit 1
fi
source "$CONDA_BASE/etc/profile.d/conda.sh"

# 检查并激活 Conda 环境
conda env list | grep imls_splatting || { echo "Environment imls_splatting not found"; exit 1; }
conda activate imls_splatting || { echo "Failed to activate imls_splatting environment"; exit 1; }

# 检查并进入 gaussian-splatting 目录
if [ ! -d "gaussian-splatting" ]; then
    echo "Directory gaussian-splatting not found"
    exit 1
fi
cd gaussian-splatting

# 检查第一个 train.py 和输入路径
if [ ! -f "train.py" ]; then
    echo "train.py not found in gaussian-splatting"
    exit 1
fi
if [ ! -d "$data_path" ]; then
    echo "Data path $data_path does not exist"
    exit 1
fi

# 运行第一个 train.py
echo "Running first train.py with expname=$expname, data_path=$data_path"
python3 train.py --eval --white_background --resolution 2 --expname "$expname" -s "$data_path" || { echo "First train.py failed"; exit 1; }

# 返回上级目录
cd ..

# 检查第二个 train.py 和 checkpoint
if [ ! -f "train.py" ]; then
    echo "train.py not found in current directory"
    exit 1
fi
checkpoint="./gaussian-splatting/output/$expname/point_cloud/iteration_30000/point_cloud.ply"
if [ ! -f "$checkpoint" ]; then
    echo "Checkpoint $checkpoint not found"
    exit 1
fi

# 运行第二个 train.py
echo "Running second train.py with expname=$expname, checkpoint=$checkpoint"
python3 train.py --eval --white_background --SSAA 1 --resolution 1 --expname "$expname" --meshscale 1.0 --start_checkpoint_ply "$checkpoint" -s "$data_path" || { echo "Second train.py failed"; exit 1; }

# 提取网格
echo "Running extract.py with expname=$expname"
python3 extract.py --config "./output/$expname/opt.json" --model_path "output/$expname" --save_path "output/$expname/pred.ply" --grid_resolution 512 || { echo "extract.py failed"; exit 1; }

# 检查 metric.py 和输入路径
if [ ! -f "metric.py" ]; then
    echo "metric.py not found"
    exit 1
fi
if [ ! -f "$gt_path" ]; then
    echo "Ground truth path $gt_path does not exist"
    exit 1
fi
pred_path="output/$expname/pred-high.ply"
if [ ! -f "$pred_path" ]; then
    echo "Predicted mesh $pred_path not found"
    exit 1
fi

# 运行 metric.py
echo "Running metric.py with gt_path=$gt_path, pred_path=$pred_path"
python metric.py --gt_path "$gt_path" --pred_path "$pred_path" --mode "head" || { echo "metric.py failed"; exit 1; }

# 计算并输出总耗时
end_time=$SECONDS
elapsed_time=$((end_time - start_time))
echo "Total execution time: $elapsed_time seconds"