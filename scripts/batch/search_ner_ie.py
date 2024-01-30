import subprocess
import itertools
import os

# 定义超参数列表
ro_layers_values = [1, 3, 5, 7, 12]
lam_values = [0.0, 0.1, 0.5, 1.0, 10]
lam_lr_values = [0.02, 0.1, 0.3, 0.5, 0.7, 0.9]

# 生成超参数组合
param_combinations = list(itertools.product(ro_layers_values, lam_lr_values, lam_values))

# 脚本路径
script_path = "examples/run_ner.py"
logging_dir_base = "./logs/ner/ie/search/"

# 其他固定的超参数
lr = 7e-5
fixed_args = [
    "--dataset_name", "custom-ie",
    "--ro_info",
    "--do_train",
    "--do_eval",
    "--model_name_or_path", "./layoutlmv3-base-1028",
    "--output_dir", "./results/layoutlmv3-base-finetuned-custom-ner",
    "--overwrite_output_dir", "yes",
    "--segment_level_layout", "1",
    "--visual_embed", "1",
    "--input_size", "224",
    "--max_steps", "1000",
    "--save_steps", "-1",
    "--evaluation_strategy", "steps",
    "--eval_steps", "100",
    "--per_device_train_batch_size", "1",
    "--gradient_accumulation_steps", "16",
    "--dataloader_num_workers", "8",
    "--logging_steps", "1",
    "--learning_rate", f"{lr}",
]

# 指定 GPU 索引列表
gpu_indexes = [0, 1, 2, 3, 4, 5, 6, 7]

# 创建进程对象，每个进程负责一个GPU
num_gpus = len(gpu_indexes)
processes = [None] * num_gpus

# 循环遍历超参数组合并执行脚本
for i, (ro_layers, lam_lr, lam) in enumerate(param_combinations):
    gpu_index = gpu_indexes[i % num_gpus]
    
    # 构建完整的命令
    logging_dir = os.path.join(logging_dir_base, f"{lam}-{ro_layers}-{lam_lr}")
    python_command = " ".join([
        "python", script_path,
        *fixed_args,
        "--ro_layers", str(ro_layers),
        "--lam_lr", str(lam_lr),
        "--lam", str(lam),
        "--logging_dir", logging_dir,
    ])
    # python_command = f"sleep 2 && echo {i}"
    command = [
        f"export CUDA_DEVICE_ORDER=PCI_BUS_ID",
        f"export CUDA_VISIBLE_DEVICES={gpu_index}",
        "export TOKENIZERS_PARALLELISM=false",
        "export PYTHONPATH=\"/root/layout:$PYTHONPATH\"",
        python_command,
    ]
    # 启动进程
    if processes[i % num_gpus] is not None:
        processes[i % num_gpus].wait()
    processes[i % num_gpus] = subprocess.Popen(" && ".join(command), shell=True)

