import itertools
import subprocess
import os

script_base = "examples/"
logging_base = "logs"

ro_info = [False, True]
lam_lrs = [1, None]
lams = [0.01, 10]
params = list(itertools.product(ro_info, lam_lrs, lams))

# 其他固定的超参数
fixed_args = [
    "--ro_info",
    "--do_train",
    "--do_eval",
    # "--model_name_or_path", "./layoutlmv3-base-1028",
    # "--output_dir", "./results/layoutlmv3-base-finetuned-custom-ner",
    "--overwrite_output_dir", "yes",
    "--segment_level_layout", "1",
    "--visual_embed", "1",
    "--input_size", "224",
    "--save_steps", "-1",
    "--evaluation_strategy", "steps",
    "--per_device_train_batch_size", "1",
    "--dataloader_num_workers", "8",
    "--logging_steps", "1",
]

# 指定 GPU 索引列表
gpu_indexes = [1,2,3,4,5,6,7]

# 创建进程对象，每个进程负责一个GPU
num_gpus = len(gpu_indexes)
processes = [None] * num_gpus

LR_HALF = {1e-4:1e-5, 2e-5:1e-5, 5e-5:2e-5}

datasets = [
    "cord",
    "sroie"
]
i = 0
for dataset in datasets:
    script_name = "run_ner.py"
    script_dir = os.path.join(script_base, script_name)
    training_volume = ["--max_steps", "2"]
    gradient_accumulation_steps = ["--gradient_accumulation_steps", "64"]
    eval_steps = ["--eval_steps", "100"]
    
    lrs = [1e-4] if dataset == 'cord' else [2e-5, 5e-5]
    # for size in ['large', 'base']:
    for size in ['base']:
        model_name_or_path = f"./layoutlmv3-{size}-2048"
        lrs = [LR_HALF[lr] for lr in lrs] if size=='large' else lrs
        ro_layers = 12 if size=='base' else 24
        for lr in lrs:
            for param in params:
                ro_info, lam_lr, lam = param
                if lam_lr is None:
                    lam_lr = lr
                if ro_info:
                    logging_dir = os.path.join(logging_base, dataset, f"{size}-{lr}-{lam}-{lam_lr}")
                else:
                    logging_dir = os.path.join(logging_base, dataset, f"{size}-{lr}")
                output_dir = os.path.join("./results", dataset, f"{size}-{lr}-{lam}-{lam_lr}")
                # 构建完整的命令
                python_command = " ".join([
                    "python", script_dir,
                    "--dataset_name", dataset,
                    "--model_name_or_path", model_name_or_path,
                    *fixed_args,
                    "--learning_rate", f"{lr}",
                    "--ro_info" if ro_info else "",
                    "--ro_layers", str(ro_layers),
                    "--lam_lr", str(lam_lr),
                    "--lam", str(lam),
                    "--logging_dir", logging_dir,
                    "--output_dir", output_dir,
                    *training_volume,
                    *gradient_accumulation_steps,
                    *eval_steps,
                ])

                gpu_index = gpu_indexes[i % num_gpus]
                i += 1
                command = [
                    "export CUDA_DEVICE_ORDER=PCI_BUS_ID",
                    f"export CUDA_VISIBLE_DEVICES={gpu_index}",
                    "export TOKENIZERS_PARALLELISM=false",
                    "export PYTHONPATH=\"/root/layout:$PYTHONPATH\"",
                    python_command,
                ]
                if processes[i % num_gpus] is not None:
                    processes[i % num_gpus].wait()
                processes[i % num_gpus] = subprocess.Popen(" && ".join(command), shell=True)