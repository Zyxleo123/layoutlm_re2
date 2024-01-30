import subprocess
import os

script_base = "examples/"
logging_base = "./logs/"

dataset_task_to_params = {
    "ner_ie": [[10, 3, 0.7], [0.1, 1, 0.1]], 
    "ner_ori": [[0, 3, 3], [1.0, 7, 0.1]],
    "re_ie": [[10, 12, 0]],
    "re_ori": [[5, 12 ,0.01]],
}

seeds = [21, 28, 42]

# 其他固定的超参数
fixed_args = [
    "--ro_info",
    "--do_train",
    "--do_eval",
    "--model_name_or_path", "./layoutlmv3-base-1028",
    "--output_dir", "./results/layoutlmv3-base-finetuned-custom-ner",
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
gpu_indexes = [0, 1, 2, 3, 4, 5, 6, 7]

# 创建进程对象，每个进程负责一个GPU
num_gpus = len(gpu_indexes)
processes = [None] * num_gpus

i = 0
for task_dataset, params in dataset_task_to_params.items():
    task, dataset = task_dataset.split("_")
    script_name = "run_" + task + ".py"
    script_dir = os.path.join(script_base, script_name)
    dataset_name = "custom-" + dataset
    if task == "re":
        lr = 1e-5
    elif dataset == "ori":
        lr = 3e-5
    else:
        lr = 7e-5
    training_volume = ["--num_train_epochs", "400"] if task == "re" else ["--max_steps", "1000"]
    gradient_accumulation_steps = ["--gradient_accumulation_steps", "16"] if task == "ner" else []
    eval_steps = ["--eval_steps", "100"] if task == "ner" else ["--eval_steps", "1000"]
    for param in params:
        for seed in seeds:
            gpu_index = gpu_indexes[i % num_gpus]
            i += 1
            lam, ro_layers, lam_lr = param
            logging_dir = os.path.join(logging_base, task, dataset, "run", f"{lam}-{ro_layers}-{lam_lr}-{seed}")
            # 构建完整的命令
            python_command = " ".join([
                "python", script_dir,
                "--dataset_name", dataset_name,
                *fixed_args,
                "--ro_layers", str(ro_layers),
                "--lam_lr", str(lam_lr),
                "--lam", str(lam),
                "--seed", str(seed),
                "--logging_dir", logging_dir,
                *training_volume,
                *gradient_accumulation_steps,
                *eval_steps,
                "--learning_rate", f"{lr}",
            ])
            command = [
                "export CUDA_DEVICE_ORDER=PCI_BUS_ID",
                f"export CUDA_VISIBLE_DEVICES={gpu_index}",
                "export TOKENIZERS_PARALLELISM=false",
                "export PYTHONPATH=\"/root/layout:$PYTHONPATH\"",
                python_command,
            ]
            # 启动进程
            print(" && ".join(command))
            print('-' * 100)
            # if processes[i % num_gpus] is not None:
            #     processes[i % num_gpus].wait()
            # processes[i % num_gpus] = subprocess.Popen(" && ".join(command), shell=True)