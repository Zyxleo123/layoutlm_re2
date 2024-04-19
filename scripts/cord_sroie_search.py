import itertools
import subprocess
import os

script_base = "examples/"
logging_base = "logs"

ro_info = [True]
lam_lrs = [0.01, 0.05, 0.1, 0.15, 0.2, 0.5, 1.0]
lams = [0.01, 0.1, 1.0, 10.0]
params = []
params.append((False, 0, 0))
params.extend(list(itertools.product(ro_info, lam_lrs, lams)))

# 其他固定的超参数
fixed_args = [
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
    "--per_device_eval_batch_size", "1",
    "--dataloader_num_workers", "8",
    "--logging_steps", "1",
]
# 指定 GPU 索引列表
gpu_processes = {0:None, 1:None, 2:None, 3:None, 4:None, 5:None, 7:None}
available_gpus = gpu_processes.keys()

base_cord_lrs = [1e-4]
large_cord_lrs = [5e-5]
base_sroie_lrs = [7e-5, 9e-5, 1.2e-4]
large_sroie_lrs = [3e-5, 5e-5, 7e-5, 1e-4]
LRS = {"cord": {"base": base_cord_lrs, "large": large_cord_lrs}, "sroie": {"base": base_sroie_lrs, "large": large_sroie_lrs}}

datasets = [
    "cord",
    # "sroie",
]
i = 0
for dataset in datasets:
    script_name = "run_ner.py"
    script_dir = os.path.join(script_base, script_name)
    training_volume = ["--max_steps", "1000"] if dataset == 'cord' else ["--max_steps", "2000"]
    gradient_accumulation_steps = ["--gradient_accumulation_steps", "16"]
    eval_steps = ["--eval_steps", "100"]

    for size in ['base']:
    # for size in ['large', 'base']:
        length = 1028 if dataset == 'cord' else 1542
        model_name_or_path = f"./layoutlmv3-{size}-{length}"
        lrs = LRS[dataset][size]
        ro_layers = 12 if size=='base' else 24
        for lr in lrs:
            for param in params:
                ro_info, lam_lr, lam = param
                if lam_lr is None:
                    lam_lr = lr
                file_name = f"{size}-{lr}-{lam}-{lam_lr}" if ro_info else f"{size}-{lr}"
                # file_name += f"-{training_volume[1]}"
                logging_dir = os.path.join(logging_base, dataset, file_name)
                output_dir = os.path.join("./results", dataset, file_name)
                # 构建完整的命令
                python_command = " ".join([
                    # "nohup",
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

                gpu_index = None
                for gpu_idx in available_gpus:
                    if gpu_processes[gpu_idx] is None or gpu_processes[gpu_idx].poll() is not None:
                        gpu_processes[gpu_idx] = None
                        gpu_index = gpu_idx
                if gpu_index is None:
                    print("All GPUs are occupied, waiting for a free GPU...")
                    while gpu_index is None:
                        for gpu_idx in available_gpus:
                            if gpu_processes[gpu_idx] is None or gpu_processes[gpu_idx].poll() is not None:
                                gpu_processes[gpu_idx] = None
                                gpu_index = gpu_idx
                command = [
                    "export CUDA_DEVICE_ORDER=PCI_BUS_ID",
                    f"export CUDA_VISIBLE_DEVICES={gpu_index}",
                    "export TOKENIZERS_PARALLELISM=false",
                    "export PYTHONPATH=\"/root/layout:$PYTHONPATH\"",
                    python_command,
                ]
                # start a new process
                output_log = os.path.join('nohup', dataset, file_name)
                if os.path.exists(f"{output_log}.log"):
                    print(f"Skip {output_log}")
                    continue
                command = " && ".join(command)
                command += f" > {output_log}.log 2>&1"
                gpu_processes[gpu_index] = subprocess.Popen(command, shell=True)
                print(f"Scheduled {output_log} on GPU {gpu_index}...")