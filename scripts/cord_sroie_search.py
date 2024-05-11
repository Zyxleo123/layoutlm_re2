import itertools
import subprocess
import os

script_base = "examples/"
logging_base = "logs"

ro_info = [True]
lam_lrs = [0.01, 0.05, 0.1, 0.5, 1.0]
lams = [0.01, 0.1, 1.0, 10.0]
params = []
params.append((False, 0, 0))
params.extend(list(itertools.product(ro_info, lam_lrs, lams)))

# 其他固定的超参数
batch_size = 1
fixed_args = [
    "--do_train",
    "--do_eval",
    "--overwrite_output_dir", "yes",
    "--segment_level_layout", "1",
    "--visual_embed", "1",
    "--input_size", "224",
    "--save_steps", "-1",
    "--evaluation_strategy", "steps",
    "--per_device_train_batch_size", str(batch_size),
    "--per_device_eval_batch_size", str(batch_size),
    "--dataloader_num_workers", "1",
    "--logging_steps", "1",
]
# 指定 GPU 索引列表
gpu_processes = {5:None}
# gpu_processes = {4:None}
available_gpus = gpu_processes.keys()

lrs = {'cord': {'base': 1e-4, 'large': 5e-5}, 'sroie': {'base': 7e-5, 'large': 3e-5}}

datasets = [
    "cord",
    # "sroie",
]
for dataset in datasets:
    dataset_name = ["--dataset_name", "cord"] if dataset == "cord" else [""] # run_sroie_ner.py does not need dataset_name
    script_name = "run_ner.py" if dataset == "cord" else "run_sroie_ner.py"
    script_dir = os.path.join(script_base, script_name)
    training_volume = ["--max_steps", "1000"]
    eval_steps = ["--eval_steps", "100"]
    gradient_accumulation_steps = 16 // batch_size if dataset == "cord" else 16 // batch_size
    gradient_accumulation_steps = ["--gradient_accumulation_steps", str(gradient_accumulation_steps)]

    for size in ['large']:
        length = 514
        model_name_or_path = f"./layoutlmv3-{size}-{length}"
        ro_layer = 12 if size == 'base' else 24
        for param in params:
            ro_info, lam_lr, lam = param
            lr = lrs[dataset][size]
            if lam_lr is None:
                lam_lr = lr
            file_name = f"{size}-{lr}-{lam}-{lam_lr}" if ro_info else f"{size}-{lr}"
            logging_dir = os.path.join(logging_base, dataset, file_name)
            output_dir = os.path.join("./results", dataset, file_name)
            # 构建完整的命令
            python_command = " ".join([
                # "nohup",
                "python", script_dir,
                *dataset_name,
                "--model_name_or_path", model_name_or_path,
                *fixed_args,
                "--learning_rate", f"{lr}",
                "--ro_info" if ro_info else "",
                "--ro_layers", str(ro_layer),
                "--lam_lr", str(lam_lr),
                "--lam", str(lam),
                "--logging_dir", logging_dir,
                "--output_dir", output_dir,
                *gradient_accumulation_steps,
                *training_volume,
                *eval_steps,
            ])
            gpu_index = None
            for gpu_idx in available_gpus:
                if gpu_processes[gpu_idx] is None or gpu_processes[gpu_idx].poll() is not None:
                    gpu_processes[gpu_idx] = None
                    gpu_index = gpu_idx
            while gpu_index is None:
                for gpu_idx in available_gpus:
                    if gpu_processes[gpu_idx] is None or gpu_processes[gpu_idx].poll() is not None:
                        # if error, abort the process
                        if gpu_processes[gpu_idx].poll() is not None and gpu_processes[gpu_idx].poll() != 0:
                            print(f"Error on GPU {gpu_idx}!")
                            exit(1)
                        gpu_processes[gpu_idx] = None
                        gpu_index = gpu_idx
            command = [
                "export CUDA_DEVICE_ORDER=PCI_BUS_ID",
                f"export CUDA_VISIBLE_DEVICES={gpu_index}",
                "export http_proxy=127.0.0.1:7890",
                "export https_proxy=127.0.0.1:7890",
                "export TOKENIZERS_PARALLELISM=false",
                "export PYTHONPATH=\"/root/layout:$PYTHONPATH\"",
                python_command,
            ]
            # start a new process
            output_log = os.path.join('nohup_sroie_word', dataset, file_name)
            if os.path.exists(f"{output_log}.log"):
                print(f"Skip {output_log}")
                continue
            command = " && ".join(command)
            command += f" > {output_log}.log 2>&1"
            gpu_processes[gpu_index] = subprocess.Popen(command, shell=True)
            print(f"Scheduled {output_log} on GPU {gpu_index}...", flush=True)