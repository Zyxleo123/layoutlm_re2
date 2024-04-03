import itertools
import subprocess
import os

script_base = "examples/"
logging_base = "./logs-large/"

seeds = [42]

# 其他固定的超参数
fixed_args = [
    "--do_train",
    "--do_eval",
    # "--model_name_or_path", "./layoutlmv3-base-1028",
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

gpu_processes = {0: None, 2: None}
available_gpus = gpu_processes.keys()


task_datasets = [
    # "re_ori",
    # "re_ie",
    "ner_ori",
    "ner_ie",
]
for ft in (None,):
    for task_dataset in task_datasets:
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
        model_name_or_path = "./layoutlmv3-large-" + task + "-1028" + ("-ft" if ft == "ft" else "")
        logging_dir = os.path.join(logging_base, task, dataset, "seed-d", "ft" if ft == "ft" else "pretrain")
        for seed in seeds:
            python_command = " ".join([
                "python", script_dir,
                "--dataset_name", dataset_name,
                "--model_name_or_path", model_name_or_path,
                *fixed_args,
                "--seed", str(seed),
                "--logging_dir", logging_dir,
                *training_volume,
                *gradient_accumulation_steps,
                *eval_steps,
                "--learning_rate", f"{lr}",
            ])
            # test every process, to see if threre is any process that has finished
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
            gpu_processes[gpu_index] = subprocess.Popen(" && ".join(command), shell=True)