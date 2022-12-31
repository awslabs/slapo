import os

models = ["bert-large-uncased", "roberta-large", "albert-large-v2", "EleutherAI/gpt-neo-1.3B", "facebook/opt-350m", "t5-large", "wideresnet-250M"]
impls = ["slapo-megatron", "slapo-deepspeed"]

model_name_mapping = {
    "bert": "bert-large-uncased",
    "roberta": "roberta-large",
    "albert": "albert-large-v2",
    "gpt": "EleutherAI/gpt-neo-1.3B",
    "opt": "facebook/opt-350m",
    "t5": "t5-large",
    "wideresnet": "wideresnet-250M",
}

for model in models:
    for impl in impls:
        for n_gpu in [2, 4, 8]:
            if any(m in model for m in ["opt", "t5", "gpt"]):
                seq_len = 1024
            else:
                seq_len = 512
            print(f"Running {impl} on {model} with {n_gpu} GPU")
            batch_size = "batch_size"
            ckpt_ratio = "ckpt_ratio"
            cmd = "python3 -m slapo.tune"
            cmd += f" --config {os.getcwd()}/../examples/{model}/tune_cfg.py"
            cmd += f" --db results/{model}-gpu{n_gpu}-{impl}.json"
            cmd += f" --error-stop symbol"
            cmd += f"  bench_single_node.py {impl}"
            cmd += f" --model {model_name_mapping[model]} --gpus {n_gpu} --seq-len {seq_len}"
            if model == "t5-large":
                cmd += " --seq-len-dec 512"
            cmd += f" --batch-size {batch_size}"
            cmd += f" --gradient-checkpoint {ckpt_ratio}"
            print(cmd)
            os.system(cmd)
