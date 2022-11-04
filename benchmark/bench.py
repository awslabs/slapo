import os
import sys
import re
import json

import torch
import matplotlib.pyplot as plt
from dataclasses import dataclass, asdict
from transformers import AutoConfig, PretrainedConfig

MEGATRON_NO_FUSE = True
BATCH_SIZE = 8

print("Pytorch version\t:", torch.__version__)
print("CUDA version\t:", torch.version.cuda)

for i in range(torch.cuda.device_count()):
    print(f"GPU{i}\t\t:", torch.cuda.get_device_name(i))


@dataclass
class Exp:
    name: str  # Experiment name
    model: str  # huggingface model name
    batch_size: int  # batch size per GPU
    seq_len: int = None  # input sequence length

    ## Improve speed / reduce memory
    bf16: bool = False  # Faster, less memory. Recommend if GPU supports
    fp16: bool = False  # Faster, less memory, but need to scale loos.
    # Recommend if BF16 is not available.
    optim: str = "adamw_hf"  # Optimization method
    grad_ckpt: bool = False  # save memory with an extra forward
    grad_accum: int = 1  # accumulate gradients for better performance
    steps: int = 20  # number of parameter updates

    ## Multi-GPUs
    gpus: str = "0"  # GPUs to use. "0,1" means use GPU 0 and 1
    tensor_para: int = 1  # Tensor parallelism
    deepspeed: bool = False  # if or not use deepspeed
    ds_config: str = ""  # deepspeed config

    ## kwargs
    kwargs: dict = None

    def __post_init__(self):
        model_conf = AutoConfig.from_pretrained(self.model)
        get = lambda *keys: max(
            [getattr(model_conf, k) if hasattr(model_conf, k) else 0 for k in keys]
        )
        self.num_layers = get("num_hidden_layers", "n_layer")
        self.num_gpus = len(self.gpus.split(","))
        self.hidden_size = get("hidden_size", "n_embd", "d_model")
        self.vocab_size = get("vocab_size")
        self.num_heads = get("num_attention_heads", "n_head")
        if self.seq_len is None:
            self.seq_len = get("max_position_embeddings", "n_ctx")
        n, h, s, v = self.num_layers, self.hidden_size, self.seq_len, self.vocab_size
        att, ffn, embed = (
            4 * h * s**2 + 8 * s * h**2,
            16 * s * h**2,
            2 * s * h * v,
        )
        forward = n * (att + ffn) + embed
        # TFLOPs to train one example
        self.tflops = (4 * forward if self.grad_ckpt else 3 * forward) / 1e12
        if self.deepspeed:
            self.launcher = "deepspeed"
        else:
            self.launcher = f"torchrun --nproc_per_node {self.num_gpus}"

    def print_results(self):
        print("Total samples / second\t: %.1f" % self.samples_per_sec)
        print("Per GPU memory (GB)\t: %.1f" % self.gpu_mem)
        print(
            "Per GPU TFLOPs\t\t: %.1f"
            % (self.samples_per_sec * self.tflops / self.num_gpus)
        )


def compare(exps, fig_name):
    fig, ax = plt.subplots(ncols=3, figsize=(9, len(exps) / 2))
    x = list(range(len(exps)))
    for i, (y, l) in enumerate(
        (
            ([e.samples_per_sec for e in exps], "Samples / sec"),
            (
                [e.samples_per_sec * e.tflops / e.num_gpus for e in exps],
                "per GPU TFLOPS",
            ),
            ([e.gpu_mem for e in exps], "per GPU memory (GB)"),
        )
    ):
        bar = ax[i].barh(
            x, y, align="center", height=0.6, color=plt.get_cmap("Set1")(x)
        )
        ax[i].bar_label(bar, fmt="%.2f", label_type="center")
        ax[i].invert_yaxis()
        ax[i].set_xlabel(l)
        if i == 0:
            ax[i].set_yticks(x, labels=[e.name for e in exps])
        else:
            ax[i].set_yticklabels([])

    plt.title(fig_name)
    plt.savefig(fig_name + ".png", format="png", dpi=200, bbox_inches="tight")
    plt.show()


def megatron_bert(exp, script_file=None):
    import megatron

    if script_file is None:
        path = megatron.__path__[0]
        script_file = f"{path}/../pretrain_bert.py"

    cmd = f"""{exp.launcher} {script_file} \
--num-layers {exp.num_layers} --hidden-size {exp.hidden_size} \
--num-attention-heads {exp.num_heads} \
--tensor-model-parallel-size {exp.tensor_para} \
--micro-batch-size {exp.batch_size} \
--seq-length {exp.seq_len} --max-position-embeddings {exp.seq_len} \
--train-iters {exp.steps} \
--data-path bert-sample_text_sentence \
--vocab-file bert-large-uncased-vocab.txt \
--data-impl mmap --lr 0.00015 --log-interval 5"""
    if exp.bf16:
        cmd += " --bf16"
    if exp.fp16:
        cmd += " --fp16"

    if exp.kwargs is not None and "flags" in exp.kwargs:
        cmd += " " + " ".join(exp.kwargs["flags"])

    cmd += " > megatron.log 2>&1"
    print(cmd)
    os.system(cmd)
    ret = megatron_log(exp, "megatron.log")
    if ret is not None:
        ret.print_results()
    else:
        ret = exp
        ret.samples_per_sec = 0
        ret.gpu_mem = 0
    return ret


def megatron_log(exp, log_filename):
    with open(log_filename) as f:
        text = f.read()
    # Find the last number after the key, returns 0 if not exists
    query = lambda key: float(
        next(iter(reversed(re.findall(key + ": +([\d\.]+)", text))), 0)
    )
    if "CUDA out of memory" in text:
        print("Out of GPU memory, try a smaller batch size")
        return
    iter_time = query("elapsed time per iteration \(ms\)")
    if iter_time == 0:
        print(f'Failed. Check "{log_filename}" to find error')
        return
    exp.samples_per_sec = query("global batch size") / iter_time * 1e3
    exp.gpu_mem = query("max allocated") / 1e3
    print(
        "Breakdown(ms)\t\t: total %.2f, forward %.2f, backward %.2f, backward-params-all-reduce %.2f, optimizer %.2f"
        % (
            iter_time,
            query("forward-compute"),
            query("backward-compute"),
            query("backward-params-all-reduce"),
            query("optimizer"),
        )
    )
    return exp


kwargs = {}
if MEGATRON_NO_FUSE:
    kwargs = {
        "flags": [
            "--no-bias-gelu-fusion",
            "--no-bias-dropout-fusion",
            "--no-persist-layer-norm",
            "--no-masked-softmax-fusion",
        ]
    }

hf_bert = []
for idx, n_gpu in enumerate((1, 2, 4, 8)):
   gpus = ",".join([str(e) for e in range(n_gpu)])
   batch_size = BATCH_SIZE * (idx + 1)
   hf_bert.append(
       megatron_bert(
           Exp(
               f"HF bs{batch_size} ({n_gpu} GPU)",
               "bert-large-uncased",
               batch_size,
               fp16=True,
               gpus=gpus,
               tensor_para=n_gpu,
               kwargs=kwargs,
           ),
           script_file="./../examples/bert/pretrain_hf_bert.py"
       )
   )
compare(hf_bert, f"HF-MS")

mega_bert = []
no_fuse = " no fuse" if MEGATRON_NO_FUSE else ""
for idx, n_gpu in enumerate((1, 2, 4, 8)):
    gpus = ",".join([str(e) for e in range(n_gpu)])
    batch_size = BATCH_SIZE * (idx + 1)
    mega_bert.append(
        megatron_bert(
            Exp(
                f"Megatron bs{batch_size}{no_fuse} ({n_gpu} GPU)",
                "bert-large-uncased",
                batch_size,
                fp16=True,
                gpus=gpus,
                tensor_para=n_gpu,
                kwargs=kwargs,
            )
        )
    )
compare(mega_bert, f"Megatron{'-nofuse' if no_fuse else ''}")
