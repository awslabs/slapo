# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import importlib
import math
import os
import subprocess
from collections import OrderedDict
from dataclasses import dataclass

import matplotlib.pyplot as plt
import pkg_resources
import torch
from transformers import AutoConfig

from deepspeed_utils import add_deepspeed_parser, parse_deepspeed_kwargs, run_deepspeed
from megatron_utils import add_megatron_parser, parse_megatron_kwargs, run_megatron


@dataclass
class Exp:
    name: str  # Experiment name
    model: str  # huggingface model name
    batch_size: int  # batch size per GPU
    seq_len: int  # input sequence length
    seq_len_dec: int  # Decoder sequence length. Encoder-decoder model only.

    impl: str  # Implementation. "slapo" or "megatron"

    ## Improve speed / reduce memory
    bf16: bool = False  # Faster, less memory. Recommend if GPU supports
    fp16: bool = False  # Faster, less memory, but need to scale loos.
    optim: str = "adamw_hf"  # Optimization method
    grad_ckpt: str = ""  # Empty means no checkpointing; otherwise:
    # Megatron: "full" or "selective".
    # Slapo: a floating point indicating
    # the checkpointing ratio. For example, 0.5 means
    # to checkpoint a half of layers.
    grad_accum: int = 1  # accumulate gradients for better performance
    steps: int = 40  # number of parameter updates

    ## Multi-GPUs
    gpus: str = "0"  # GPUs to use. "0,1" means use GPU 0 and 1
    tensor_para: int = 1  # Tensor parallelism

    ## kwargs
    kwargs: dict = None

    def __post_init__(self):
        self.num_gpus = len(self.gpus.split(","))
        self.launcher = f"torchrun --nproc_per_node {self.num_gpus}"

        try:
            model_conf = AutoConfig.from_pretrained(self.model)
        except:
            # Not a model in HF hub. Use fake values to avoid Megatron errors.
            self.num_layers = 1
            self.hidden_size = 1
            self.vocab_size = 1
            self.num_heads = 1
            self.tflops = 0
            return

        get = lambda *keys: max(
            [getattr(model_conf, k) if hasattr(model_conf, k) else 0 for k in keys]
        )
        self.num_layers = get("num_hidden_layers", "n_layer")
        self.hidden_size = get("hidden_size", "n_embd", "d_model")
        self.vocab_size = get("vocab_size")
        self.num_heads = get("num_attention_heads", "n_head")
        if self.seq_len_dec == 0:
            # Encoder or decoder only models.
            n, h, s, v = (
                self.num_layers,
                self.hidden_size,
                self.seq_len,
                self.vocab_size,
            )
            att, ffn, embed = (
                4 * h * s**2 + 8 * s * h**2,
                16 * s * h**2,
                2 * s * h * v,
            )
            forward = n * (att + ffn) + embed
            # TFLOPs to train one example. Note that we use model TFLOPS instead of
            # hardware TFLOPS, so having checkpoints or not does not matter.
            self.tflops = 3 * forward / 1e12
        else:
            # Encoder-decoder models.
            self.num_decoder_layers = get("num_decoder_layers")
            self.d_kv = get("d_kv")
            self.d_ff = get("d_ff")

            h, s_e, s_d, v = (
                self.hidden_size,
                self.seq_len,
                self.seq_len_dec,
                self.vocab_size,
            )

            # If not specified in HF config, num_decoder_layers are the same as num_layers.
            l_e, l_d = self.num_layers, self.num_decoder_layers

            # Calculate TFLOPS of T5.
            gated = False  # HF/Megatron T5 don't gate by default.

            # Note that we use model TFLOPS instead of
            # hardware TFLOPS, so having checkpoints or not does not matter.
            c = 3  # 4 if self.grad_ckpt else 3

            enc_flops = 1 + s_e / 6 / h
            if gated:
                enc_flops += 1 / 3 + 1 / 6 / h
            enc_flops *= c * l_e * 24 * s_e * h**2

            dec_flops = (
                1
                + 1 / 6
                + s_e / 6 / s_d
                + s_d / 6 / h
                + s_e / 6 / h
                + v / 4 / c / l_d / h
            )
            if gated:
                dec_flops += 1 / 3 + 1 / 6 / h
            dec_flops *= 24 * c * s_d * l_d * h**2

            # TFLOPs to train one example
            self.tflops = (enc_flops + dec_flops) / 1e12

    def print_results(self, append_to=""):
        prefix = f"{self.impl}\t{self.model}\t{self.seq_len}\t{self.seq_len_dec}\t"
        prefix += f"{len(self.gpus.split(','))}\t{self.batch_size}\t{self.grad_ckpt}"

        def append_log(msg):
            if append_to:
                with open(append_to, "a") as filep:
                    filep.write(f"{prefix}\t{msg}\n")

        if self.error_code == 1:
            append_log(f"na\toom\toom\toom")
        elif self.error_code == 2:
            append_log(f"na\tfail\tfail\tfail")
        else:
            print("Total samples / second\t: %.1f" % self.samples_per_sec)
            print("Per GPU memory (GB)\t: %.1f" % self.gpu_mem)
            per_gpu_tflops = self.samples_per_sec * self.tflops / self.num_gpus
            print("Per GPU TFLOPs\t\t: %.1f" % per_gpu_tflops)
            append_log(
                f"{self.param_per_gpu / 1e6:.2f}\t{self.samples_per_sec:.2f}\t"
                f"{self.gpu_mem:.2f}\t{per_gpu_tflops:.2f}"
            )


def parse_args():
    common_parser = argparse.ArgumentParser(add_help=False)
    common_parser.add_argument("--model", type=str, help="Model name")
    common_parser.add_argument(
        "--error-stop",
        action="store_true",
        help="Stop when error occurs. Note that out-of-memory is "
        "not considdered as error",
    )
    common_parser.add_argument(
        "--seq-len", type=int, default=512, help="Sequence length. Default: 512"
    )
    common_parser.add_argument(
        "--seq-len-dec",
        type=int,
        default=0,
        help="Decoder sequence length. Only used by encoder-decoder models. Default: 0",
    )
    common_parser.add_argument(
        "--dtype", type=str, default="fp16", help="Model dtype. Default: fp16"
    )
    common_parser.add_argument(
        "--gpus",
        type=str,
        default="pow2",
        help="Number of GPUs to be used. Options: "
        "1. A single number (e.g., 1); "
        "2. A comma separated list of GPU numbers (e.g., 1,2); "
        "3. A string 'pow2' (e.g., pow2) to cover power of 2 GPUs (e.g., 1,2,4,8).",
    )
    common_parser.add_argument(
        "--batch-sizes",
        type=str,
        default="8 * int(math.log2(n) + 1)",
        help="An expression with `n` as GPU number to to calculate batch size. "
        "`math` can be used. Default: 8 * int(math.log2(n) + 1)",
    )
    common_parser.add_argument(
        "--gradient-checkpoint",
        type=str,
        default="0",
        help="Gradient checkpointing. Empty means no checkpointing; otherwise: "
        "Megatron: 'full' or 'selective'. HF w. schedule: a floating point indicating "
        "the checkpointing ratio. For example, 0.5 means to checkpoint "
        "a half of layers.",
    )
    common_parser.add_argument(
        "--draw-fig",
        action="store_true",
        help="Draw a figure of the results",
    )
    common_parser.add_argument(
        "--append-to",
        type=str,
        default="",
        help="Append the results to a file",
    )

    parser = argparse.ArgumentParser()
    subprasers = parser.add_subparsers(
        dest="impl",
        help="Model implementation (slapo-megatron, slapo-deepspeed, "
        "megatron, deepspeed, eager, torchscript, and env). "
        "Note that 'env' only dumps environments without benchmarking.",
    )
    subprasers.add_parser(
        "slapo-megatron",
        parents=[common_parser],
        help="HuggingFace model with Slapo on Megatron",
    )
    subprasers.add_parser(
        "slapo-deepspeed",
        parents=[common_parser],
        help="HuggingFace model with Slapo on DeepSpeed ZeRO-3",
    )
    subprasers.add_parser(
        "env",
        parents=[common_parser],
        help="Dump environment variables",
    )
    subprasers.add_parser(
        "eager",
        parents=[common_parser],
        help="PyTorch Eager Mode",
    )
    subprasers.add_parser(
        "torchscript",
        parents=[common_parser],
        help="TorchScript implementation",
    )

    add_deepspeed_parser(common_parser, subprasers)
    add_megatron_parser(common_parser, subprasers)
    return parser.parse_args()


def parse_gpus(gpus):
    n_gpu = torch.cuda.device_count()
    if gpus == "pow2":
        n_gpus = [2**i for i in range(int(math.log2(n_gpu)) + 1)]
    elif "," in gpus:
        n_gpus = [int(e) for e in gpus.split(",")]
    else:
        n_gpus = [int(gpus)]

    assert (
        min(n_gpus) > 0 and max(n_gpus) <= n_gpu
    ), f"GPU numbers must be in 0 - {n_gpu}, but got {n_gpus}"

    print("GPUs to be used\t:")
    for i in range(max(n_gpus)):
        print(f"GPU{i}\t\t:", torch.cuda.get_device_name(i))

    return n_gpus


def compare(exps, fig_name):
    _, ax = plt.subplots(ncols=3, figsize=(9, len(exps) / 2))
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
    file_name = fig_name.replace(" ", "-").replace("/", "-").replace("|", "-")
    plt.savefig(f"{file_name}.png", format="png", dpi=200, bbox_inches="tight")
    print(f"Result saved to {file_name}.png")
    plt.show()


def get_pkg_info(lib_name):
    """Get the information of the given package."""
    # Check availability
    try:
        mod = importlib.import_module(lib_name)
    except ImportError:
        return ("N/A", "N/A")

    # Fetch version.
    if hasattr(mod, "__version__"):
        version = mod.__version__
    else:
        try:
            version = pkg_resources.get_distribution(lib_name).version
        except Exception:
            version = "N/A"

    # If local repo is available, fetching more.
    def run_git_cmd(cmd):
        try:
            root_dir = os.path.dirname(os.path.dirname(os.path.abspath(mod.__file__)))
            ret = (
                subprocess.check_output(
                    cmd, cwd=root_dir, stderr=open(os.devnull, "wb")
                )
                .decode("utf-8")
                .strip()
            )
        except Exception:  # pylint: disable=broad-except
            ret = "N/A"
        return ret

    # Get the current branch
    branch_name = run_git_cmd(["git", "rev-parse", "--abbrev-ref", "HEAD"])
    if branch_name == "N/A":
        return (version, "N/A")

    # Get the remote name of the branch
    remote_name = run_git_cmd(
        ["git", "config", "--get", f"branch.{branch_name}.remote"]
    )
    if remote_name == "N/A":
        return (version, "N/A")

    # Get the remote URL of the branch
    remote_url = run_git_cmd(["git", "config", "--get", f"remote.{remote_name}.url"])

    # Get the commit hash
    commit_hash = run_git_cmd(["git", "rev-parse", "--short", "HEAD"])

    if remote_url.startswith("git@"):
        remote_url = remote_url.replace(":", "/")
        remote_url = remote_url.replace("git@", "https://")
    commit_url = f"{remote_url.replace('.git', '')}/commit/{commit_hash}"

    return (version, commit_url)


def list_envs(append_to=None):
    from tabulate import tabulate

    LIBS = [
        "torch",
        "epoi",
        "transformers",
        "xformers",
        "megatron",
        "deepspeed",
        "triton",
    ]
    data = OrderedDict()

    print("===== Environment =====\n")
    data["GPU"] = torch.cuda.get_device_name(0)
    data["CUDA"] = torch.version.cuda

    print(
        tabulate(
            data.items(),
            headers=["Env", "Value"],
            stralign="center",
            numalign="center",
        )
    )
    if append_to:
        with open(append_to, "a") as filep:
            filep.write("Env\tValue\n")
            for key, val in data.items():
                filep.write(f"{key}\t{val}\n")

    # Self
    data = OrderedDict()
    data["self"] = get_pkg_info("slapo")

    # Other libs
    for lib in LIBS:
        data[lib] = get_pkg_info(lib)

    print(
        tabulate(
            [(k, v[0], v[1]) for k, v in data.items()],
            headers=["Package", "Version", "Commit URL"],
            stralign="center",
            numalign="center",
        )
    )
    if append_to:
        with open(append_to, "a") as filep:
            filep.write("Package\tVersion\tCommitURL\n")
            for key, val in data.items():
                filep.write(f"{key}\t{val[0]}\t{val[1]}\n")
    print("===== Environment =====\n")


def main():
    args = parse_args()
    if args.impl == "env":
        list_envs(args.append_to)
        return

    framework = "deepspeed" if "deepspeed" in args.impl else "megatron"
    if args.impl == "megatron":
        impl_name = "Megatron"
    elif args.impl == "deepspeed":
        impl_name = "DeepSpeed"
    elif args.impl == "torchscript":
        impl_name = "TorchScript"
    elif args.impl == "eager":
        impl_name = "PyTorch Eager"
    elif args.impl == "slapo-megatron":
        impl_name = "Slapo-Megatron"
    elif args.impl == "slapo-deepspeed":
        impl_name = "Slapo-DeepSpeed"
    else:
        raise RuntimeError(f"Unrecognized implementation {impl_name}")

    title = f"{impl_name} {args.model}"
    memo = ""

    n_gpus = parse_gpus(args.gpus)
    batch_size_exp = args.batch_sizes.replace('"', "")
    get_batch_size = eval(f"lambda n: {batch_size_exp}", {"math": math})

    # Deal with configurations.
    kwargs = {"env": []}
    if framework == "megatron":
        runner = run_megatron
        kwargs, memo = parse_megatron_kwargs(args, kwargs, memo)
    else:
        runner = run_deepspeed
        kwargs, memo = parse_deepspeed_kwargs(args, kwargs, memo)

    kwargs["env"].append(f"IMPL={args.impl}")

    results = []
    for n_gpu in n_gpus:
        gpus = ",".join([str(e) for e in range(n_gpu)])
        batch_size = get_batch_size(n_gpu)
        results.append(
            runner(
                Exp(
                    f"BS{batch_size} ({n_gpu} GPU)",
                    args.model,
                    batch_size,
                    args.seq_len,
                    args.seq_len_dec,
                    impl=args.impl,
                    grad_ckpt=args.gradient_checkpoint,
                    fp16=args.dtype == "fp16",
                    gpus=gpus,
                    tensor_para=n_gpu,
                    kwargs=kwargs,
                ),
                args,
            )
        )
        if results[-1].error_code == 2 and args.error_stop:
            print("Stop benchmarking due to error")
            break

    if args.draw_fig:
        compare(results, f"{title}{memo}")


if __name__ == "__main__":
    main()
