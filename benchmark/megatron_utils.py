# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import re

def add_megatron_parser(common_parser, subprasers):
    mt_parser = subprasers.add_parser(
        "megatron",
        parents=[common_parser],
        help="Megatron model",
    )
    mt_parser.add_argument(
        "--disable-fuse-kernels",
        action="store_true",
        help="Disable fusion kernels in Megatron models.",
    )

def parse_megatron_kwargs(args, kwargs, memo):
    if hasattr(args, "disable_fuse_kernels") and args.disable_fuse_kernels:
        kwargs["flags"] = [
            "--no-bias-gelu-fusion",
            "--no-bias-dropout-fusion",
            "--no-persist-layer-norm",
            "--no-masked-softmax-fusion",
        ]
        memo += "|no_fuse"

    if args.gradient_checkpoint != "0":
        memo += f"|grad_ckpt {args.gradient_checkpoint}"
        kwargs["env"].append(f"ckpt_ratio={args.gradient_checkpoint}")

    return kwargs, memo

def megatron_bert_cmd(exp):
    if exp.impl == "megatron":
        import megatron

        path = megatron.__path__[0]
        script_file = f"{path}/../pretrain_bert.py"
    else:
        import slapo

        path = slapo.__path__[0]
        script_file = f"{path}/../examples/bert/megatron_hf.py"

    return (
        script_file,
        [
            f"--seq-length {exp.seq_len}",
            f"--max-position-embeddings {exp.seq_len}",
            "--data-path bert-sample_text_sentence",
            "--vocab-file bert-large-uncased-vocab.txt",
        ],
    )


def megatron_albert_cmd(exp):
    if exp.impl == "megatron":
        raise NotImplementedError
    else:
        import slapo

        path = slapo.__path__[0]
        script_file = f"{path}/../examples/albert/megatron_hf.py"

    return (
        script_file,
        [
            f"--seq-length {exp.seq_len}",
            f"--max-position-embeddings {exp.seq_len}",
            "--data-path bert-sample_text_sentence",
            "--vocab-file bert-large-uncased-vocab.txt",
        ],
    )


def megatron_roberta_cmd(exp):
    if exp.impl == "megatron":
        raise NotImplementedError
    else:
        import slapo

        path = slapo.__path__[0]
        script_file = f"{path}/../examples/roberta/megatron_hf.py"

    return (
        script_file,
        [
            f"--seq-length {exp.seq_len}",
            f"--max-position-embeddings {exp.seq_len}",
            "--data-path bert-sample_text_sentence",
            "--vocab-file bert-large-uncased-vocab.txt",
        ],
    )


def megatron_gpt_cmd(exp):
    if exp.impl == "megatron":
        import megatron

        path = megatron.__path__[0]
        script_file = f"{path}/../pretrain_gpt.py"
    else:
        import slapo

        path = slapo.__path__[0]
        script_file = f"{path}/../examples/gpt/megatron_hf.py"

    return (
        script_file,
        [
            f"--seq-length {exp.seq_len}",
            f"--max-position-embeddings {exp.seq_len}",
            "--data-path gpt2-sample_text_document",
            "--vocab-file gpt2-vocab.json",
            "--merge-file gpt2-merges.txt",
        ],
    )


def megatron_opt_cmd(exp):
    if exp.impl == "megatron":
        raise NotImplementedError("Megatron does not support OPT")
    else:
        import slapo

        path = slapo.__path__[0]
        script_file = f"{path}/../examples/opt/megatron_hf.py"

    return (
        script_file,
        [
            f"--seq-length {exp.seq_len}",
            f"--max-position-embeddings {exp.seq_len}",
            "--data-path gpt2-sample_text_document",
            "--vocab-file gpt2-vocab.json",
            "--merge-file gpt2-merges.txt",
        ],
    )


def megatron_t5_cmd(exp):
    if exp.impl == "megatron":
        import megatron

        path = megatron.__path__[0]
        script_file = f"{path}/../pretrain_t5.py"
    else:
        import slapo

        path = slapo.__path__[0]
        script_file = f"{path}/../examples/t5/megatron_hf.py"

    assert hasattr(exp, "d_kv") and hasattr(exp, "d_ff")
    return (
        script_file,
        [
            f"--encoder-seq-length {exp.seq_len}",
            f"--decoder-seq-length {exp.seq_len_dec}",
            f"--max-position-embeddings {exp.seq_len}",
            f"--kv-channels {exp.d_kv}",
            f"--ffn-hidden-size {exp.d_ff}",
            "--data-path bert-sample_text_sentence",
            "--vocab-file bert-large-uncased-vocab.txt",
            "--vocab-extra-ids 100",
        ],
    )


MEGATRON_COMMAND_BY_MODEL = {
    "bert": megatron_bert_cmd,
    "albert": megatron_albert_cmd,
    "gpt": megatron_gpt_cmd,
    "t5": megatron_t5_cmd,
    "roberta": megatron_roberta_cmd,
    "opt": megatron_opt_cmd,
}


def megatron_log(exp, log_filename):
    with open(log_filename) as f:
        text = f.read()
    # Find the last number after the key, returns 0 if not exists
    def query(key, last_only=True):
        values = re.findall(key + ": +([\d\.]+)", text)
        if not values:
            return None
        if last_only:
            return float(values[-1])
        return [float(v) for v in values]

    if "CUDA out of memory" in text:
        print("Out of GPU memory, try a smaller batch size")
        exp.error_code = 1
        return exp

    iter_times = query("elapsed time per iteration \(ms\)", last_only=False)
    if not iter_times:
        print(f'Failed. Check "{log_filename}" to find error')
        exp.error_code = 2
        return exp

    # 1. Every 5 steps, Megatron reports the average iteration time of the past 5 steps.
    # 2. We remove the first value (of the first 5 steps) as the warmup.
    avg_time = lambda times: (sum(times[1:]) * 5) / (exp.steps - 5)

    iter_time = avg_time(iter_times)
    forward_compute_time = avg_time(query("forward-compute", last_only=False))
    backward_compute_time = avg_time(query("backward-compute", last_only=False))
    backward_param_all_reduce_time = avg_time(
        query("backward-params-all-reduce", last_only=False)
    )
    optimizer_time = avg_time(query("optimizer", last_only=False))

    param_per_gpu = query(
        "parameters on \(tensor, pipeline\) model parallel rank \(0, 0\)"
    )
    exp.param_per_gpu = param_per_gpu
    exp.samples_per_sec = query("global batch size") / iter_time * 1e3
    exp.gpu_mem = query("max allocated") / 1e3
    print(f"per GPU params\t\t: {param_per_gpu / 1e6:.2f}M")
    print(
        f"Breakdown(ms)\t\t: total {iter_time:.2f}, "
        f"forward {forward_compute_time:.2f}, "
        f"backward {backward_compute_time:.2f}, "
        f"backward-params-all-reduce {backward_param_all_reduce_time:.2f}, "
        f"optimizer {optimizer_time:.2f}"
    )
    exp.error_code = 0
    return exp


def run_megatron(exp, args):
    for model_key, gen in MEGATRON_COMMAND_BY_MODEL.items():
        short_name = exp.model.split("/")[-1].split("-")[0]
        if model_key == short_name:
            script_file, data_args = gen(exp)
            break
    else:
        raise ValueError(f"Unsupported model {exp.model}")

    cmd = f"""MODEL_NAME={exp.model} {exp.launcher} {script_file} \
--num-layers {exp.num_layers} --hidden-size {exp.hidden_size} \
--num-attention-heads {exp.num_heads} \
--tensor-model-parallel-size {exp.tensor_para} \
--micro-batch-size {exp.batch_size} \
--train-iters {exp.steps} {' '.join(data_args)} \
--data-impl mmap --lr 0.00015 --log-interval 5 --eval-iters 1"""
    if exp.grad_ckpt and exp.grad_ckpt != "0":
        if "slapo" in args.impl:
            # Gradient checkpoint ratio for HF w. Slapo is passed
            # via environment variable.
            grad_ckpt = "full"
        else:
            if exp.grad_ckpt == "full":
                cmd += f" --recompute-method uniform"
            grad_ckpt = exp.grad_ckpt
        cmd += f" --recompute-granularity {grad_ckpt}"
    if exp.bf16:
        cmd += " --bf16"
    if exp.fp16:
        cmd += " --fp16"

    if exp.kwargs is not None:
        if "flags" in exp.kwargs:
            cmd += " " + " ".join(exp.kwargs["flags"])
        if "env" in exp.kwargs and exp.kwargs["env"]:
            cmd = f"{' '.join(exp.kwargs['env'])} {cmd}"

    cmd += " > log.txt 2>&1"
    print(cmd)
    os.system(cmd)
    ret = megatron_log(exp, "log.txt")
    if ret.error_code != 0:
        ret.samples_per_sec = 0
        ret.gpu_mem = 0
    ret.print_results(append_to=args.append_to)
    return ret
