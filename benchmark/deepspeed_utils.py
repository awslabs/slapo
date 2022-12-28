# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import re


def add_deepspeed_parser(common_parser, subprasers):
    subprasers.add_parser(
        "deepspeed",
        parents=[common_parser],
        help="DeepSpeed",
    )


def parse_deepspeed_kwargs(args, kwargs, memo):
    return kwargs, memo


def identify_model_key(exp):
    for model_key in ["bert", "gpt", "albert", "t5", "opt", "roberta"]:
        short_name = exp.model.split("/")[-1].split("-")[0]
        if model_key == short_name:
            return model_key
    raise ValueError(f"Unknown model key for {exp.model}")


def deepspeed_log(exp, log_filename):
    with open(log_filename) as f:
        text = f.read()
    # Find the last number after the key, returns 0 if not exists
    def query(key, last_only=True):
        values = re.findall(key + "=+([\d\.]+)", text)
        if not values:
            return None
        if last_only:
            return float(values[-1])
        return [float(v) for v in values]

    if "CUDA out of memory" in text:
        print("Out of GPU memory, try a smaller batch size")
        exp.error_code = 1
        return exp

    samples_per_sec = query("SamplesPerSec", last_only=False)
    if not samples_per_sec:
        print(f'Failed. Check "{log_filename}" to find error')
        exp.error_code = 2
        return exp

    # 1. Every 10 steps, DeepSpeed reports the average samples/sec from beginning.
    # 2. We remove the first value (of the first 10 steps) as the warmup.
    n_records = len(samples_per_sec)
    avg_samples_per_sec = (
        samples_per_sec[-1] * 10 * n_records - samples_per_sec[0] * 10
    ) / (10 * (n_records - 1))

    exp.param_per_gpu = 0  # DeepSpeed doesn't report this.
    exp.samples_per_sec = avg_samples_per_sec
    exp.gpu_mem = query("MaxMemAllocated")
    exp.error_code = 0
    return exp


def run_deepspeed(exp, args):
    import slapo

    model_key = identify_model_key(exp)
    path = slapo.__path__[0]
    script_file = f"{path}/../examples/{model_key}/deepspeed_hf.py"

    cmd = f"""CUDA_VISIBLE_DEVICES={exp.gpus} deepspeed {script_file} \
--model_name {exp.model} --seq_len {exp.seq_len} \
--disable_pipeline --batch_size {exp.batch_size} \
--iter_nums {exp.steps}"""
    if model_key == "t5":
        cmd += f" --dec_seq_len {exp.seq_len_dec}"
    if exp.grad_ckpt:
        cmd += f" --checkpoint {exp.grad_ckpt}"
    if "slapo" not in args.impl:
        cmd += " --disable_schedule"
    # if exp.fp16:
    #     cmd += " --fp16"

    if exp.kwargs is not None:
        if "env" in exp.kwargs and exp.kwargs["env"]:
            cmd = f"{' '.join(exp.kwargs['env'])} {cmd}"

    cmd += " > log.txt 2>&1"
    print(cmd)
    os.system(cmd)
    ret = deepspeed_log(exp, "log.txt")
    if ret.error_code != 0:
        ret.samples_per_sec = 0
        ret.gpu_mem = 0
    ret.print_results(append_to=args.append_to)
    return ret
