# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""
End-to-end tests
"""

import pytest
import os

from slapo.model_dialect import get_dialect_cls


def parse_log(impl, log_file):
    with open(log_file) as f:
        text = f.read()

    if impl in ["slapo-megatron", "megatron"]:
        parser = get_dialect_cls("log_parser", "megatron")
        param_per_gpu, samples_per_sec, gpu_mem, error_code = parser.parse_log(log_file)
    elif impl in ["slapo-deepspeed", "deepspeed"]:
        parser = get_dialect_cls("log_parser", "deepspeed")
        param_per_gpu, samples_per_sec, gpu_mem, error_code = parser.parse_log(log_file)
    else:
        raise RuntimeError("Please provide correct `impl`")
    return (error_code, samples_per_sec, text)


# fmt: off
@pytest.mark.parametrize("model,impl,n_gpu,batch_size,ckpt_ratio", [
    ("wideresnet-250M", "slapo-megatron", "1", "48", "0.34"),
    ("wideresnet-250M", "slapo-deepspeed", "4", "256", "0.67"),
    ("bert-large-uncased", "slapo-megatron", "2", "20", "0"),
    ("bert-large-uncased", "slapo-deepspeed", "2", "28", "0"),
    ("EleutherAI/gpt-neo-1.3B", "slapo-megatron", "2", "2", "1.0"),
    ("EleutherAI/gpt-neo-1.3B", "slapo-deepspeed", "4", "8", "0.67"),
    ("t5-large", "slapo-megatron", "4", "24", "0.67"),
])
# fmt: on
def test_end2end(model, impl, n_gpu, batch_size, ckpt_ratio):
    if any(m in model for m in ["opt", "t5", "gpt"]):
        seq_len = 1024
    else:
        seq_len = 512
    print(f"Running {impl} on {model} with {n_gpu} GPU", flush=True)
    if impl == "deepspeed":
        ckpt_ratio = "1.0"
    elif impl == "megatron":
        ckpt_ratio = "full"
    cmd = f"cd benchmark && python3 bench_single_node.py {impl}"
    cmd += f" --model {model} --gpus {n_gpu} --seq-len {seq_len}"
    if model == "t5":
        cmd += " --seq-len-dec 512"
    cmd += f" --batch-size {batch_size}"
    cmd += f" --gradient-checkpoint {ckpt_ratio}"
    cmd += f" > run_script.log 2>&1"
    print(cmd, flush=True)
    os.system(cmd)
    print("\n", flush=True)
    error_code, samples_per_sec, text = parse_log(impl, "benchmark/log.txt")
    print(f"\tThroughput: {samples_per_sec:.2f}")
    assert error_code == 0


if __name__ == "__main__":
    pytest.main([__file__])
