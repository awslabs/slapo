# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""
End-to-end tests
"""

import os
import pytest

from slapo.model_dialect import get_dialect_cls


def parse_log(impl, log_file):
    with open(log_file, "r", encoding="utf-8") as f:
        text = f.read()

    if impl in {"slapo-megatron", "megatron"}:
        parser = get_dialect_cls("log_parser", "megatron")
        _, samples_per_sec, _, error_code = parser.parse_log(log_file)
    elif impl in {"slapo-deepspeed", "deepspeed"}:
        parser = get_dialect_cls("log_parser", "deepspeed")
        _, samples_per_sec, _, error_code = parser.parse_log(log_file)
    else:
        raise RuntimeError("Please provide correct `impl`")
    return (error_code, samples_per_sec, text)


# fmt: off
@pytest.mark.parametrize("model,impl,n_gpu,batch_size,seq_len,ckpt_ratio", [
    ("wideresnet-250M", "slapo-megatron", "1", "48", "512", "0.34"),
    ("wideresnet-250M", "slapo-deepspeed", "4", "256", "512", "0.67"),
    ("bert-large-uncased", "slapo-megatron", "2", "10", "512", "0"),
    ("bert-large-uncased", "slapo-deepspeed", "2", "28", "512", "0"),
    ("EleutherAI/gpt-neo-125M", "slapo-megatron", "2", "1", "512", "1.0"),
    ("t5-base", "slapo-megatron", "4", "8", "1024", "0.67"),
])
# fmt: on
def test_end2end(model, impl, n_gpu, batch_size, seq_len, ckpt_ratio):
    print(f"Running {impl} on {model} with {n_gpu} GPU", flush=True)
    if impl == "deepspeed":
        ckpt_ratio = "1.0"
    elif impl == "megatron":
        ckpt_ratio = "full"
    cmd = f"python3 benchmark/bench_single_node.py {impl}"
    cmd += f" --model {model} --gpus {n_gpu} --seq-len {seq_len}"
    if "t5" in model:
        cmd += " --seq-len-dec 512"
    cmd += f" --batch-size {batch_size}"
    cmd += f" --gradient-checkpoint {ckpt_ratio}"
    cmd += " > run_script.log 2>&1"
    print(cmd, flush=True)
    os.system(cmd)
    print("\n", flush=True)
    error_code, samples_per_sec, text = parse_log(impl, "log.txt")
    print(f"\tThroughput: {samples_per_sec:.2f}")
    if error_code == 1:
        print("oom")
        print(text)
    elif error_code == 2:
        print("fail")
        print(text)
    assert error_code == 0


if __name__ == "__main__":
    pytest.main([__file__])
