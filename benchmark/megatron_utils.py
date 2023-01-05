# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os
from slapo.model_dialect import get_dialect_cls


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


def megatron_wideresnet_cmd(exp):
    if exp.impl == "megatron":
        raise NotImplementedError("Megatron does not support WideResNet")
    else:
        import slapo

        path = slapo.__path__[0]
        script_file = f"{path}/../examples/wideresnet/megatron_hf.py"

    return (
        script_file,
        [  # Fake configs.
            f"--seq-length 1",
            f"--max-position-embeddings 10",
        ],
    )


MEGATRON_COMMAND_BY_MODEL = {
    "bert": megatron_bert_cmd,
    "albert": megatron_albert_cmd,
    "gpt": megatron_gpt_cmd,
    "t5": megatron_t5_cmd,
    "roberta": megatron_roberta_cmd,
    "opt": megatron_opt_cmd,
    "wideresnet": megatron_wideresnet_cmd,
}


def megatron_log(exp, log_filename):
    parser = get_dialect_cls("log_parser", "megatron")
    param_per_gpu, samples_per_sec, gpu_mem, error_code = parser.parse_log(log_filename)
    if error_code != 0:
        exp.error_code = error_code
        return exp
    else:
        exp.param_per_gpu = param_per_gpu  # DeepSpeed doesn't report this.
        exp.samples_per_sec = samples_per_sec
        exp.gpu_mem = gpu_mem
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

    flag_debug = "DEBUG=1" if args.debug else ""
    cmd = f"""{flag_debug} MODEL_NAME={exp.model} {exp.launcher} {script_file} \
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
