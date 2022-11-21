# Examples

This folder includes some examples of using model schedule. 
Before running examples, you need to install the following Python packages:

- xformers: https://github.com/facebookresearch/xformers
- epoi: https://github.com/comaniac/epoi

It is recommended to build both packages from source (clone the repo and run
`pip install -e ".[dev]"`).

All examples are tested with the following commit hashs. If you encounter any failure,
please try to fix the commit hash first to see if that resolves the issue.

- xformers: c101579
- epoi: 1980c3a

Meanwhile, for each example model, we apply a series of schedules as introdcued as follows.

1. Replace attention layers with flash attention.
2. Fused QKV.
3. Shard word embedding, QKV and MLP for multi-GPUs.
4. Add checkpoints.
