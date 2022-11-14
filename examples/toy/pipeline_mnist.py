from data.mnist_loader import load_data_wrapper

import os
import json
import torch
import argparse
import torch.nn as nn
from deepspeed.pipe import PipelineModule
from deepspeed.utils.logging import LoggerFactory
import deepspeed

config_dict = {
    "train_batch_size": 64,
    "train_micro_batch_size_per_gpu": 4,
    "steps_per_print": 1,
    "zero_allow_untested_optimizer": True,
    "optimizer": {
        "type": "Adam",
        "params": {
            "lr": 0.001,
            "weight_decay": 0.01,
            "bias_correction": True,
            "eps": 1e-6
        }
    },
    "gradient_clipping": 0.0,
    "fp16": {
        "enabled": True,
        "initial_scale_power": 10
    },
    "zero_optimization": {
        "stage": 0,
        "overlap_comm": True,
        "contiguous_gradients": False,
        # "reduce_bucket_size": 20,
        "stage3_param_persistence_threshold": 0,
    }
}


class MNISTDataset(torch.utils.data.Dataset):
    def __init__(self, size, dtype) -> None:
        super().__init__()
        train_data, _, _ = load_data_wrapper()
        self.data = train_data[:size] 
        self.dtype = dtype

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        entry = self.data[idx]
        # print(f'shape, {entry[0].shape}')
        X = torch.tensor(entry[0].reshape(-1), dtype=self.dtype)
        y = torch.tensor(entry[1].reshape(-1), dtype=self.dtype)
        return X, y


def get_toy_model(nlayers, hidden_size, num_stages=2, input_size=784, nclasses=10, seed=123, checkpoint_interval=1):
    """"""
    layers = []

    for i in range(nlayers):
        if i == 0:
            layers.append(nn.Linear(input_size, hidden_size))
        elif i == (nlayers - 1):
            layers.append(nn.Linear(hidden_size, nclasses))
        else:
            layers.append(nn.Linear(hidden_size, hidden_size))

    module = PipelineModule(
        layers=layers,
        num_stages=num_stages,
        loss_fn=torch.nn.CrossEntropyLoss(),
        base_seed=seed,
        partition_method='uniform',
        activation_checkpoint_interval=checkpoint_interval
    )

    return module

def create_config_from_dict(tmpdir, config_dict):
    config_path = os.path.join(tmpdir, 'temp_config.json')
    with open(config_path, 'w') as fd:
        json.dump(config_dict, fd)
    return config_path

def get_args(config_dict):
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument('--zero', type=int, default=0)
    parser.add_argument('--nlayers', type=int, default=2)
    parser.add_argument('--hidden-size', type=int, default=100)
    parser.add_argument('--num-stages', type=int, default=2)
    parser.add_argument('--tmp', default='/tmp', help="temporary directory to save intermediate data")
    parser.add_argument('--use-ds-context-manager', default=False, action='store_true')
    parser.add_argument('--nepochs', type=int, default=1)
    parser.add_argument('--nsteps-per-epoch', type=int, default=10)
    parser.add_argument('--nsamples', type=int, default=2000)
    args = parser.parse_args()  #args=''

    config_dict["zero_optimization"]["stage"] = args.zero
    print('config_dict["zero_optimization"]', config_dict["zero_optimization"])
    config_path = create_config_from_dict(args.tmp, config_dict)

    args.deepspeed_config = config_path
    return args

def main():
    """"""
    torch.random.manual_seed(123)
    global config_dict
    args = get_args(config_dict)
    logger = LoggerFactory.create_logger('toy_pipeline_model')
    deepspeed.init_distributed(dist_backend="nccl")

    module = get_toy_model(nlayers=args.nlayers, 
                            hidden_size=args.hidden_size,
                            num_stages=args.num_stages,)


    engine, optimizer, _, _ = deepspeed.initialize(
        args=args,
        model=module,
        model_parameters=[p for p in module.parameters() if p.requires_grad],
    )
    args.fp16 = engine.fp16_enabled()
    data_dtype = torch.half if args.fp16 else torch.float
    mnist_dataset = MNISTDataset(args.nsamples, data_dtype)
    micro_bs = engine.train_micro_batch_size_per_gpu()
    mnist_loader = torch.utils.data.DataLoader(mnist_dataset, batch_size=micro_bs)

    for eidx in range(args.nepochs):
        data_iter = iter(mnist_loader)
        for sidx in range(args.nsteps_per_epoch):
            loss = engine.train_batch(data_iter)
            logger.info(f'epoch {eidx}, step {sidx} loss: {loss}')

if __name__ == "__main__":
    main()