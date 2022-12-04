from torch.nn import Module
from torchvision.models.resnet import ResNet, Bottleneck


def model_schedule(model):
    import ms
    import inspect
    import torch.distributed as dist

    input_names = ["x"]
    sig = inspect.signature(model.forward)
    concrete_args = {
        p.name: p.default for p in sig.parameters.values() if p.name not in input_names
    }

    try:
        world_size = dist.get_world_size()
        rank = dist.get_rank()
    except:
        print("torch.distributed not initialized. Use world size 1 and rank 0")
        world_size = 1
        rank = 0

    sch = ms.create_schedule(
        model,
        world_size=world_size,
        rank=rank,
        tracer="pytorch",
        concrete_args=concrete_args,
    )

    model, _ = ms.build(sch)
    model.cuda()
    return model


def get_model(width_per_group, layers) -> Module:
    kwargs = {"width_per_group": width_per_group}
    model = model_schedule(ResNet(Bottleneck, layers, **kwargs))
    return model
