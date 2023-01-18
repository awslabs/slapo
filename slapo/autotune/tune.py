# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The module to tune schedules."""
import argparse
import copy
import importlib
import json
import os
import pathlib
import re
import sys
import time

from slapo.logger import get_logger
from slapo.model_dialect import get_dialect_cls


logger = get_logger()


def must_fix(func):
    """Decorator to mark a function in Symbol to ensure its value has been fixed."""

    def wrapper(self, *args, **kwargs):
        return self.is_fixed() and func(self, *args, **kwargs)

    return wrapper


class Symbol:
    """A tunable symbol."""

    def __init__(self, name, vals):
        self.name = name
        self.vals = vals
        self.fixed_idx = -1

    @property
    def value(self):
        if not self.is_fixed():
            raise ValueError(f"The value of symbol {self.name} has not been fixed")
        return self.vals[self.fixed_idx]

    @must_fix
    def __gt__(self, other):
        return self.value > other

    @must_fix
    def __ge__(self, other):
        return self.value >= other

    @must_fix
    def __lt__(self, other):
        return self.value < other

    @must_fix
    def __le__(self, other):
        return self.value <= other

    def __len__(self):
        return len(self.vals)

    def add(self, val):
        """Add a value to the symbol. If the value is already in the symbol, do nothing."""
        if val not in self.vals:
            self.vals.append(val)

    def fix_at(self, idx):
        """Fix the value of this symbol at the given index."""
        if idx < len(self.vals):
            self.fixed_idx = idx
            return
        raise ValueError(
            f"Cannot fix {self.name} with {len(self.vals)} values at idx {idx}"
        )

    def is_fixed(self):
        """Check if the value of this symbol has been fixed."""
        return self.fixed_idx != -1


class Space:
    """The tuning space."""

    def __init__(self):
        self.space = {}
        self.idx_to_name = []
        self.fixed_idx = -1

    def create_symbol(self, name, vals):
        """Create a symbol in the space. If the symbol already exists:
        1) If the symbol is fixed, do nothing;
        2) Otherwise re-create the symbol, because its candidate values may change
           due to other fixed symbols.
        """
        if name in self.space:
            # Ignore if the value has been fixed; otherwise re-generate the symbol.
            if self.space[name].is_fixed():
                return self.space[name]

        # Create a new symbol.
        if name not in self.space:
            self.idx_to_name.append(name)

        self.space[name] = Symbol(name, vals)
        return self.space[name]

    def next(self):
        """Get the next symbol to fix."""
        if self.fixed_idx + 1 < len(self.space):
            self.fixed_idx += 1
            return self.space[self.idx_to_name[self.fixed_idx]]
        return None

    def reset(self):
        """Reset the space to the initial state."""
        self.fixed_idx = -1
        for symbol in self.space.values():
            symbol.fixed_idx = -1

    def to_dict(self):
        """Convert the space to a dict. Note that all symbols must be fixed
        before calling this function.
        """
        cfg = {}
        for symbol in self.space.values():
            cfg[symbol.name] = symbol.value
        return cfg

    def clone(self):
        """Clone the space."""
        return copy.deepcopy(self)

    @staticmethod
    def cfg_dict_to_str(cfg_dict):
        """Convert a config dict to a string for logging and debugging."""
        ret = "("
        for idx, (k, v) in enumerate(cfg_dict.items()):
            is_last = idx == len(cfg_dict) - 1
            last_ch = ")" if is_last else ", "
            ret += f"{k}: {v}{last_ch}"
        return ret

    def log_space(self, training_script_args, update_space_fn):
        """Print the tuning space for logging."""

        def _run(space, count=0):
            symbol = space.next()
            if symbol is not None:
                for idx in range(len(symbol.vals)):
                    symbol.fix_at(idx)
                    space = update_space_fn(training_script_args, space)
                    count = _run(space.clone(), count)
                return count

            logger.info(f"\t{self.cfg_dict_to_str(space.to_dict())}")
            return count + 1

        logger.info("Enumerating the search space:")
        count = _run(self.clone())
        logger.info(f"Space size: {count}")

    def __repr__(self):
        ret = "Space(\n"
        for idx, symbol in enumerate(self.space.values()):
            is_last = idx == len(self.space) - 1
            val = f"{symbol.value} (fixed)" if symbol.is_fixed() else str(symbol.vals)
            last_ch = ")" if is_last else ",\n"
            ret += f"\t{symbol.name}: {val}{last_ch}"
        return ret


class Database:
    """A simple database to store the results of tuning in Dict."""

    def __init__(self, db_file_name=None):
        self.db_file_name = db_file_name
        self.db = {}
        if self.db_file_name:
            logger.info(f"Tuning records will be saved to {self.db_file_name}")

    def load(self):
        """Load the database from the file."""
        if self.db_file_name and os.path.exists(self.db_file_name):
            with open(self.db_file_name, "r") as filep:
                self.db = json.load(filep)
            logger.info(
                f"Loaded {len(self.db)} tuning records from {self.db_file_name}"
            )

    def commit(self, key, data):
        """Commit the data to the database and update the DB file."""
        self.db[key] = data
        if self.db_file_name:
            with open(self.db_file_name, "w") as filep:
                json.dump(self.db, filep, indent=2)


def parse_args():
    parser = argparse.ArgumentParser("Auto-Tuning for Model Schedule")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="The config file including tuning space definition, etc",
    )
    parser.add_argument(
        "--db",
        type=str,
        help="The file path to store tuning records in JSON format",
    )
    parser.add_argument(
        "--error-stop",
        type=str,
        default="none",
        choices=["none", "symbol", "all"],
        help="When error occurs, either stop tuning the current symbol or all symbols",
    )
    parser.add_argument(
        "training_script",
        type=str,
        help="The full path to the training script. The defined tunable parameters in "
        "config file can be used in both the script and the arguments",
    )
    parser.add_argument(
        "training_script_args",
        nargs=argparse.REMAINDER,
        help="The arguments to the training script",
    )
    return parser.parse_args()


def convert_nargs_to_dict(nargs):
    """Convert the arguments to a dict."""
    if not nargs:
        return {}

    def infer_type(val):
        try:
            val = float(val)
            if val // 1 == val:
                return int(val)
            return val
        except ValueError:
            return val

    def remove_leading_minus(name):
        idx = re.match("-*", name).end()
        return name[idx:]

    ret = {}

    ptr = 0
    while ptr < len(nargs):
        arg_name = nargs[ptr]
        ptr += 1
        if (
            ptr == len(nargs)
            or not arg_name.startswith(("-", "--"))
            or nargs[ptr].startswith(("-", "--"))
        ):
            # True/False flag or positional argument.
            ret[remove_leading_minus(arg_name)] = 1
        else:
            vals = []
            while ptr < len(nargs) and not nargs[ptr].startswith(("-", "--")):
                vals.append(infer_type(nargs[ptr]))
                ptr += 1
            ret[remove_leading_minus(arg_name)] = vals[0] if len(vals) == 1 else vals
    return ret


def run_training_script(args, tuneable_cfg):
    """Run the training script with the given config."""
    train_script_args = " ".join(args.training_script_args)

    # Replace tunable values in training script arguments.
    for key, val in tuneable_cfg.items():
        train_script_args = re.sub(key, f'"{str(val)}"', train_script_args)

    # Set all tunable parameters as environment variables.
    env = " ".join(f"{k}={v}" for k, v in tuneable_cfg.items())

    cmd = f"{env} python3 {args.training_script} {train_script_args}"
    cmd += " > run_script.log 2>&1"
    logger.info(f"\tRunning command: {cmd}")
    os.system(cmd)
    return "run_script.log"


def tune(args, get_bs_range, eval_fn):
    """Tune the given space with an evaluation function."""
    training_script_args = convert_nargs_to_dict(args.training_script_args)
    min_bs, max_bs, step = get_bs_range(training_script_args)
    bs_range = list(range(min_bs, max_bs + 1, step))
    ckpt_ratio_range = [1.0, 0.92, 0.84, 0.67, 0.5, 0.34, 0.25]
    early_stopping_patience = 0

    def is_valid(config):
        if "slapo-deepspeed" in training_script_args:
            # DeepSpeed uses data parallelism requiring the global batch size
            # can be divided by number of devices
            return config["batch_size"] % int(training_script_args["gpus"]) == 0
        return True

    def binary_search(data, cfg_dict, key, curr_best, lt=0, rt=None):
        nonlocal early_stopping_patience
        logger.info(f"Binary searching {key} without OOM")
        if rt is None:
            rt = len(data) - 1
        while lt <= rt:
            mid = (lt + rt) // 2
            cfg_dict[key] = data[mid]
            logger.info(f"- Evaluating {cfg_dict}")
            # early pruning
            if is_valid(cfg_dict):
                thrpt = eval_fn(cfg_dict)
            else:
                thrpt = 0.0
                logger.info(
                    f"Invalid configuration point {cfg_dict}, n_gpu={training_script_args['gpus']}"
                )
            time.sleep(0.5)
            logger.info(f"\tThroughput: {thrpt:.2f}")
            # TODO: threshold should be a larger value used for pruning
            # maybe provide an interface for the users
            if thrpt < 0.01:
                rt = mid - 1
            else:
                lt = mid + 1
            if thrpt > curr_best[1]:
                curr_best = (cfg_dict.copy(), thrpt)
                early_stopping_patience = 0
            else:
                early_stopping_patience += 1
            # set step 5 as the patience
            if early_stopping_patience >= 5:
                return mid, None, curr_best
            logger.info(
                f"\tCurrent best config: {curr_best[0]}, " f"thrpt: {curr_best[1]:.2f}"
            )
        if thrpt < 0.01:
            mid = mid - 1
        return mid, thrpt, curr_best

    def _run(min_bs, max_bs, step):
        if "megatron" in training_script_args:
            ckpt_ratio = "full"
        else:
            ckpt_ratio = 1.0
        cfg_dict = {"batch_size": max_bs, "ckpt_ratio": ckpt_ratio}
        # suppose the user given minimum bs is always executable
        logger.info(f"Evaluating inital config...")
        logger.info(f"- Evaluating {cfg_dict}")
        thrpt = eval_fn(cfg_dict)
        logger.info(f"\tThroughput: {thrpt:.2f}")
        curr_best = (cfg_dict.copy(), thrpt)
        if thrpt == 0:  # OOM
            mid, thrpt, curr_best = binary_search(
                bs_range, cfg_dict, "batch_size", curr_best
            )
            max_bs = bs_range[mid]
        else:
            mid = 0
        logger.info(f"Maximum batch size without OOM: {max_bs}")
        if (
            "slapo-megatron" in training_script_args
            or "slapo-deepspeed" in training_script_args
        ):
            for bs in reversed(list(range(min_bs, max_bs + 1, step))):
                cfg_dict["batch_size"] = bs
                mid, thrpt, curr_best = binary_search(
                    ckpt_ratio_range, cfg_dict, "ckpt_ratio", curr_best, lt=mid
                )
                if thrpt is None:  # early stopping
                    break
        return curr_best

    logger.info("Start tuning...")
    curr_best = _run(min_bs, max_bs, step)
    logger.info("Tuning done!")
    return curr_best[0]


def load_config(config_file):
    """Load required functions from the tuning config."""
    path = pathlib.Path(config_file).absolute()
    sys.path.append(str(path.parent))
    module = importlib.import_module(path.stem)
    if not hasattr(module, "get_bs_range"):
        raise ValueError("Missing 'get_bs' function in config file")
    return module.get_bs_range


def parse_log(args, log_file):
    with open(log_file) as f:
        text = f.read()

    if "slapo-megatron" in args or "megatron" in args:
        parser = get_dialect_cls("log_parser", "megatron")
        param_per_gpu, samples_per_sec, gpu_mem, error_code = parser.parse_log(log_file)
    elif "slapo-deepspeed" in args or "deepspeed" in args:
        parser = get_dialect_cls("log_parser", "deepspeed")
        param_per_gpu, samples_per_sec, gpu_mem, error_code = parser.parse_log(log_file)
    else:
        raise RuntimeError("Please provide correct `impl`")
    return (error_code, samples_per_sec, text)


def main():
    """Entry point."""
    args = parse_args()
    get_bs_range = load_config(args.config)
    db = Database(args.db)

    def eval_fn(cfg):
        log_file = run_training_script(args, cfg)
        error_code, thrpt, memo = parse_log(
            convert_nargs_to_dict(args.training_script_args), "log.txt"
        )
        with open(log_file, "r") as filep:
            log_file_ctx = filep.read()
        db.commit(
            Space.cfg_dict_to_str(cfg),
            {
                "error_code": error_code,
                "thrpt": thrpt,
                "log": log_file_ctx,
                "memo": memo,
            },
        )
        if args.error_stop == "all" and error_code != 0:
            raise ValueError("Stop tuning due to error. Check log for details")
        return thrpt if error_code == 0 else 0

    curr_best = tune(args, get_bs_range, eval_fn)
    logger.info(f"Best config: {curr_best}")


if __name__ == "__main__":
    main()
