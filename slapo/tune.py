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

    def print_space(self, training_script_args, update_space_fn):
        """Print the tuning space for logging."""

        def _run(space, count=0):
            symbol = space.next()
            if symbol is not None:
                for idx in range(len(symbol.vals)):
                    symbol.fix_at(idx)
                    space = update_space_fn(training_script_args, space)
                    count = _run(space.clone(), count)
                return count

            print(f"\t{self.cfg_dict_to_str(space.to_dict())}")
            return count + 1

        print("Enumerating the search space:")
        count = _run(self.clone())
        print(f"Space size: {count}")

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
            print(f"Tuning records will be saved to {self.db_file_name}")

    def load(self):
        """Load the database from the file."""
        if self.db_file_name and os.path.exists(self.db_file_name):
            with open(self.db_file_name, "r") as filep:
                self.db = json.load(filep)
            print(f"Loaded {len(self.db)} tuning records from {self.db_file_name}")

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
            ret[arg_name.replace("-", "")] = 1
        else:
            vals = []
            while ptr < len(nargs) and not nargs[ptr].startswith(("-", "--")):
                vals.append(infer_type(nargs[ptr]))
                ptr += 1
            ret[arg_name.replace("-", "")] = vals[0] if len(vals) == 1 else vals
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
    print(f"\tRunning command: {cmd}")
    os.system(cmd)
    return "run_script.log"


def tune(args, update_space_fn, eval_fn):
    """Tune the given space with an evaluation function."""
    training_script_args = convert_nargs_to_dict(args.training_script_args)
    space = update_space_fn(training_script_args, Space())
    space.print_space(training_script_args, update_space_fn)

    def _run(space, curr_best):
        symbol = space.next()
        if symbol is not None:
            for idx in range(len(symbol.vals)):
                symbol.fix_at(idx)
                space = update_space_fn(training_script_args, space)
                curr_best, has_error = _run(space.clone(), curr_best)
                if has_error and args.error_stop == "symbol":
                    print(f"\tStop tuning {symbol.name} due to error")
                    break
            return curr_best, False

        cfg_dict = space.to_dict()
        print(f"- Evaluating {cfg_dict}")
        thrpt = eval_fn(cfg_dict)
        print(f"\tThroughput: {thrpt:.2f}")
        if curr_best[0] is None or thrpt >= curr_best[1]:
            curr_best = (space.clone(), thrpt)
        print(
            f"\tCurrent best config: {Space.cfg_dict_to_str(curr_best[0].to_dict())}, "
            f"thrpt: {curr_best[1]:.2f}"
        )
        return curr_best, thrpt == 0

    print("Start tuning...")
    curr_best, _ = _run(space.clone(), (None, 0))
    print("Tuning done!")
    return curr_best[0]


def load_config(config_file):
    """Load required functions from the tuning config."""
    path = pathlib.Path(config_file).absolute()
    sys.path.append(str(path.parent))
    module = importlib.import_module(path.stem)
    if not hasattr(module, "update_space"):
        raise ValueError("Missing 'update_space' function in config file")
    if not hasattr(module, "parse_log"):
        raise ValueError("Missing 'parse_log' function in config file")
    return module.update_space, module.parse_log


def main():
    """Entry point."""
    args = parse_args()
    update_space_fn, parse_log_fn = load_config(args.config)
    db = Database(args.db)

    def eval_fn(cfg):
        log_file = run_training_script(args, cfg)
        error_code, thrpt, memo = parse_log_fn(log_file)
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

    best_cfg = tune(args, update_space_fn, eval_fn)
    print(f"Best config: {Space.cfg_dict_to_str(best_cfg.to_dict())}")


if __name__ == "__main__":
    main()
