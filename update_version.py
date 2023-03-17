# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# Modification: Alpa. See https://github.com/alpa-projects/alpa/blob/main/update_version.py

"""
This is the global script that set the version information.
This script runs and update all the locations that related to versions
List of affected files:
- root/slapo/version.py
"""
import os
import re
import argparse
import logging

from setup import get_version, get_version_in_version_py


PROJ_ROOT = os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))
CURR_VERSON = get_version_in_version_py(PROJ_ROOT)


# Implementations
def update(file_name, pattern, repl, dry_run=False):
    """Update the version specified in the given file using the given pattern."""
    update = []
    hit_counter = 0
    need_update = False
    with open(file_name) as file:
        for line in file:
            result = re.findall(pattern, line)
            if result:
                assert len(result) == 1
                hit_counter += 1
                if result[0] != repl:
                    line = re.sub(pattern, repl, line)
                    need_update = True
                    print("%s: %s -> %s" % (file_name, result[0], repl))
                else:
                    print("%s: version is already %s" % (file_name, repl))

            update.append(line)
    if hit_counter != 1:
        raise RuntimeError("Cannot find version in %s" % file_name)

    if need_update and not dry_run:
        with open(file_name, "w") as output_file:
            for line in update:
                output_file.write(line)


def sync_version(pub_ver, local_ver, dry_run):
    """Synchronize version."""

    # Sanity check.
    target_ver = pub_ver
    if pub_ver.find("dev") != -1:
        target_ver = pub_ver[: pub_ver.find(".dev")]

    if not CURR_VERSON.startswith(target_ver):
        raise RuntimeError(
            f"The existing version ({CURR_VERSON}) is not the dev version of "
            f"the target version ({target_ver}). Please update the version.py "
            f"manually to {target_ver}.dev0, commit and re-tag."
        )

    # python uses the PEP-440: local version
    update(
        os.path.join(PROJ_ROOT, "slapo", "version.py"),
        r"(?<=__version__ = \")[.0-9a-z\+]+",
        local_ver,
        dry_run,
    )


def main():
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Detect and synchronize version.")
    parser.add_argument(
        "--print-version",
        action="store_true",
        help="Print version to the command line. No changes is applied to files.",
    )
    parser.add_argument(
        "--git-describe",
        action="store_true",
        help="Use git describe to generate development version.",
    )
    parser.add_argument("--dry-run", action="store_true")

    opt = parser.parse_args()
    pub_ver, local_ver = CURR_VERSON, CURR_VERSON
    if opt.git_describe:
        pub_ver, local_ver = get_version()

    if opt.print_version:
        print(local_ver)
    else:
        sync_version(pub_ver, local_ver, opt.dry_run)


if __name__ == "__main__":
    main()
