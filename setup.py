#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import setuptools

PROJ_NAME = "slapo"
this_dir = os.path.dirname(os.path.abspath(__file__))


if os.getenv("BUILD_VERSION"):
    version = os.getenv("BUILD_VERSION")
else:
    version_txt = os.path.join(this_dir, "version.txt")
    with open(version_txt) as filep:
        version = filep.readline().strip()


def write_version_file():
    version_path = os.path.join(this_dir, PROJ_NAME, "version.py")
    with open(version_path, "w") as f:
        f.write("# noqa: C801\n")
        f.write(f'__version__ = "{version}"\n')
        tag = os.getenv("GIT_TAG")
        if tag is not None:
            f.write(f'git_tag = "{tag}"\n')


def setup():
    write_version_file()
    setuptools.setup(
        name=PROJ_NAME,
        description="Slapo: A Scahedule LAnguage for Progressive Optimization.",
        version=version,
        setup_requires=[],
        install_requires=[],
        packages=setuptools.find_packages(),
        url="TBA",
        python_requires=">=3.7",
        classifiers=[
            "Programming Language :: Python :: 3.7",
            "Programming Language :: Python :: 3.8",
            "Programming Language :: Python :: 3.9",
            "Topic :: Scientific/Engineering :: Artificial Intelligence",
            "Operating System :: OS Independent",
        ],
        zip_safe=True,
    )


if __name__ == "__main__":
    setup()
