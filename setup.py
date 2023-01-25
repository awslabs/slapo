# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import re
import setuptools

ROOT_DIR = os.path.dirname(__file__)


def get_version():
    with open(os.path.join(ROOT_DIR, "slapo", "version.py")) as fp:
        version_match = re.search(
            r"^__version__ = ['\"]([^'\"]*)['\"]", fp.read(), re.M
        )
        if version_match:
            return version_match.group(1)
    raise RuntimeError("Unable to find version string")


def setup():
    with open("README.md", encoding="utf-8") as fp:
        long_description = fp.read()

    setuptools.setup(
        name="slapo",
        description="Slapo: A Scahedule LAnguage for Progressive Optimization.",
        version=get_version(),
        author="Slapo Community",
        long_description=long_description,
        long_description_content_type="text/markdown",
        setup_requires=[],
        install_requires=[],
        packages=setuptools.find_packages(),
        url="https://github.com/awslabs/slapo",
        python_requires=">=3.7",
        classifiers=[
            "Programming Language :: Python :: 3.7",
            "Programming Language :: Python :: 3.8",
            "Programming Language :: Python :: 3.9",
            "Programming Language :: Python :: 3.10",
            "Topic :: Scientific/Engineering :: Artificial Intelligence",
            "Operating System :: OS Independent",
        ],
        zip_safe=True,
    )


if __name__ == "__main__":
    setup()
