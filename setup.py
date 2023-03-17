# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import re
import subprocess

import setuptools

ROOT_DIR = os.path.dirname(__file__)


def get_version_in_version_py(root_dir):
    """Get the current version specified in version.py."""
    with open(os.path.join(root_dir, "slapo", "version.py")) as fp:
        version_match = re.search(
            r"^__version__ = ['\"]([^'\"]*)['\"]", fp.read(), re.M
        )
        if version_match:
            return version_match.group(1)
    raise RuntimeError("Unable to find version string")


def py_str(cstr):
    return cstr.decode("utf-8")


def get_version():
    """Get PEP-440 compatible public and local version using git describe.
    If not applicable, fallback to the version specified in version.py.

    Returns
    -------
    pub_ver: str
        Public version.
    local_ver: str
        Local version (with additional label appended to pub_ver).

    Notes
    -----
    - We follow PEP 440's convention of public version
      and local versions.
    - Only tags conforming to vMAJOR.MINOR.REV (e.g. "v0.7.0")
      are considered in order to generate the version string.
      See the use of `--match` in the `git` command below.
    Here are some examples:
    - pub_ver = '0.7.0', local_ver = '0.7.0':
      We are at the 0.7.0 release.
    - pub_ver =  '0.8.dev94', local_ver = '0.8.dev94+g0d07a329e':
      We are at the the 0.8 development cycle.
      The current source contains 94 additional commits
      after the most recent tag(v0.7.0),
      the git short hash tag of the current commit is 0d07a329e.
    """
    curr_version = get_version_in_version_py(ROOT_DIR)
    cmd = [
        "git",
        "describe",
        "--tags",
        "--match",
        "v[0-9]*.[0-9]*.[0-9]*",
        "--match",
        "v[0-9]*.[0-9]*.dev[0-9]*",
    ]
    try:
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, cwd=ROOT_DIR
        )
        (out, _) = proc.communicate()
        msg = py_str(out)
        retcode = proc.returncode
    except Exception as err:
        msg = str(err)
        retcode = -1

    if retcode != 0:
        if msg.find("not a git repository") != -1:
            return curr_version, curr_version
        print("WARNING: git describe: %s, use %s", msg, curr_version)
        return curr_version, curr_version
    describe = msg.strip()
    arr_info = describe.split("-")

    # Remove the v prefix, mainly to be robust
    # to the case where v is not presented as well.
    if arr_info[0].startswith("v"):
        arr_info[0] = arr_info[0][1:]

    # Hit the exact tag (stable version).
    if len(arr_info) == 1:
        return arr_info[0], arr_info[0]

    if len(arr_info) != 3:
        print("WARNING: Invalid output from git describe %s", describe)
        return curr_version, curr_version

    dev_pos = arr_info[0].find(".dev")

    # Development versions:
    # The code will reach this point in case it can't match a full release version,
    # such as v0.7.0.
    #
    # 1. in case the last known label looks like vMAJ.MIN.devN e.g. v0.8.dev0, we use
    # the current behaviour of just using vMAJ.MIN.devNNNN+gGIT_REV
    if dev_pos != -1:
        dev_version = arr_info[0][: arr_info[0].find(".dev")]
    # 2. in case the last known label looks like vMAJ.MIN.PATCH e.g. v0.8.0
    # then we just carry on with a similar version to what git describe provides,
    # which is vMAJ.MIN.PATCH.devNNNN+gGIT_REV
    else:
        dev_version = arr_info[0]

    pub_ver = "%s.dev%s" % (dev_version, arr_info[1])
    local_ver = "%s+%s" % (pub_ver, arr_info[2])
    return pub_ver, local_ver


def setup():
    with open("README.md", encoding="utf-8") as fp:
        long_description = fp.read()

    setuptools.setup(
        name="slapo",
        description="Slapo: A Schedule Language for Progressive Optimization.",
        version=get_version()[1],
        author="Slapo Community",
        long_description=long_description,
        long_description_content_type="text/markdown",
        setup_requires=[],
        install_requires=[
            "packaging",
            "psutil",
        ],
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
