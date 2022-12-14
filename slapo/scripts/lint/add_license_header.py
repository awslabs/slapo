# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# Modification: Apache TVM. See https://github.com/apache/tvm/blob/main/tests/lint/add_asf_header.py

"""Helper tool to add license header to files."""
import os
import sys

header_cstyle = """
/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
""".strip()

header_mdstyle = """
<!--- Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved. -->
<!--- SPDX-License-Identifier: Apache-2.0  -->
""".strip()

header_pystyle = """
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
""".strip()

header_rststyle = """
..  Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
    SPDX-License-Identifier: Apache-2.0
""".strip()

header_groovystyle = """
// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0
""".strip()

header_cmdstyle = """
:: Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
:: SPDX-License-Identifier: Apache-2.0
""".strip()

FMT_MAP = {
    "sh": header_pystyle,
    "cc": header_cstyle,
    "c": header_cstyle,
    "cu": header_cstyle,
    "cuh": header_cstyle,
    "mm": header_cstyle,
    "m": header_cstyle,
    "go": header_cstyle,
    "java": header_cstyle,
    "h": header_cstyle,
    "py": header_pystyle,
    "toml": header_pystyle,
    "yml": header_pystyle,
    "yaml": header_pystyle,
    "rs": header_cstyle,
    "md": header_mdstyle,
    "cmake": header_pystyle,
    "mk": header_pystyle,
    "rst": header_rststyle,
    "gradle": header_groovystyle,
    "tcl": header_pystyle,
    "xml": header_mdstyle,
    "storyboard": header_mdstyle,
    "pbxproj": header_cstyle,
    "plist": header_mdstyle,
    "xcworkspacedata": header_mdstyle,
    "html": header_mdstyle,
    "bat": header_cmdstyle,
}


def has_license_header(lines):
    """Check if the file has the license header."""
    copyright = False
    license = False
    for line in lines:
        if line.find("Copyright Amazon.com, Inc. or its affiliates.") != -1:
            copyright = True
        elif line.find("SPDX-License-Identifier") != -1:
            license = True
        if copyright and license:
            return True
    return False


def add_header(file_path, header):
    """Add header to file"""
    if not os.path.exists(file_path):
        print("%s does not exist" % file_path)
        return

    lines = open(file_path).readlines()
    if has_license_header(lines):
        print("%s has license header...skipped" % file_path)
        return

    print("%s miss license header...added" % file_path)
    with open(file_path, "w") as outfile:
        # Insert the license at the second line if the first line has a special usage.
        insert_line_idx = 0
        if lines and lines[0][:2] in ["#!", "<?", "<html>", "// !$"]:
            insert_line_idx = 1

        # Write the pre-license lines.
        for idx in range(insert_line_idx):
            outfile.write(lines[idx])

        # Write the license.
        outfile.write(header + "\n\n")

        # Wright the rest of the lines.
        outfile.write("".join(lines[insert_line_idx:]))


def main(args):
    if len(args) != 2:
        print("Usage: python add_license_header.py <file_list>")

    for file_path in open(args[1]):
        if file_path.startswith("---"):
            continue
        if file_path.find("File:") != -1:
            file_path = file_path.split(":")[-1]
        file_path = file_path.strip()

        if not file_path:
            continue

        suffix = file_path.split(".")[-1]
        if suffix in FMT_MAP:
            add_header(file_path, FMT_MAP[suffix])
        elif os.path.basename(file_path) == "gradle.properties":
            add_header(file_path, FMT_MAP["h"])
        else:
            print("Unrecognized file type: %s" % file_path)


if __name__ == "__main__":
    main(sys.argv)