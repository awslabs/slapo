..  Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
    SPDX-License-Identifier: Apache-2.0

..  Licensed to the Apache Software Foundation (ASF) under one
    or more contributor license agreements.  See the NOTICE file
    distributed with this work for additional information
    regarding copyright ownership.  The ASF licenses this file
    to you under the Apache License, Version 2.0 (the
    "License"); you may not use this file except in compliance
    with the License.  You may obtain a copy of the License at

..    http://www.apache.org/licenses/LICENSE-2.0

..  Unless required by applicable law or agreed to in writing,
    software distributed under the License is distributed on an
    "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
    KIND, either express or implied.  See the License for the
    specific language governing permissions and limitations
    under the License.

.. _setup:

############
Installation
############

We provide two approaches to installing Slapo. The easiest way is to install Slapo through PYPI:

.. code-block:: console

  $ pip install slapo

For developers, you can also install Slapo from source, and the change to the codebase will directly take effect when importing the package.

.. code-block:: console

  $ git clone https://github.com/awslabs/slapo.git slapo
  $ cd slapo
  $ pip install -e ".[dev]"

To verify the installation, you can run the following command. If no output is printed, you have installed Slapo successfully!

.. code-block:: console

  $ python -c "import slapo"
