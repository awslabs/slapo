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

####################
Slapo Documentation
####################

Slapo is a schedule language for progressive optimization of large model training. It decouples model execution from definition, enabling users to use a set of schedule primitives to convert a PyTorch model for common model training optimizations without directly changing the model itself, and thus addresses the tension between training efficiency and usability.


.. toctree::
   :maxdepth: 1
   :caption: Getting Started

   setup/index
   gallery/quick-start

.. toctree::
   :maxdepth: 1
   :caption: Tutorials

   gallery/attention-single-gpu

.. toctree::
   :maxdepth: 1
   :caption: Reference

   python_api/index
   genindex
