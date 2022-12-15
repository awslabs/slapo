<!--- Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved. -->
<!--- SPDX-License-Identifier: Apache-2.0  -->

# WideResNet

The scripts are modified from:
https://github.com/zarzen/DeepSpeedExamples/blob/branch-v0.3.16-47ec97e/WideResnet

To run with DeepSpeed, you need to check ds_config_size.json to make sure
the batch sizes are correct. Then run the following command:

```
bash ds_launch.sh train.py ds_config_250m.json
```

Note that ds_launch.sh now only runs on a single node. If you want to test on multiple nodes,
you need to provide `hostfile` and uncomment related logic in the script.
