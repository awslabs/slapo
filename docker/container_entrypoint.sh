# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
#!/bin/bash
# This script is called on container startup.

CLUSTER_USER=deepspeed

echo "starting container services..."
if [ ! -d /run/sshd ]; then
    echo "creating /run/sshd"
    sudo mkdir -p /run/sshd
fi

echo "starting SSHD."
/usr/sbin/sshd -f /home/$CLUSTER_USER/.ssh/sshd_config

sleep infinity

