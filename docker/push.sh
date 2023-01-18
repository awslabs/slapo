#!/usr/bin/env bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

set -e

#
# Push the docker image to docker hub.
#
# Usage: push.sh <VERSION> <PASSWORD>
#
# VERSION: The pushed version. The format should be like v0.12.
#
# PASSWORD: The docker hub account password.
#
DOCKER_HUB_ACCOUNT=metaprojdev

# Get the version.
VERSION=$( echo "$1" | tr '[:upper:]' '[:lower:]' )
shift 1

# Get the docker hub account password.
PASSWORD="$1"
shift 1

LOCAL_IMAGE_NAME=slapo:latest
REMOTE_IMAGE_NAME_VER=${DOCKER_HUB_ACCOUNT}/dev:slapo-${VERSION}
REMOTE_IMAGE_NAME_LST=${DOCKER_HUB_ACCOUNT}/dev:slapo-latest

echo "Login docker hub"
docker login -u ${DOCKER_HUB_ACCOUNT} -p ${PASSWORD}

echo "Uploading ${LOCAL_IMAGE_NAME} as ${REMOTE_IMAGE_NAME_VER}"
docker tag ${LOCAL_IMAGE_NAME} ${REMOTE_IMAGE_NAME_VER}
docker push ${REMOTE_IMAGE_NAME_VER}

echo "Uploading ${LOCAL_IMAGE_NAME} as ${REMOTE_IMAGE_NAME_LST}"
docker tag ${LOCAL_IMAGE_NAME} ${REMOTE_IMAGE_NAME_LST}
docker push ${REMOTE_IMAGE_NAME_LST}
