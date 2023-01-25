# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import copy
import json
import re
import sys
import time
from datetime import datetime

import boto3

LOG_GROUP_NAME = "/aws/batch/job"


def create_config():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--profile", help="profile name of aws account.", type=str, default=None
    )
    parser.add_argument(
        "--region",
        help="Default region when creating new connections",
        type=str,
        default="us-west-2",
    )
    parser.add_argument(
        "--platform",
        help="The platform to run the job on",
        type=str,
        choices=["CPU", "GPU", "multi-GPU"],
        default="CPU",
    )
    parser.add_argument(
        "--image",
        help="The docker image for running the job",
        type=str,
    )
    parser.add_argument("--name", help="name of the job", type=str, default="ci-job")
    parser.add_argument(
        "--job-queue",
        help="The job queue",
        type=str,
        default="ci-cpu-queue",
    )
    parser.add_argument(
        "--job-def-cfg",
        help="The job definition config file",
        type=str,
        default="ci/batch/job-def-cfg.json",
    )
    parser.add_argument(
        "--entry-script",
        help="The path to the job entry script *in the docker image*",
        type=str,
        default="/batch/entry.sh",
    )
    parser.add_argument(
        "--source-ref", help="e.g. main, refs/pull/500/head", type=str, default="main"
    )
    parser.add_argument(
        "--repo", help="e.g. user_name/raf", type=str, default="awslabs/raf"
    )
    parser.add_argument(
        "--save-output",
        help="output to be saved, relative to working directory. "
        "it can be either a single file or a directory",
        type=str,
    )
    parser.add_argument(
        "--command",
        help="command to run",
        type=str,
        default="git rev-parse HEAD | tee stdout.log",
    )
    parser.add_argument(
        "--wait",
        help="block wait until the job completes." "Non-zero exit code if job fails.",
        action="store_true",
    )
    parser.add_argument(
        "--timeout",
        help="job timeout in minutes. Default 120 mins",
        default=120,
        type=int,
    )

    return parser.parse_args()


def tprint_custom_time(timestampe, message):
    """Print a message to stdout with the given timestamp."""
    print("[%s UTC] %s" % (timestampe.isoformat()[:19], message))


def tprint(message):
    """Print a message to stdout with a current timestamp."""
    tprint_custom_time(datetime.utcnow(), message)


def fetch_cloud_watch_logs(cloudwatch, log_stream_name, start_timestamp):
    """Fetch CloudWatch log stream and print new logs since the given start time.
    Parameters
    ----------
    cloudwatch : boto3.client
        Boto3 client for CloudWatch.
    log_stream_name : str
        Name of the CloudWatch log stream.
    start_timestamp : int
        Start time of the CloudWatch log stream.
    Returns
    -------
    last_time_stamp: int
        The current last time stamp of the CloudWatch log stream.
    """
    event_args = {
        "logGroupName": LOG_GROUP_NAME,
        "logStreamName": log_stream_name,
        "startTime": start_timestamp,
        "startFromHead": True,  # Query logs from oldest to newest.
    }

    last_timestamp = start_timestamp - 1
    while True:
        log_events = cloudwatch.get_log_events(**event_args)

        for event in log_events["events"]:
            last_timestamp = event["timestamp"]
            tprint_custom_time(
                datetime.utcfromtimestamp(last_timestamp / 1000.0), event["message"]
            )

        next_token = log_events["nextForwardToken"]
        if next_token and event_args.get("nextToken") != next_token:
            event_args["nextToken"] = next_token
        else:
            break
    return last_timestamp


def terminate_previous_job(aws_batch, job_queue, job_name):
    """Terminate the running jobs with the same job name.
    Parameters
    ----------
    aws_batch : boto3.client
    job_queue : str
        Name of the job queue.
    job_name : str
        Name of the job.
    """
    list_args = {
        "jobQueue": job_queue,
        "filters": [{"name": "JOB_NAME", "values": [job_name]}],
    }
    while True:
        try:
            response = aws_batch.list_jobs(**list_args)
        except Exception:
            return
        for job in response["jobSummaryList"]:
            if job["status"] in [
                "SUBMITTED",
                "PENDING",
                "RUNNABLE",
                "STARTING",
                "RUNNING",
            ]:
                tprint("Terminate previous job %s" % job["jobId"])
                aws_batch.terminate_job(jobId=job["jobId"], reason="New job submitted")

        next_token = response.get("nextToken")
        if next_token and list_args.get("nextToken") != next_token:
            list_args["nextToken"] = next_token
        else:
            break


def get_job_def(aws_batch, job_def_cfg_file, platform, image):
    """Get the job definition from the config file and platform. It first uses the config file
    to find the job definition name by the given platform, and then determine the revision
    by the docker image name. There are several cases:
    1. Found zero revision: Create a new job definition revision based on the latest one.
    2. Found one revision: Use the revision.
    3. Found more than one revision: Use the latest revision.
    Parameters
    ----------
    aws_batch : boto3.client
        Boto3 client for AWS Batch.
    job_def_cfg_file : str
        The job definition config file.
    platform : str
        The platform to run the job on.
    image : str
        The docker image name and tag.
    Returns
    -------
    job_def : str
        The complete job definition name and revision.
    """
    with open(job_def_cfg_file) as filep:
        job_def_cfg = json.load(filep)

    if platform not in job_def_cfg:
        raise ValueError(
            "Platform %s is not specified in the job definition config file %s"
            % (platform, job_def_cfg_file)
        )
    job_def_name = job_def_cfg[platform]

    # Query for all revisions.
    job_defs = []
    desc_args = {
        "jobDefinitionName": job_def_name,
        "status": "ACTIVE",
    }
    while True:
        response = aws_batch.describe_job_definitions(**desc_args)
        job_defs += response["jobDefinitions"]
        if not job_defs:
            raise RuntimeError(
                "At least one revision has to be craeted and actived for job definition %s"
                % job_def_name
            )

        next_token = response.get("nextToken")
        if next_token and desc_args.get("nextToken") != next_token:
            desc_args["nextToken"] = next_token
        else:
            break

    match_revision = None
    latest_revision = None
    for job_def in job_defs:
        if "containerProperties" in job_def:
            if job_def["containerProperties"]["image"] == image:
                if (
                    not match_revision
                    or match_revision["revision"] < job_def["revision"]
                ):
                    match_revision = job_def
            if not latest_revision or latest_revision["revision"] < job_def["revision"]:
                latest_revision = job_def

    # Create a new revision based on the latest one.
    if match_revision is None:
        if latest_revision is None:
            raise RuntimeError(
                "No job definition revision with containerProperties found for %s"
                % job_def_name
            )
        print(
            "Create a new revision for job definition %s with image %s based on revision %d"
            % (job_def_name, image, latest_revision["revision"])
        )
        new_job_def = copy.deepcopy(job_def)
        del new_job_def["revision"]
        del new_job_def["status"]
        del new_job_def["jobDefinitionArn"]
        new_job_def["containerProperties"]["image"] = image
        response = aws_batch.register_job_definition(**new_job_def)
        new_job_def["revision"] = response["revision"]
        match_revision = new_job_def

    # Return the latest matched revision.
    return f"{job_def_name}:{match_revision['revision']}"


def main():
    args = create_config()

    # Establish AWS clients (AWS credentaial is required).
    session = boto3.Session(profile_name=args.profile, region_name=args.region)
    aws_batch = session.client("batch")
    aws_cloudwatch_logs = session.client("logs")

    # Process job information.
    job_name = re.sub("[^A-Za-z0-9_\-]", "", args.name)[:128]  # Canonicalize job name.
    job_queue = args.job_queue
    job_def = get_job_def(aws_batch, args.job_def_cfg, args.platform, args.image)
    wait = args.wait

    parameters = {
        "ENTRY_SCRIPT": args.entry_script,
        "SOURCE_REF": args.source_ref,
        "REPO": args.repo,
        "PLATFORM": args.platform,
        "COMMAND": args.command,
    }
    job_args = dict(
        jobName=job_name,
        jobQueue=job_queue,
        jobDefinition=job_def,
        timeout={"attemptDurationSeconds": int(args.timeout) * 60},
        containerOverrides={
            "command": [
                "bash",
                "Ref::ENTRY_SCRIPT",
                "Ref::SOURCE_REF",
                "Ref::REPO",
                "Ref::COMMAND",
            ],
        },
        parameters=parameters,
    )
    if args.save_output:
        job_args["containerOverrides"]["command"].append("SAVE_OUTPUT=Ref::SAVE_OUTPUT")
        parameters["SAVE_OUTPUT"] = args.save_output

    terminate_previous_job(aws_batch, job_queue, job_name)
    response = aws_batch.submit_job(**job_args)

    job_id = response["jobId"]
    tprint(
        "Submitted job %s: %s (ID: %s) to the queue %s"
        % (job_def, job_name, job_id, job_queue)
    )

    running = False
    start_timestamp = 0
    log_stream_name = None
    print_period = 30  # The period to print the job status in seconds.
    while wait:
        response = aws_batch.describe_jobs(jobs=[job_id])
        status = response["jobs"][0]["status"]
        if status in {"SUCCEEDED", "FAILED"}:
            if log_stream_name is None:
                # If the job is ended within a print period so that
                # we have not got the log stream name, we need to get it here.
                log_stream_name = response["jobs"][0]["container"]["logStreamName"]
            if log_stream_name:
                start_timestamp = (
                    fetch_cloud_watch_logs(
                        aws_cloudwatch_logs, log_stream_name, start_timestamp
                    )
                    + 1
                )

            reason = (
                "(reason: %s)" % response["jobs"][0]["statusReason"]
                if "statusReason" in response["jobs"][0]
                else ""
            )
            tprint("=" * 80)
            tprint("[%s (%s)] %s %s" % (job_name, job_id, status, reason))
            sys.exit(status == "FAILED")
        elif status == "RUNNING":
            print_period = (
                10  # Shorten the period to print the job status when it starts running.
            )
            log_stream_name = response["jobs"][0]["container"]["logStreamName"]
            if not running:
                running = True
                tprint("[%s (%s)] Start running" % (job_name, job_id))
                if log_stream_name:
                    tprint("LogStream %s\n%s" % (log_stream_name, "=" * 80))
            if log_stream_name:
                start_timestamp = (
                    fetch_cloud_watch_logs(
                        aws_cloudwatch_logs, log_stream_name, start_timestamp
                    )
                    + 1
                )
        else:
            tprint("[%s (%s)] Now at status %s" % (job_name, job_id, status))
        sys.stdout.flush()
        time.sleep(print_period)


if __name__ == "__main__":
    main()
