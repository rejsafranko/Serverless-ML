import datetime
import os

import boto3

_ssm = boto3.client("ssm")


def get_last_run() -> datetime.datetime | None:
    param = os.getenv("SSM_LAST_RUN_PARAM")
    if not param:
        return None
    try:
        resp = _ssm.get_parameter(Name=param)
        return datetime.fromisoformat(resp["Parameter"]["Value"])
    except _ssm.exceptions.ParameterNotFound:
        return None


def set_last_run() -> None:
    param = os.getenv("SSM_LAST_RUN_PARAM")
    if not param:
        return
    _ssm.put_parameter(
        Name=param,
        Value=datetime.now(datetime.timezone.utc).isoformat(),
        Overwrite=True,
        Tier="Standard",
    )
