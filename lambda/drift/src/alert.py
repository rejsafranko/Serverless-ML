import os
import json

import aws_lambda_powertools
import boto3

logger = aws_lambda_powertools.Logger()
_sns = boto3.client("sns")
SNS_TOPIC_ARN = os.getenv("SNS_TOPIC_ARN")


def alert_drift(columns: list[str], ks_stats: dict[str, float]):
    if not SNS_TOPIC_ARN:
        logger.warning("Drift detected but SNS_TOPIC_ARN not set")
        return
    msg = {
        "message": "Data drift detected",
        "columns": columns,
        "ks_statistics": {c: ks_stats[c] for c in columns},
    }
    _sns.publish(
        TopicArn=SNS_TOPIC_ARN,
        Message=json.dumps(msg),
        Subject="⚠️ Serverless-ML Drift Alert",
    )
    logger.info("Alert published to SNS")
