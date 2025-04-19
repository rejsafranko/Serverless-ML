import os
import time

import aws_lambda_powertools

from alert import alert_drift
from db import fetch_new_rows, persist_ks_results, get_connection
from ks_test import run_ks_tests, split_reference_current
from ssm import get_last_run, set_last_run
from utils import response

logger = aws_lambda_powertools.Logger()
tracer = aws_lambda_powertools.Tracer()

MIN_ROWS = int(os.getenv("MIN_ROWS", "100"))
DRIFT_THRESHOLD = float(os.getenv("DRIFT_THRESHOLD", "0.05"))


@tracer.capture_lambda_handler
@logger.inject_lambda_context
def handler(event, context):
    start_ts = time.time()
    last_run = get_last_run()

    with get_connection() as conn:
        new_rows = fetch_new_rows(conn, since=last_run)
        logger.info("Fetched %s new rows", len(new_rows))

        if len(new_rows) < MIN_ROWS:
            logger.info("Not enough data; skipping drift check")
            set_last_run()
            return response({"message": "skip", "rows": len(new_rows)})

        reference, current = split_reference_current(new_rows)
        ks_results = run_ks_tests(reference, current)

        logger.info("KS results: %s", ks_results)
        persist_ks_results(conn, ks_results)

        drifted_columns = [
            col for col, stat in ks_results.items() if stat > DRIFT_THRESHOLD
        ]

    if drifted_columns:
        alert_drift(drifted_columns, ks_results)

    set_last_run()

    return response(
        {
            "duration_ms": round((time.time() - start_ts) * 1000),
            "rows": len(new_rows),
            "drifted_columns": drifted_columns,
        }
    )
