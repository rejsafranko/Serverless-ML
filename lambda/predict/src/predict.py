from __future__ import annotations
import json
import logging
import os

from typing import Any, Union

from Config import Config
from FeatureStorage import FeatureStorage
from ModelService import ModelService

logger = logging.getLogger()
logger.setLevel(logging.INFO)

_MODEL_SERVICE: ModelService | None = None
_FEATURE_STORAGE: FeatureStorage | None = None


def handler(event: dict[str, Any], context: Any) -> dict[str, Any]:
    global _MODEL_SERVICE, _FEATURE_STORAGE

    try:
        if _MODEL_SERVICE is None or _FEATURE_STORAGE is None:
            _MODEL_SERVICE, _FEATURE_STORAGE = _bootstrap()

        input_features = _parse_input(event)
        prediction = _MODEL_SERVICE.inference(input=input_features)  # type: ignore

        try:
            _FEATURE_STORAGE.store_new_labeled_feature(
                table_name=os.getenv("FEATURE_TABLE", "mental_health_features"),
                features=input_features,
                label=prediction,
            )
        except Exception:
            logger.exception("Feature storage write failed")

        return _response(200, {"prediction": prediction})

    except ValueError as ve:
        logger.warning("Bad request: %s", ve)
        return _response(400, {"message": str(ve)})

    except Exception:
        logger.exception("Inference failed")
        return _response(500, {"message": "Inference failed"})


def _bootstrap() -> tuple[ModelService, FeatureStorage]:
    logger.info("Cold start – initializing infrastructure")
    config = Config()
    model_repo, feature_storage = config.configure_infrastructure()
    model_name = _fetch_champion_model_name()
    logger.info("Champion model: %s", model_name)
    model_service = ModelService()
    model_service.set_model(
        model_repo.load_model(
            bucket_name=os.getenv("MODEL_BUCKET", "ml-demo-models"),
            model_name=model_name,
        )
    )
    return model_service, feature_storage


def _parse_input(event: dict[str, Any]) -> Union[list, dict]:
    try:
        body = event.get("body") or "{}"
        data = json.loads(body)
    except json.JSONDecodeError as exc:
        raise ValueError("Request body is not valid JSON") from exc

    input_features = data.get("features")
    if input_features is None:
        raise ValueError("'features' field missing in request body")
    return input_features


def _fetch_champion_model_name() -> str:
    import wandb

    runs = wandb.Api().runs(
        path="codx-solutions/ml-demo", order="-summary_metrics.accuracy"
    )
    if not runs:
        raise RuntimeError("No WandB runs found for project ‘ml-demo’")
    return runs[0].summary.get("model_name")


def _response(status_code: int, body: dict[str, Any]) -> dict[str, Any]:
    return {
        "statusCode": status_code,
        "headers": {"Content-Type": "application/json"},
        "body": json.dumps(body),
    }
