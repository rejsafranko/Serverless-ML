import logging
import os

import dotenv
import wandb

from FeatureStorage import FeatureStorage
from ModelRepository import ModelRepository


class Config:
    def __init__(self) -> None:
        dotenv.load_dotenv()
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        self._environment_variables = self._validate_env()

    def _validate_env(self) -> dict[str, str]:
        required = [
            "AWS_ACCESS_KEY",
            "AWS_SECRET_KEY",
            "AWS_DATABASE_HOST",
            "AWS_DATABASE_NAME",
            "AWS_DATABASE_USERNAME",
            "AWS_DATABASE_PASSWORD",
            "WANDB_API_KEY",
        ]
        env = {var: os.getenv(var) for var in required}
        missing = [k for k, v in env.items() if not v]
        if missing:
            raise ValueError(f"Missing environment variables: {', '.join(missing)}")
        return env

    def configure_infrastructure(self) -> tuple[ModelRepository, FeatureStorage]:
        wandb.login(key=self._environment_variables["WANDB_API_KEY"])
        wandb.init(project="ml-demo", entity="codx-solutions")
        return (
            ModelRepository(
                access_key=self._environment_variables["AWS_ACCESS_KEY"],
                secret_key=self._environment_variables["AWS_SECRET_KEY"],
            ),
            FeatureStorage(
                host=self._environment_variables["AWS_DATABASE_HOST"],
                user=self._environment_variables["AWS_DATABASE_USERNAME"],
                password=self._environment_variables["AWS_DATABASE_PASSWORD"],
                database_name=self._environment_variables["AWS_DATABASE_NAME"],
            ),
        )
