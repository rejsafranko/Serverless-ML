import logging
import os

import dotenv
import wandb

from .FeatureStorage import FeatureStorage
from .ModelRepository import ModelRepository


class Config:
    def __init__(self) -> None:
        dotenv.load_dotenv()
        self._environment_variables: dict[str, str] = (
            self._configure_environment_variables()
        )

        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def _configure_environment_variables(self) -> dict[str, str]:
        """
        Loads and configures environment variables.
        Logs missing variables and raises an error if any are missing.
        """
        required_vars = [
            "AWS_ACCESS_KEY",
            "AWS_SECRET_KEY",
            "AWS_DATABASE_HOST",
            "AWS_DATABASE_NAME",
            "AWS_DATABASE_USERNAME",
            "AWS_DATABASE_PASSWORD",
            "WANDB_API_KEY",
        ]

        env_vars = {var: os.getenv(var) for var in required_vars}

        missing_vars = [var for var, value in env_vars.items() if not value]
        if missing_vars:
            self.logger.error(
                f"Missing environment variables: {', '.join(missing_vars)}"
            )
            raise ValueError(
                f"Missing environment variables: {', '.join(missing_vars)}"
            )

        return env_vars

    def configure_infrastructure(
        self,
    ) -> tuple[ModelRepository, FeatureStorage]:
        """
        Configures the infrastructure: ModelRepository and FeatureStorage.
        Initializes WandB, S3, and Database connections.

        :return: Tuple containing ModelRepository and FeatureStorage instances
        """
        wandb_api_key = self._environment_variables["WANDB_API_KEY"]
        wandb.login(key=wandb_api_key)
        wandb.init(project="ml-demo", entity="codx-solutions")

        model_repository = ModelRepository(
            access_key=self._environment_variables["AWS_ACCESS_KEY"],
            secret_key=self._environment_variables["AWS_SECRET_KEY"],
        )
        feature_storage = FeatureStorage(
            host=self._environment_variables["AWS_DATABASE_HOST"],
            user=self._environment_variables["AWS_DATABASE_USERNAME"],
            password=self._environment_variables["AWS_DATABASE_PASSWORD"],
            database_name=self._environment_variables["AWS_DATABASE_NAME"],
        )

        self.logger.info("Infrastructure configured successfully.")

        return model_repository, feature_storage
