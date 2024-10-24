import os

import dotenv

from typing import Dict, Tuple

from .FeatureStorage import FeatureStorage
from .ModelRepository import ModelRepository


class Config:
    def __init__(self):
        dotenv.load_dotenv()
        self._environment_variables: Dict[str, str] = (
            self._configure_environment_variables()
        )

    def _configure_environment_variables(self) -> None:
        access_key = os.getenv("AWS_ACCESS_KEY")
        secret_key = os.getenv("AWS_SECRET_KEY")
        host = os.getenv("AWS_DATABASE_HOST")  # RDS host
        database_name = os.getenv("AWS_DATABASE_NAME")
        user = os.getenv("AWS_DATABASE_USERNAME")
        password = os.getenv("AWS_DATABASE_PASSWORD")
        self._environment_variables = {
            "access_key": access_key,
            "secret_key": secret_key,
            "host": host,
            "database_name": database_name,
            "user": user,
            "password": password,
        }

    def configure_infrastructure(self) -> Tuple[ModelRepository, FeatureStorage]:
        model_repository = ModelRepository(
            access_key=self._environment_variables["access_key"],
            secret_key=self._environment_variables["secret_key"],
        )

        feature_storage = FeatureStorage(
            host=self._environment_variables["host"],
            user=self._environment_variables["user"],
            password=self._environment_variables["password"],
            database_name=self._environment_variables["database_name"],
        )

        return (model_repository, feature_storage)
