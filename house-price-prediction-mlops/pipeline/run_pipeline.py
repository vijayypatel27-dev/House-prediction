from src.data_ingestion import DataIngestion
from src.data_preprocessing import DataPreprocessor
from src.model_training import ModelTrainer


def run_pipeline(config_path: str = "params.yaml") -> None:
    DataIngestion(config_path).run()
    DataPreprocessor(config_path).run()
    ModelTrainer(config_path).run()


if __name__ == "__main__":
    run_pipeline()
