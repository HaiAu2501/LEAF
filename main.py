import hydra
from dotenv import load_dotenv
from hydra.utils import instantiate
from omegaconf import DictConfig

from src.template import Algorithm
from src.evaluator import Evaluator
from utils.loader import Dataset
from utils.logger import Logger

load_dotenv()

@hydra.main(version_base=None, config_path="cfg", config_name="config")
def main(cfg: DictConfig):
    logger: Logger = Logger(log_dir=cfg.paths.log_dir)
    alg: Algorithm = instantiate(cfg.alg, logger=logger)
    dataset: Dataset = instantiate(cfg.dataset, logger=logger)
    evaluator: Evaluator = instantiate(cfg.evaluator, alg=alg, dataset=dataset)

    print(f"Dataset: {dataset.name} | Task: {dataset.task_type} | Algorithm: {alg.name}")
    mean_score, std_score = evaluator.run()
    print(f"Mean Score: {mean_score:.4f}, Std Dev: {std_score:.4f}")

if __name__ == "__main__":
    main()