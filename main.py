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

    evaluator = Evaluator(
        alg=alg,
        dataset=dataset,
        n_data_splits=cfg.evaluator.n_data_splits,
        n_alg_runs=cfg.evaluator.n_alg_runs,
        random_state=cfg.evaluator.random_state,
        verbose=cfg.evaluator.verbose
    )

    mean_score, std_score = evaluator.run()
    print(f"Mean Score: {mean_score:.4f}, Std Dev: {std_score:.4f}")

if __name__ == "__main__":
    main()