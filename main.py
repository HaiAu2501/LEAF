import hydra
import numpy as np

from dotenv import load_dotenv
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from src.template import Algorithm
from src.evaluator import Evaluator
from utils.loader import Dataset

load_dotenv()

OmegaConf.register_new_resolver("range", lambda start, end: range(int(start), int(end)))
OmegaConf.register_new_resolver("logscale", lambda start, end, num: np.logspace(np.log10(float(start)), np.log10(float(end)), int(num)))

@hydra.main(version_base=None, config_path="cfg", config_name="config")
def main(cfg: DictConfig):
    print("Experiment Configuration:")
    # print(cfg)
    
    alg: Algorithm = instantiate(cfg.alg)
    # print(alg.param_grid)
    
    dataset: Dataset = instantiate(cfg.dataset)
    # print(dataset.name)

    evaluator = Evaluator(
        alg=alg,
        dataset=dataset,
        n_data_splits=cfg.evaluator.n_data_splits,
        n_alg_runs=cfg.evaluator.n_alg_runs,
        random_state=cfg.evaluator.random_state,
        verbose=cfg.evaluator.verbose
    )

    mean_score, std_score = evaluator.run()
    print(f"Mean Score: {mean_score:.2f}, Std Dev: {std_score:.2f}")

if __name__ == "__main__":
    main()