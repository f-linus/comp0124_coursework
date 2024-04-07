import itertools
import logging
import os

import pandas as pd

from comp0124_coursework.model.base_environment import BaseEnvironment

logger = logging.getLogger(__name__)


def experiment(search_space: dict):
    parameter_combinations = itertools.product(
        *[search_space[key] for key in search_space]
    )

    parameter_dicts = [
        dict(zip(search_space.keys(), combination))
        for combination in parameter_combinations
    ]

    logger.info(f"Searching over {len(parameter_dicts)} parameter combinations")

    # Load previous results if they exist
    if os.path.exists("experiment_results.csv"):
        results = pd.read_csv("experiment_results.csv")
    else:
        results = pd.DataFrame()

    # Run the experiment
    for parameters in parameter_dicts:
        index = len(results)

        logger.info(f"Running with parameters: {parameters}")

        # create dir to store renders to
        os.makedirs(f"renders_experiment{index}", exist_ok=True)

        environment = BaseEnvironment()
        environment.run(
            **parameters,
            render_file_dir=f"renders_experiment{index}",
            logging_interval=500,
        )

        result = pd.DataFrame({k: str(v) for k, v in parameters.items()}, index=[index])
        result.index.name = "experiment"

        result["food_collected"] = environment.food_collected
        result["collection_rate_mean"] = environment.food_collection_rates.mean()
        result["collection_rate_std"] = environment.food_collection_rates.std()

        results = pd.concat([results, result])
        results.to_csv("experiment_results.csv", index=True)

        logger.info(f"Total food collected: {environment.food_collected}")
        logger.info(f"Mean collection rate: {environment.food_collection_rates.mean()}")
        logger.info(f"Std collection rate: {environment.food_collection_rates.std()}")
    return


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    search_space = {
        "size": [(600, 600)],
        "nest_location": [(300, 300)],
        "no_ants": [200],
        "no_iterations": [5000],
        "pheremone_decay_rate": [0.5, 0.1, 0.2, 0.3],
        "pheremone_intensity_coefficient": [0.001, 0.002, 0.003],
    }

    experiment(search_space=search_space)
