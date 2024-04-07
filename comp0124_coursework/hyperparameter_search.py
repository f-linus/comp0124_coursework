import itertools
import logging

import pandas as pd

from comp0124_coursework.model.base_environment import BaseEnvironment

logger = logging.getLogger(__name__)


def hyperparameter_search(search_space: dict):
    parameter_combinations = itertools.product(
        *[search_space[key] for key in search_space]
    )

    parameter_dicts = [
        dict(zip(search_space.keys(), combination))
        for combination in parameter_combinations
    ]

    logger.info(f"Searching over {len(parameter_dicts)} parameter combinations")

    results = pd.DataFrame()
    for parameters in parameter_dicts:
        logger.info(f"Running with parameters: {parameters}")
        environment = BaseEnvironment()
        environment.run(**parameters, render_to_file=False, logging_interval=1000)

        result = pd.DataFrame({k: str(v) for k, v in parameters.items()}, index=[0])
        result["food_collected"] = environment.food_collected

        results = pd.concat([results, result])
        results.to_csv("hyperparameter_search_results.csv", index=False)

        logger.info(f"Total food collected: {environment.food_collected}")

    return


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    search_space = {
        "size": [(600, 600)],
        "nest_location": [(300, 300)],
        "no_ants": [200],
        "no_iterations": [5000],
        "pheremone_decay_rate": [0.1, 0.3, 0.5, 0.7],
        "pheremone_intensity_coefficient": [0.003, 0.005, 0.007],
    }

    hyperparameter_search(search_space=search_space)
