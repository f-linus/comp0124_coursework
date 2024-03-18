import logging
import time

import cv2
import numpy as np
import pandas as pd

from comp0124_coursework.agent.ant import Ant

logger = logging.getLogger(__name__)


class Grid:
    def __init__(self, size: tuple) -> None:
        self.size = size

        self.ant_layer = np.zeros(size, dtype=np.int8)
        self.food_layer = np.zeros(size, dtype=np.int8)
        self.home_pheromone_layer = np.zeros(size, dtype=np.float32)
        self.food_pheromone_layer = np.zeros(size, dtype=np.float32)


class BaseEnvironment:
    def __init__(self):
        pass

    def run(
        self,
        size: tuple,
        nest_location: tuple,
        no_ants=200,
        no_iterations=10000,
        phereomone_decay_rate=0.6,
        logging_interval=50,
        render_to_screen=False,
        render_interval=1,
        render_step_wait_time=1,
        save_rendering_interval=1000,
        extra_iterations_to_save_rendering=[10, 100, 500],
    ):
        self.nest_location = nest_location
        self.pheremonone_decay_rate = phereomone_decay_rate
        self.logging_interval = logging_interval
        self.render_to_screen = render_to_screen
        self.render_interval = render_interval
        self.render_step_wait_time = render_step_wait_time
        self.save_rendering_interval = save_rendering_interval
        self.extra_iterations_to_save_rendering = extra_iterations_to_save_rendering

        self.grid = Grid(size)
        self.ants = set()
        self.avg_step_time = 0
        self.avg_render_time = 0
        self.iteration = 1
        self.food_consumption_log = pd.DataFrame()

        # spawn ants
        for i in range(no_ants):
            rnd_rotation = np.random.uniform(0, 2 * np.pi)

            ant = Ant(f"ant_{i}", self, self.nest_location, rnd_rotation)
            self.ants.add(ant)

        # spawn some food
        self.grid.food_layer[0:100, 0:100] = 1
        self.grid.food_layer[200:300, 0:100] = 1
        self.grid.food_layer[400:500, 0:100] = 1

        # save initial food amount
        self.last_food_amount = np.sum(self.grid.food_layer)

        if render_to_screen:
            cv2.namedWindow("env_vis", cv2.WINDOW_NORMAL)

        while self.iteration <= no_iterations:
            self.render_handling()
            self.step()
            self.log_handling()

            self.iteration += 1

        cv2.destroyAllWindows()

    def step(self):
        start_time = time.time()

        self.pheremonone_decay()

        # step ants
        for ant in self.ants:
            ant.step()

        self.avg_step_time += (time.time() - start_time) * 1000 / self.logging_interval

    def pheremonone_decay(self):
        self.grid.home_pheromone_layer = np.max(
            np.stack(
                [
                    self.grid.home_pheromone_layer - self.pheremonone_decay_rate,
                    np.zeros(self.grid.size),
                ],
                axis=2,
            ),
            axis=2,
        )
        self.grid.food_pheromone_layer = np.max(
            np.stack(
                [
                    self.grid.food_pheromone_layer - self.pheremonone_decay_rate,
                    np.zeros(self.grid.size),
                ],
                axis=2,
            ),
            axis=2,
        )

    def log_handling(self):
        if self.iteration % self.logging_interval == 0 and self.iteration != 0:
            logger.info(
                f"Step: {self.iteration} - Total step time: {self.avg_step_time + self.avg_render_time:.2f}ms (step time: {self.avg_step_time:.2f}ms - render time: {self.avg_render_time:.2f}ms)"
            )
            logger.info(
                f"Food collected over last {self.logging_interval} steps: {self.last_food_amount - np.sum(self.grid.food_layer)} (remaining: {np.sum(self.grid.food_layer)})"
            )

            # append to food consumption log
            self.food_consumption_log = pd.concat(
                [
                    self.food_consumption_log,
                    pd.DataFrame(
                        {
                            "iteration": self.iteration,
                            "food_collected": self.last_food_amount
                            - np.sum(self.grid.food_layer),
                            "food_remaining": np.sum(self.grid.food_layer),
                        },
                        index=[0],
                    ),
                ]
            )

            # update CSV file (or crete it if it does not exist)
            self.food_consumption_log.to_csv("food_consumption_log.csv", index=False)

            self.avg_step_time = 0
            self.avg_render_time = 0
            self.last_food_amount = np.sum(self.grid.food_layer)

    def render_handling(
        self,
    ):
        if self.render_to_screen and self.iteration % self.render_interval == 0:
            start_time = time.time()
            image = self.render()
            cv2.imshow("env_vis", image)

            cv2.waitKey(self.render_step_wait_time)
            self.avg_render_time += (
                (time.time() - start_time)
                * 1000
                / self.render_interval
                / self.logging_interval
            )

        if self.iteration in self.extra_iterations_to_save_rendering or (
            self.iteration % self.save_rendering_interval == 0
        ):
            if not self.render_to_screen or self.iteration % self.render_interval != 0:
                image = self.render()
            cv2.imwrite(f"render_{self.iteration}.png", image * 255)

    def render(self):
        # food phereomone layer
        home_layer = (
            np.repeat(self.grid.home_pheromone_layer[:, :, np.newaxis], 3, axis=2)
            * [1, 0.3, 0.3]
            / 1000
        )

        food_layer = (
            np.repeat(self.grid.food_pheromone_layer[:, :, np.newaxis], 3, axis=2)
            * [0.3, 1, 0.3]
            / 1000
        )

        image = np.clip(home_layer + food_layer, 0, 1)

        # ants
        image[self.grid.ant_layer == 1] = [1.0, 1.0, 1.0]

        # food
        image[self.grid.food_layer == 1] = [0.0, 0.0, 1.0]

        # 6x6 for nest
        image[
            self.nest_location[0] - 3 : self.nest_location[0] + 3,
            self.nest_location[1] - 3 : self.nest_location[1] + 3,
        ] = [
            1.0,
            1.0,
            0.0,
        ]
        return image


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    no_of_ants_scenarios = [50, 100]  # , 150, 200, 250, 300, 350, 400, 450, 500]
    avg_food_collected_per_log_interval = []

    for no_of_ants in no_of_ants_scenarios:
        env = BaseEnvironment()
        env.run(
            size=(600, 600),
            no_iterations=5000,
            nest_location=(300, 300),
            no_ants=no_of_ants,
            render_to_screen=False,
        )

        avg_food_collected_per_log_interval.append(
            env.food_consumption_log["food_collected"].mean()
        )

        print(f"Finished scenario with {no_of_ants} ants")

    # create CSV file with results
    pd.DataFrame(
        {
            "no_of_ants": no_of_ants_scenarios,
            "avg_food_collected_per_log_interval": avg_food_collected_per_log_interval,
        }
    ).to_csv("food_collected_per_no_ants.csv", index=False)
