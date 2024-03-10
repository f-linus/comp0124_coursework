import logging
import time

import cv2
import numpy as np

from comp0124_coursework.agent.ant import Ant

logger = logging.getLogger(__name__)


class Grid:
    ANT = 1
    FOOD = 2

    def __init__(self, size: tuple) -> None:
        self.size = size

        self.physical_layer = np.zeros(size, dtype=np.int8)
        self.food_pheromone_layer = np.zeros(size, dtype=np.float32)


class BaseEnvironment:
    def __init__(self, size: tuple, nest_location: tuple, logging_interval=40) -> None:
        self.grid = Grid(size)
        self.nest_location = nest_location
        self.ants = set()
        self.logging_interval = logging_interval
        self.avg_step_time = 0
        self.avg_render_time = 0

    def run(
        self,
        no_ants=50,
        no_iterations=1000,
        render=False,
        render_interval=10,
        render_step_wait_time=1,
        iterations_to_save_rendering=[10, 100, 500, 1000],
    ):
        # spawn ants
        for i in range(no_ants):
            rnd_rotation = np.random.uniform(0, 2 * np.pi)

            ant = Ant(f"ant_{i}", self, self.nest_location, rnd_rotation)
            self.ants.add(ant)

        if render:
            cv2.namedWindow("env_vis", cv2.WINDOW_NORMAL)

        for i in range(1, no_iterations + 1):
            if render and i % render_interval == 0:
                start_time = time.time()
                image = self.render()
                cv2.imshow("env_vis", image)

                cv2.waitKey(render_step_wait_time)
                self.avg_render_time += (
                    (time.time() - start_time)
                    * 1000
                    / render_interval
                    / self.logging_interval
                )

            if i in iterations_to_save_rendering:
                if not render or i % render_interval != 0:
                    image = self.render()
                cv2.imwrite(f"render_{i}.png", image * 255)

            self.step()

            if i % self.logging_interval == 0 and i != 0:
                logger.info(
                    f"Step: {i} - Total step time: {self.avg_step_time + self.avg_render_time:.2f}ms (step time: {self.avg_step_time:.2f}ms - render time: {self.avg_render_time:.2f}ms)"
                )
                self.avg_step_time = 0
                self.avg_render_time = 0

        cv2.destroyAllWindows()

    def step(self):
        start_time = time.time()

        self.pheremonone_decay()

        # step ants
        for ant in self.ants:
            ant.step()

        self.avg_step_time += (time.time() - start_time) * 1000 / self.logging_interval

    def render(self):
        # food phereomone layer
        image = np.repeat(
            self.grid.food_pheromone_layer[:, :, np.newaxis], 3, axis=2
        ) * [1, 0.3, 0.3]

        # ants
        image[self.grid.physical_layer == Grid.ANT] = [1.0, 1.0, 1.0]

        return image

    def pheremonone_decay(self):
        self.grid.food_pheromone_layer = np.max(
            np.stack(
                [self.grid.food_pheromone_layer - 0.004, np.zeros(self.grid.size)],
                axis=2,
            ),
            axis=2,
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    env = BaseEnvironment((1000, 1000), (500, 500))
    env.run(render=False)
