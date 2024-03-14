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
        self.home_pheromone_layer = np.zeros(size, dtype=np.float32)
        self.food_pheromone_layer = np.zeros(size, dtype=np.float32)


class BaseEnvironment:
    def __init__(self):
        pass

    def run(
        self,
        size: tuple,
        nest_location: tuple,
        logging_interval=40,
        no_ants=50,
        no_iterations=10000,
        render_to_screen=False,
        render_interval=10,
        render_step_wait_time=1,
        save_rendering_interval=1000,
        extra_iterations_to_save_rendering=[10, 100, 500],
    ):
        self.grid = Grid(size)
        self.nest_location = nest_location
        self.ants = set()
        self.logging_interval = logging_interval
        self.avg_step_time = 0
        self.avg_render_time = 0
        self.iteration = 1
        self.pheremonone_decay_rate = 1

        self.render_to_screen = render_to_screen
        self.render_interval = render_interval
        self.render_step_wait_time = render_step_wait_time
        self.save_rendering_interval = save_rendering_interval
        self.extra_iterations_to_save_rendering = extra_iterations_to_save_rendering

        # spawn ants
        for self.iteration in range(no_ants):
            rnd_rotation = np.random.uniform(0, 2 * np.pi)

            ant = Ant(f"ant_{self.iteration}", self, self.nest_location, rnd_rotation)
            self.ants.add(ant)

        # spawn some food
        self.grid.physical_layer[200:210, 200:210] = Grid.FOOD

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
            self.avg_step_time = 0
            self.avg_render_time = 0

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
        image[self.grid.physical_layer == Grid.ANT] = [1.0, 1.0, 1.0]

        # food
        image[self.grid.physical_layer == Grid.FOOD] = [0.0, 0.0, 1.0]

        return image


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    env = BaseEnvironment()
    env.run((600, 600), (300, 300), render_to_screen=False)
