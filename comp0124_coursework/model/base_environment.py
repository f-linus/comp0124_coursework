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

        self.obstacle_layer = np.zeros(size, dtype=np.int8)
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
        no_ants=50,
        no_iterations=10000,
        phereomone_decay_rate=0.6,
        logging_interval=50,
        render_to_screen=False,
        render_interval=1,
        render_step_wait_time=1,
        save_rendering_interval=1000,
        extra_iterations_to_save_rendering=[10, 50, 100, 200, 300, 400, 500],
        food_spawn_prob=0.0035,
        food_spawn_size=0.5,
        food_saturation_threshold=2000,
        no_obstacles=15,
        ant_collision=False,
        food_depletion=True,
        pheremone_intensity_coefficient=0.005,
        nest_detection_radius=5,
        no_of_navigation_probes=10,
        probing_distance=5,
        food_detection_box_size=5,
        movement_speed=1,
        rotation_noise=0.2,
        seed=0,
    ):
        self.no_ants = no_ants
        self.nest_location = nest_location
        self.pheremonone_decay_rate = phereomone_decay_rate
        self.logging_interval = logging_interval
        self.render_to_screen = render_to_screen
        self.render_interval = render_interval
        self.render_step_wait_time = render_step_wait_time
        self.save_rendering_interval = save_rendering_interval
        self.extra_iterations_to_save_rendering = extra_iterations_to_save_rendering
        self.food_spawn_prob = food_spawn_prob
        self.food_spawn_size = food_spawn_size
        self.food_saturation_threshold = food_saturation_threshold
        self.ant_collision = ant_collision
        self.food_depletion = food_depletion
        self.pheremone_intensity_coefficient = pheremone_intensity_coefficient
        self.nest_detection_radius = nest_detection_radius
        self.no_of_navigation_probes = no_of_navigation_probes
        self.probing_distance = probing_distance
        self.food_detection_box_size = food_detection_box_size
        self.movement_speed = movement_speed
        self.rotation_noise = rotation_noise

        self.grid = Grid(size)
        self.ants = set()
        self.avg_step_time = 0
        self.avg_render_time = 0
        self.iteration = 1
        self.food_collected = 0
        self.food_collection_log = pd.DataFrame()

        self.last_collection_amount = 0

        self.food_spawning_rng = np.random.default_rng(seed)
        self.obstacle_rng = np.random.default_rng(seed + 1)
        np.random.seed(seed + 2)

        # create obstacles
        self.add_random_obstacles(size, no_obstacles)

        # spawn ants
        self.spawn_ants(no_ants)

        if render_to_screen:
            cv2.namedWindow("env_vis", cv2.WINDOW_NORMAL)

        while self.iteration <= no_iterations:
            self.render_handling()
            self.step()
            self.log_handling()

            self.iteration += 1

        cv2.destroyAllWindows()

    def spawn_ants(self, no_ants: int):
        for i in range(no_ants):
            rnd_rotation = np.random.uniform(0, 2 * np.pi)

            ant = Ant(
                f"ant_{i}",
                self,
                self.nest_location,
                rnd_rotation,
                self.ant_collision,
                self.food_depletion,
                self.pheremone_intensity_coefficient,
                self.nest_detection_radius,
                self.no_of_navigation_probes,
                self.probing_distance,
                self.food_detection_box_size,
                self.movement_speed,
                self.rotation_noise,
            )
            self.ants.add(ant)

    def add_random_obstacles(self, size: tuple, no_obstacles: int):
        for _ in range(no_obstacles):
            x = self.obstacle_rng.integers(0, size[0])
            y = self.obstacle_rng.integers(0, size[1])
            width = self.obstacle_rng.integers(10, 150)
            height = self.obstacle_rng.integers(10, 200 - width)
            cv2.rectangle(
                self.grid.obstacle_layer, (x, y), (x + width, y + height), 1, -1
            )

        # remove obstacles from nest location
        self.grid.obstacle_layer[
            self.nest_location[0] - 40 : self.nest_location[0] + 40,
            self.nest_location[1] - 40 : self.nest_location[1] + 40,
        ] = 0

    def step(self):
        start_time = time.time()

        self.pheremonone_decay()
        self.spawn_food()

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

    def spawn_food(self):
        remaining_food = np.sum(self.grid.food_layer)
        if (
            self.food_spawning_rng.uniform() < self.food_spawn_prob
            and remaining_food < self.food_saturation_threshold
        ):
            x = self.food_spawning_rng.integers(0, self.grid.size[0])
            y = self.food_spawning_rng.integers(0, self.grid.size[1])
            cv2.circle(
                self.grid.food_layer,
                (x, y),
                int(self.food_spawn_size * np.sqrt(self.no_ants)),
                1,
                -1,
            )

            # remove food from food layer where there are obstacles
            self.grid.food_layer[self.grid.obstacle_layer == 1] = 0

    def log_handling(self):
        if self.iteration % self.logging_interval == 0 and self.iteration != 0:
            logger.info(
                f"Step: {self.iteration} - Total step time: {self.avg_step_time + self.avg_render_time:.2f}ms (step time: {self.avg_step_time:.2f}ms - render time: {self.avg_render_time:.2f}ms)"
            )
            logger.info(
                f"Food collected over last {self.logging_interval} steps: {self.food_collected - self.last_collection_amount} (remaining: {np.sum(self.grid.food_layer)})"
            )

            # append to food consumption log
            self.food_collection_log = pd.concat(
                [
                    self.food_collection_log,
                    pd.DataFrame(
                        {
                            "iteration": self.iteration,
                            "food_collected": self.food_collected
                            - self.last_collection_amount,
                            "food_remaining": np.sum(self.grid.food_layer),
                        },
                        index=[0],
                    ),
                ]
            )

            # update CSV file (or crete it if it does not exist)
            self.food_collection_log.to_csv("food_collection_log.csv", index=False)

            self.avg_step_time = 0
            self.avg_render_time = 0
            self.last_collection_amount = self.food_collected

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
        obstacle_layer = np.repeat(
            self.grid.obstacle_layer[:, :, np.newaxis], 3, axis=2
        ) * [0.2, 0.2, 0.2]

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

        image = np.clip(obstacle_layer + home_layer + food_layer, 0, 1)

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

    # load CSV file with results (if it exists)
    try:
        food_collected_per_no_ants = pd.read_csv("food_collected_per_no_ants.csv")
    except FileNotFoundError:
        food_collected_per_no_ants = pd.DataFrame()

    no_of_ants_scenarios = [500]
    no_steps = 40000

    for no_of_ants in no_of_ants_scenarios:
        env = BaseEnvironment()
        env.run(
            size=(600, 600),
            no_iterations=no_steps,
            nest_location=(300, 300),
            no_ants=no_of_ants,
            render_to_screen=False,
        )

        # append to results
        food_collected_per_no_ants = pd.concat(
            [
                food_collected_per_no_ants,
                pd.DataFrame(
                    {
                        "no_of_ants": no_of_ants,
                        "avg_food_collected_per_log_interval": np.mean(
                            env.food_collection_log["food_collected"]
                        ),
                    },
                    index=[0],
                ),
            ]
        )

        print(f"Finished scenario with {no_of_ants} ants")

        # create or update CSV file with results
        food_collected_per_no_ants.to_csv("food_collected_per_no_ants.csv", index=False)
