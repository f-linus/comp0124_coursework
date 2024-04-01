import numpy as np


class Ant:
    def __init__(
        self,
        name,
        env,
        position: tuple,
        rotation=0,
        ant_collision=False,
        food_depletion=True,
        pheremone_intensity_coefficient=0.005,
        nest_detection_radius=5,
        no_of_navigation_probes=10,
        probing_distance=5,
        food_detection_box_size=5,
        movement_speed=1,
        rotation_noise=0.2,
    ) -> None:
        self.name = name
        self.env = env
        self.pos = position
        self.rotation = rotation
        self.ant_collision = ant_collision
        self.food_depletion = food_depletion
        self.pheremone_intensity_coefficient = pheremone_intensity_coefficient
        self.nest_detection_radius = nest_detection_radius
        self.probing_distance = probing_distance
        self.no_of_navigation_probes = no_of_navigation_probes
        self.food_detection_box_size = food_detection_box_size
        self.movement_speed = movement_speed
        self.rotation_noise = rotation_noise

        self.pos_discrete = (
            int(self.pos[0]),
            int(self.pos[1]),
        )
        self.carrying_food = False
        self.last_nest_visit = env.iteration
        self.found_food_at = None

    def determine_ideal_direction(self):
        probing_directions = np.linspace(
            self.rotation - np.pi / 2,
            self.rotation + np.pi / 2,
            self.no_of_navigation_probes,
        )

        probing_positions_discrete = [
            (
                int(self.pos[0] + self.probing_distance * np.cos(direction)),
                int(self.pos[1] + self.probing_distance * np.sin(direction)),
            )
            for direction in probing_directions
        ]

        if self.carrying_food:
            # probe food pheromone layer
            concentrations = []
            for pos in probing_positions_discrete:
                if (
                    0 <= pos[0] < self.env.grid.size[0]
                    and 0 <= pos[1] < self.env.grid.size[1]
                ):
                    concentrations.append(self.env.grid.home_pheromone_layer[pos])
                else:
                    concentrations.append(0)

            # choose direction with highest concentration
            if np.max(concentrations) == 0:
                new_rotation = self.rotation
            else:
                new_rotation = probing_directions[np.argmax(concentrations)]
        else:
            # probe home pheromone layer and food layer for food
            concentrations = []
            found_food = None
            for pos in probing_positions_discrete:
                if (
                    0 <= pos[0] < self.env.grid.size[0]
                    and 0 <= pos[1] < self.env.grid.size[1]
                ):
                    concentrations.append(self.env.grid.food_pheromone_layer[pos])
                    if self.env.grid.food_layer[pos] == 1:
                        found_food = pos
                        break
                else:
                    concentrations.append(0)

            if found_food is not None:
                # if food is found, move towards it
                new_rotation = np.arctan2(
                    found_food[1] - self.pos[1], found_food[0] - self.pos[0]
                )
            else:
                # choose direction with highest concentration
                if np.max(concentrations) == 0:
                    new_rotation = self.rotation
                else:
                    new_rotation = probing_directions[np.argmax(concentrations)]
        return new_rotation

    def move(self):
        # depending on whether the ant is carrying food or not, move towards the nest or the food
        # probe the relevant pheromone layer to decide the direction
        new_rotation = self.determine_ideal_direction()

        # add random noise to direction
        new_rotation += np.random.uniform(-self.rotation_noise, self.rotation_noise)

        # move to new position if possible, if not possible flip direction
        new_position = (
            self.pos[0] + self.movement_speed * np.cos(new_rotation),
            self.pos[1] + self.movement_speed * np.sin(new_rotation),
        )
        new_position_discrete = (
            int(new_position[0]),
            int(new_position[1]),
        )

        # check if new position is within grid
        if not (
            (0 <= new_position_discrete[0] < self.env.grid.size[0])
            and (0 <= new_position_discrete[1] < self.env.grid.size[1])
        ):
            self.rotation = self.rotation + np.pi
            return

        # check if new position is not occupied by an obstacle
        if self.env.grid.obstacle_layer[new_position_discrete] == 1:
            self.rotation = self.rotation + np.pi
            return

        # check if new position is not occupied by another ant
        if (
            self.env.grid.ant_layer[new_position_discrete] == 1
            and self.pos_discrete != new_position_discrete
            and self.ant_collision
        ):
            return

        # remove ant from old position
        self.env.grid.ant_layer[self.pos_discrete] = 0

        self.pos = new_position
        self.pos_discrete = new_position_discrete
        self.rotation = new_rotation

        # add ant to new position
        self.env.grid.ant_layer[self.pos_discrete] = 1
        return

    def deposit_pheromone(self):
        if self.carrying_food:
            pheremone_concentration = 1000 * np.exp(
                self.pheremone_intensity_coefficient
                * -1
                * (self.env.iteration - self.found_food_at)
            )
            self.env.grid.food_pheromone_layer[self.pos_discrete] = np.max(
                [
                    self.env.grid.food_pheromone_layer[self.pos_discrete],
                    pheremone_concentration,
                ]
            )
        else:
            pheremone_concentration = 1000 * np.exp(
                self.pheremone_intensity_coefficient
                * -1
                * (self.env.iteration - self.last_nest_visit)
            )
            self.env.grid.home_pheromone_layer[self.pos_discrete] = np.max(
                [
                    self.env.grid.home_pheromone_layer[self.pos_discrete],
                    pheremone_concentration,
                ]
            )

    def detect_food(self):
        if self.carrying_food:
            return

        # check if were on a food source
        if self.env.grid.food_layer[self.pos_discrete] == 1:
            self.carrying_food = True
            self.found_food_at = self.env.iteration

            if self.food_depletion:
                self.env.grid.food_layer[self.pos_discrete] = 0

            # flip rotation
            self.rotation = self.rotation + np.pi
            return

    def detect_nest(self):
        if self.carrying_food:
            nest_dist = np.linalg.norm(
                np.array(self.pos) - np.array(self.env.nest_location)
            )

            if nest_dist <= self.nest_detection_radius:
                self.carrying_food = False
                self.last_nest_visit = self.env.iteration
                self.rotation = self.rotation + np.pi
                self.env.food_collected += 1

    def step(self):
        self.move()
        self.detect_food()
        self.deposit_pheromone()
        self.detect_nest()

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        return isinstance(other, Ant) and self.name == other.name
