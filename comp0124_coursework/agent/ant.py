import numpy as np


class Ant:
    def __init__(self, name, env, position: tuple, rotation=0) -> None:
        self.name = name
        self.env = env
        self.pos = position
        self.rotation = rotation
        self.carrying_food = False

    def step(self):
        current_pos_discrete = (
            int(self.pos[0]),
            int(self.pos[1]),
        )

        # update pheromone layer
        self.env.grid.food_pheromone_layer[current_pos_discrete] = 1

        # random change in rotation
        self.rotation += np.random.uniform(-0.5, 0.5)
        self.rotation = self.rotation % (2 * np.pi)

        # move forward
        desired_new_pos = (
            self.pos[0] + np.cos(self.rotation),
            self.pos[1] + np.sin(self.rotation),
        )

        # convert to discrete positions
        desired_pos_discrete = (
            int(desired_new_pos[0]),
            int(desired_new_pos[1]),
        )

        # check if new position is within bounds and not occupied
        if (
            0 <= desired_pos_discrete[0] < self.env.grid.size[0]
            and 0 <= desired_pos_discrete[1] < self.env.grid.size[1]
        ) and self.env.grid.physical_layer[desired_pos_discrete] == 0:
            # update position
            self.pos = desired_new_pos

            # update grid
            self.env.grid.physical_layer[current_pos_discrete] = 0
            self.env.grid.physical_layer[desired_pos_discrete] = 1

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        return isinstance(other, Ant) and self.name == other.name
