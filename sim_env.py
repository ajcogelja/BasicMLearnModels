import random
import pandas as pd
import numpy as np

class Model:
    def __init__(self, base_weights, num_iterations):
        self.weights = base_weights
        self.num_iterations = num_iterations

    def train(self, data):
        """
        - iterates thru training data num_iterations times
        - adjust weights based on correctness of each iteration
        - need to determine which kind of tuning algo i want to use
        """
        print(data)

    #each entity has its own model that governs its movements
class Entity:
    def __init__(self, health, action_cost, x, y, group):
        self.health = health
        self.action_cost = action_cost
        self.x = x
        self.y = y
        self.group = group # a group id signifying if entities are hostile or friendly

    #moves an entity
    def move(self, map_width, map_height, x_move, y_move):
        self.x += x_move
        self.y += y_move
        if self.y >= map_height:
            self.y = map_height - 1
        if self.y < 0:
            self.y = 0
        if self.x >= map_width:
            self.x = map_width - 1
        if self.x < 0:
            self.x = 0
        
class Environment:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.map = []
        self.timestep = 0
        self.entities = []
        self.used_group_id = 0
        default_food_value = 0 #so that tiles do not instrinsically give an reward in our objective function
        for x in range(width):
            col = []
            for y in range(height):
                col.append(default_food_value)
            self.map.append(col)

    def entity_at(self, x, y):
        for v in self.entities:
            if v.x == x and v.y == y:
                return True
        
        return False

    #randomly inserts food into the map at a point
    def spawn_food(self, value):
        x = random.randint(0, self.width - 1)
        y = random.randint(0, self.height - 1)
        self.map[x][y] = value #sets the actual square
    
    #spawns an entity at a random location
    def spawn_entity(self):
        health = 5
        decay_rate = .8
        #check if x or y are occupied first
        x = random.randint(0, self.width - 1)
        y = random.randint(0, self.height - 1)
        while(self.entity_at(x, y)):
            x = random.randint(0, self.width - 1)
            y = random.randint(0, self.height - 1)
        new_ent = Entity(health, decay_rate, x, y, self.used_group_id)
        self.used_group_id += 1 #for now all entities have diff group id's
        self.entities.append(new_ent)

    def simulate(self):
        #spawn food, progress anything else about environment
        food_val = 4
        self.spawn_food(food_val) #add a new food
        self.timestep += 1 #increase the current time
        #move every entity each iteration
        #each entity has a diff model determining its action
        #model should be perceptron/svm style I think
        for e in self.entities:
            x_move = 0
            y_move = 0
            e.move(self.width, self.height, x_move, y_move)
        
if __name__ == "__main__":
    env = Environment(20, 20)