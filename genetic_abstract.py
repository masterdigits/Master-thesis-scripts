from abc import abstractmethod, ABC
from random import choice,randint,random

class genetic_abstract(ABC):

    def __init__(self):
        self.fitness = self.evaluate_fitness()

    def mutation(self):
        self._perform_mutation()
        self.fitness = self.evaluate_fitness()

    @abstractmethod
    def _perform_mutation(self):
        pass

    @abstractmethod
    def crossover(self):
        pass

    @abstractmethod
    def evaluate_fitness(self):
        pass

class genetic_algorithm:

    def __init__(self, first_population_generator: callable, selection_model: callable, stop_condition: callable, mutation_probability: float = 0.1):
        self.first_generation_func = first_population_generator
        self.selection_model = selection_model
        self.stop_condition = stop_condition
        self.mutation_probability = mutation_probability

    def run(self):
        population = self.first_generation_func()
        population.sort(key = lambda x: x.fitness)
        population_len = len(population)
        i = 0

        while True:
            new_population = self.selection_model(population)
            while len(new_population) != population_len:
                child = choice(population).crossover(choice(population))
                if random() <= self.mutation_probability:
                    child.mutation()
                new_population.append(child)

            population = new_population
            best_one = min(population, key = lambda x: x.fitness)
            print(f"Generacja: {i} => dopasowanie: {best_one.fitness}.")
            #print(best_one)
            i+=1
            if self.stop_condition(best_one, best_one.fitness, i):
                break

def elite_selection_model(generation):
    return sorted(generation, key = lambda x: x.fitness)[:int(len(generation) / 10)]

def contest_selection_model(generation):
    step = int(len(generation) / 10)
    return [max(generation[j:j + step], key = lambda x: x.fitness) 
            for j in range(0,len(generation),step)]

def roulette_selection_model(generation):
    fitness_sum = sum([x.fitness for x in generation])
    roulette_wheel = {f:s for f,s in sorted({subject.fitness / fitness_sum:subject 
                        for subject in generation}.items(), key=lambda x: x[0])}
    ret = []
    for _ in range(10):
        r = random() * fitness_sum
        ret.append(roulette_wheel[min([fitness for fitness in 
                                       list(roulette_wheel.keys()) if fitness > r])])
    return ret

