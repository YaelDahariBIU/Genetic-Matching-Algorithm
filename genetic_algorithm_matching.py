import random
import datetime
import os
import matplotlib.pyplot as plt


# Function to read preferences from a file
def read_preferences(file_path):
    men_preferences = []
    women_preferences = []
    with open(file_path, 'r') as file:
        lines = file.readlines()

        # Assuming the first 30 lines are men's preferences
        for i in range(30):
            preferences = list(map(int, lines[i].strip().split()))
            men_preferences.append([p - 1 for p in preferences])  # Convert to 0-based index

        # The next 30 lines are women's preferences
        for i in range(30, 60):
            preferences = list(map(int, lines[i].strip().split()))
            women_preferences.append([p - 1 for p in preferences])  # Convert to 0-based index

    return men_preferences, women_preferences


# Function to calculate fitness of a chromosome
def calculate_fitness(chromosome, men_preferences, women_preferences):
    total_satisfaction = 0
    n = len(chromosome)  # Number of pairs, should be 30 in this case

    for man in range(n):
        woman = chromosome[man]

        # Men's satisfaction
        men_preference_score = (n - men_preferences[man].index(woman))

        # Women's satisfaction
        women_preference_score = (n - women_preferences[woman].index(man))

        # Average satisfaction for this couple
        couple_satisfaction = (men_preference_score + women_preference_score) / 2.0
        total_satisfaction += couple_satisfaction

    # Overall average satisfaction
    average_satisfaction = total_satisfaction / n
    return average_satisfaction


# Function to perform crossover between two parents
"""
Randomly selects two crossover points.
Copies the segment between these points from parent1 to the child.
Fills in the remaining positions with values from parent2, ensuring no duplicates.
This method helps combine genetic material from both parents while maintaining the order and uniqueness of elements,
which is crucial for matching problems where each individual can only be paired once.
"""


def crossover(parent1, parent2):
    size = len(parent1)
    start, end = sorted(random.sample(range(size), 2))
    child = [None] * size
    child[start:end] = parent1[start:end]
    p2_index = 0
    for i in range(size):
        if child[i] is None:
            while parent2[p2_index] in child:
                p2_index += 1
            child[i] = parent2[p2_index]
    return child


# Function to perform mutation on a chromosome
def mutate(chromosome, mutation_rate=0.1):
    if random.random() < mutation_rate:
        idx1, idx2 = random.sample(range(len(chromosome)), 2)
        chromosome[idx1], chromosome[idx2] = chromosome[idx2], chromosome[idx1]


# Function to perform tournament selection.
# To choose the individual with the highest fitness from the selected k individuals.
def tournament_selection(population, fitnesses, k=5):
    selected = random.sample(list(zip(population, fitnesses)), k)
    return max(selected, key=lambda x: x[1])[0]

# Function to adjust mutation rate or add new random solutions
def adjust_population_diversity(population, fitnesses, mutation_rate, threshold=1.0):
    max_fitness = max(fitnesses)
    min_fitness = min(fitnesses)

    if max_fitness - min_fitness < threshold:
        # Increase mutation rate
        mutation_rate = min(0.5, mutation_rate + 0.05)

        # Add new random solutions
        num_new_solutions = len(population) // 5
        new_solutions = [random.sample(range(30), 30) for _ in range(num_new_solutions)]
        population.extend(new_solutions)

    return mutation_rate

# Main genetic algorithm function
def genetic_algorithm(men_preferences, women_preferences, pop_size=180, generations=100, mutation_rate=0.1):
    population = [random.sample(range(30), 30) for _ in range(pop_size)]
    best_solution = None
    best_fitness = float('-inf')

    max_fitness_history = []
    min_fitness_history = []
    avg_fitness_history = []

    for generation in range(generations):
        fitnesses = [calculate_fitness(chromosome, men_preferences, women_preferences) for chromosome in population]
        max_fitness = max(fitnesses)
        min_fitness = min(fitnesses)
        avg_fitness = sum(fitnesses) / len(fitnesses)

        max_fitness_history.append(max_fitness)
        min_fitness_history.append(min_fitness)
        avg_fitness_history.append(avg_fitness)

        best_idx = fitnesses.index(max_fitness)

        if fitnesses[best_idx] > best_fitness:
            best_fitness = fitnesses[best_idx]
            best_solution = population[best_idx]

        new_population = [best_solution]  # Ensure the best solution is carried over
        while len(new_population) < pop_size:
            parent1 = tournament_selection(population, fitnesses)
            parent2 = tournament_selection(population, fitnesses)
            child = crossover(parent1, parent2)
            mutate(child, mutation_rate)
            new_population.append(child)

        population = new_population

        # Adjust population diversity
        new_mutation_rate = adjust_population_diversity(population, fitnesses, mutation_rate)
        if new_mutation_rate == mutation_rate:
            if generation % 20 == 0:
                mutation_rate = min(0.5, mutation_rate + 0.05)
        else:
            mutation_rate = new_mutation_rate

    # Plotting the fitness history
    fig = plt.figure(figsize=(10, 6))
    fig.canvas.manager.set_window_title('Matching Genetic Algorithm Fitness History')
    plt.plot(max_fitness_history, label='Max Fitness')
    plt.plot(min_fitness_history, label='Min Fitness')
    plt.plot(avg_fitness_history, label='Avg Fitness')
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.title('Fitness over Generations')
    plt.legend()
    max_fitness_label = f'Max Fitness: {max_fitness_history[-1]:.2f}'
    plt.annotate(max_fitness_label, xy=(generations - 1, max_fitness_history[-1]),
                 xytext=(generations - 10, max_fitness_history[-1] + 2),
                 arrowprops=dict(facecolor='black', arrowstyle='->'),
                 fontsize=12, bbox=dict(facecolor='white', alpha=0.6))
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    filename = f"graphs/GA_fitness_history_{timestamp}.png"
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    plt.savefig(filename)
    plt.show()

    return best_solution


if __name__ == "__main__":
    # Read preferences from the file
    men_preferences, women_preferences = read_preferences('GA_input.txt')

    # Run the genetic algorithm
    best_solution = genetic_algorithm(men_preferences, women_preferences)
    print("Best Solution:", best_solution)
    print("Fitness:", calculate_fitness(best_solution, men_preferences, women_preferences))
