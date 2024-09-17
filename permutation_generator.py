import random

# Function to generate a random permutation of numbers from 1 to 30
def generate_permutation():
    numbers = list(range(1, 31))
    random.shuffle(numbers)
    return numbers

# File to write the permutations to
filename = "random_permutations.txt"

with open(filename, "w") as file:
    for _ in range(60):
        permutation = generate_permutation()
        # Convert the permutation list to a space-separated string
        permutation_str = ' '.join(map(str, permutation))
        # Write the permutation to the file
        file.write(permutation_str + '\n')

print(f"60 random permutations have been written to {filename}")
