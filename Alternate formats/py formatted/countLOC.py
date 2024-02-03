import os

def count_lines_of_code(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return sum(1 for line in file)

def run():

    python_files = [f for f in os.listdir('.') if f.endswith('.py')]

    total = 0
    for file in python_files:
        lines = count_lines_of_code(file)
        total += lines
        print(f"{file}: {lines} lines")

    return total

if __name__ == "__main__":
    print(run())
