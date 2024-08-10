import os

def print_lines_of_code():
    cwd = os.getcwd()
    lines = 0
    num_files = 0
    for root, _, files in os.walk(cwd):
        for f in files:
            if f.endswith(".py") and not "examples" in root and "build" not in root and f != "lines_of_code.py":
                num_files += 1
                path = os.path.join(root, f)
                with open(path, "r") as file:
                    lines += len(file.readlines())
    print("lines of code: ", lines)
    print("number of files: ", num_files)

if __name__ == "__main__":
    print_lines_of_code()
