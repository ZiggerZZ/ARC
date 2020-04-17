from arc import *

if __name__== "__main__":
    n = 4
    dataset = 'training'
    print("Depth", n, dataset)
    print()
    all_solutions = solve_all_tasks(n, dataset)
    for key, value in all_solutions.items():
        print(key)
        pp.pprint(dicts(value[0]))
        print()

    print(f"Solved {len(all_solutions)}")





