import matplotlib.pyplot as plt

def visualization(historic_record):
    iterations = range(len(historic_record[0]["history"]))

    plt.figure(figsize=(8, 5))

    for entry in historic_record:
        func = entry["function"]
        history = entry["history"]
        plt.plot(iterations, history, label=func, linewidth=2)

    plt.xlabel("Iteraciones")
    plt.ylabel("Valor de fitness")
    plt.title("Convergencia del PSO")

    plt.yscale("log")       
    plt.grid(True)
    plt.legend()

    plt.savefig("tests/performance_viz.png", dpi=300, bbox_inches="tight")

    plt.show()
