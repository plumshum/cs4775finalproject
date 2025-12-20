import subprocess
import time
from memory_profiler import memory_usage
import matplotlib.pyplot as plt


FASTTREE_BIN = "./FastTree"
INPUT_FASTA = "MAPLE_alignment_example.fasta"
OUTPUT_TREE = "MAPLE_alignment_example_fasttree_2_newick.tree"


def run_fasttree():
    """
    Runs FastTree as a subprocess.
    This function is what memory_usage() will monitor.
    """
    subprocess.run(
        [FASTTREE_BIN, "-nt", "-gtr", INPUT_FASTA],
        stdout=open(OUTPUT_TREE, "w"),
        stderr=subprocess.PIPE,
        check=True,
    )


if __name__ == "__main__":
    start = time.time()

    # Track memory while FastTree runs
    mem = memory_usage(
        (run_fasttree, (), {}),
        interval=0.1,     # same as your original code
        include_children=True,
    )

    end = time.time()

    # ---- Plot memory ----
    plt.figure(figsize=(10, 6))
    plt.plot(mem, linewidth=2)
    plt.ylabel("Memory (MB)")
    plt.xlabel("Time (0.1s intervals)")
    plt.title("FastTree Memory Usage")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("fasttree_memory_usage2.pdf")

    # ---- Print stats ----
    print(f"Runtime: {end - start:.2f} seconds")
    print(f"Peak memory: {max(mem):.2f} MB")
    print(f"Output tree: {OUTPUT_TREE}")
    print("Saved memory plot to fasttree_memory_usage.pdf")
