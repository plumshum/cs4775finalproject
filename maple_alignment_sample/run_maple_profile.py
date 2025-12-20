import subprocess
import time
from memory_profiler import memory_usage
import matplotlib.pyplot as plt

MAPLE_CMD = [
    "pypy3",
    "MAPLE_original.py",
    "--input", "MAPLE_aligned_europe.txt",
    "--output", "./MAPLE_alignent_beautiful_europe_output"
]

def run_maple():
    subprocess.run(MAPLE_CMD, check=True)

if __name__ == "__main__":
    start_time = time.time()

    mem_usage = memory_usage(
        (run_maple, ),
        interval=0.1,
        retval=False
    )

    end_time = time.time()

    print("==== MAPLE PERFORMANCE ====")
    print(f"Total runtime: {end_time - start_time:.2f} seconds")
    print(f"Peak memory: {max(mem_usage):.2f} MB")
    
        # Save to PDF
    plt.figure(figsize=(10, 6))
    plt.plot(mem_usage, linewidth=2)
    plt.ylabel('Memory (MB)')
    plt.xlabel('Time (0.1s intervals)')
    plt.title('Memory Usage')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('memory_usage_original_europe_maple.pdf')  # Saves as PDF
    print(f"Peak memory: {max(mem_usage):.2f} MB")
