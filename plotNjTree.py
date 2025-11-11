'''
(Green_monkey:0.013500, ((Baboon:0.008042, (Rhesus:0.004991, Crab_eating_macaque:0.004991):0.003000):0.019610, (Gibbon:0.022270, (Orangutan:0.018940, (Gorilla:0.008964, (Bonobo:0.007840, (Human:0.006550, Chimp:0.006840):0.001220):0.001000):0.009693):0.003471):0.022040):0.013500);
'''

# use Phylo.read + Phylo.draw from BioPython
from Bio import Phylo
import matplotlib.pyplot as plt

def plot_tree(tree_file, output_file):
    tree = Phylo.read(tree_file, "newick")
    fig = plt.figure(figsize=(50, 60), dpi=200)  # Set the figure size and DPI
    axes = fig.add_subplot(1, 1, 1)
    Phylo.draw(tree, axes=axes, do_show = False)
    
    plt.tight_layout(pad=2.0) # margins and spacing
    
    plt.savefig(output_file)
    plt.close()
    print(f"Tree plot saved to {output_file}")

if __name__ == "__main__":
    plot_tree("./FinaProject/MAPLE_outputFilePrefix_tree.tree_tree.tree", "./FinaProject/MAPLE_tree3.png")


# TODO: ask chat how to use biopython for this