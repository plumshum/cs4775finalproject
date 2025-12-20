Here are the steps to run our re-implemented MAPLE algorithm on the provided example data:
1. a MAPLE alignment file is provided in the `maple_alignemnt` folder called `MAPLE_alignment_example.txt`.
2. `psudeocode.py` contains our re-implemented MAPLE algorithm. 

### Steps to run MAPLE re-implementation:

Pre-requisites:

1. [pypy3](https://pypy.org/download.html) or python3 installed. We recommend using `pypy3` for faster execution.

2. Required python packages installed. You can install them via:
`pip install -r requirements.txt`
*NOTE:* if you are using `pypy3`, make sure to use `pypy3 -m pip install -r requirements.txt`

Actual Steps:

1. Use these aligned MAPLE file to create newick tree: 

 - To change the input file, you must modify the variable `inputFile` in `psudeocode.py` around line 2713 to point to your desired MAPLE alignment file. ```    inputFile = "./maple_alignment_sample/maple_alignment_example.txt"```

- To change the output file name, you can modify the variable `outputTreeFile` in `psudeocode.py` around line 2890 to your desired output newick file name.```        outputFile = "output_alignement_example_tree.newick" ```
- Our code will also create a plot of memory usage. To change the output name of the memory usage plot, the `plt.savefig` line: ```    plt.savefig('maple_alignment_example_memory_usage.pdf')  # Saves as PDF```

With these changes made, you can run the re-implemented MAPLE algorithm using:

`python3 psudeocode.py`

- NOTE: if you need faster compilation time then use `pypy` instead of `python3`


4. Create a phylogenetic Tree. Using a simple matplotlib graph you 
can run `plotNjTree.py`. Or you can use the online phylogenetic tree called [Taxonium](https://taxonium.org/) and choose the newick file.


### Example Data Source:

COVID alignment files source : https://docs.nextstrain.org/projects/ncov/en/latest/reference/remote_inputs.html

### Understanding our re-implemented MAPLE algorithm:
Our code is divided into three steps:
1. Read input: we take in the aligned MAPLE files and return proper variables needed for our MAPLE algorithm.
2. Intialize model and likelihood structures.
- NOTE: we only use a Jukes-Cantor model. The original MAPLE algorithm uses `JC`, `GTR`, and `UNREST`.
3. Build/extend tree step-by-step
    a. Build terminal-node likelihood vector
    b. For each candidate branch, we get partial vectors and merge them
    c. Insert sample into the tree, and update local paths
4. Output: creating the final newick tree


### Archived: 

These are parts of the MAPLE algorithm that we have decided to not implement in our re-implementation for simplicity. These commands should still work though!

1. Find fasta files of aligned genomes for your diesease (i.e: Covid)

2. Create MAPLE alignment file: 
`python3 createMapleFile.py --path ./ --fasta aligned.fasta`
- NOTE: (make sure fasta file is in `maple_alignment` folder)
- NOTE: `--path ./` is where your fasta file should be located
- NOTE: `--fasta <aligned.fasta>` is where your output file will be located AND named `<aligned.fasta>`

3. Use these aligned MAPLE file to create newick tree by running `psudeocode.py`