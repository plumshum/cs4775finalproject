Steps to get final phylogentic tree output

1. Find fasta files of aligned genomes for your diesae (i.e: Covid)

2. Create MAPLE alignment file: python3 createMapleFile.py --path ./ --fasta aligned.fasta 
- NOTE: (make sure fasta file is in `maple_alignment` folder)
- NOTE: `--path ./` is where your fasta file should be located
- NOTE: `--fasta <aligned.fasta>` is where your output file will be located AND named `<aligned.fasta>`

3. Use these aligned MAPLE file to create newick tree by running `psudeocode.py`
- TODO: add the correct command once code is ready

3. Use these aligned MAPLE file to create newick trees by running MAPLEv0.7.5.py:
`python3 MAPLEv0.7.5.py --input MAPLE_alignment_example.txt --output ./MAPLE_outputFilePrefix_tree.tree`
- NOTE: if you need faster compilation time then use `pypy` instead of `python3`

4. Create a phylogenetic Tree. Using a simple matplotlib graph you 
can run `plotNjTree.py`. Or you can use the online phylogenetic tree called [Taxonium](https://taxonium.org/) and choose the newick file. 

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


### Concepts of MAPLE Algorithm:
`likelihood-based phylogenetic analysis of epidemiological genomic datasets at unprecedented scales`
- likelihood-based: probabilities
- phylogenetic: evolutionary relationships (in this case: via phylogenetic trees)
- epidemiological: diseases

'Maple aligned files': how is different from VCF?

- 'VCF': V