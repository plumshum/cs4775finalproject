Steps to get final phylogentic tree output

1. Find fasta files of aligned genomes for your diesae (i.e: Covid)
2. Create MAPLE alignment file: python3 createMapleFile.py --path ./ --fasta aligned.fasta (make sure fasta file is in maple_alignment folder)
3. Use these aligned MAPLE file to create newick trees by running MAPLEv0.7.5.py:

`python3 MAPLEv0.7.5.py --input MAPLE_alignment_example.txt --output ./MAPLE_outputFilePrefix_tree.tree`
- if you need faster compilation time then use `pypy` instead of `python3`

4. Create a phylogenetic Tree. Using a simple matplotlib graph you 
can run `plotNjTree.py`. Or you can use the online phylogenetic tree called [Taxonium](https://taxonium.org/) and choose the newick file. 
