In `psudeocode.py` we don't need

1. All functions to create time trees : our data doesn't even use non-temporal trees 
```
getPoissonCoeff()
getPartialVecTime()
mergeVectorsTime()
findProbRootTime()
rootVectorTime()
```

2. Lineage Assignment: this is just for post-hoc analysis, and not part of the core tree building

```
assignLineageByReferencePlacement()
seekPlacementOfLineageRefs()
annotateLineageAssignments()
outputLineageAssignments()
```

3. Tree Comparison/Validation: this is to compare the tree we are making with like the "ground truth"
```
prepareTreeComparison()
RobinsonFouldsWithDay1985()
```

4. Support Values/SPRTA : this supports the algorithm's action by using metrics. It's not rlly necessary since we just want to build the tree first
```
assignCoreNumbers()
findPlacementsForSamples()
outputSamplePlacements()
```

5. MAT (Mutation Annotated Tree) System: optimization for large trees using local references. um... we may be using a large tree but lets just ignore optimizations for now.
```
passGenomeListThroughBranch()
mergeMutationLists()
```

6. Tree Utilities (Output formatting): this creates a pretty output but we don't need this. We can just create a simple newick and input it into another software.
```
stringForNode()
makeTreeBinary()
countTips()
writeTSVfile()
```

7. Re-rooting: we can just arbitrarily set a root, kinda like our homoework.  
```
reRootTree()
```


# Minimal Core Implementation: Create phylogenic tree

1. Data structures: Tree class and the genom list representation
    functions:
        -
        -

2. Input:
    functions:
        `collectReference()`: this reads the reference genome
        `readConciseAlignment()`: this reads teh MAPLE format input 

3. Substitution Model Functions:
    - `updateSubMatrix()`: updates the Q matrix using the pseudo-counts. MAPLE uses models such as (JC/GTR/UNREST). Recommended to use JC69 model.
    - we also need a function or maybe a constant to  initialize the mutation rate matrices. 

4. Felsenstein's Algorithm:
    - `probVectTerminalNode()`: creates terminal node probability vectors from sample diffs. we can also *normalize* these terminal vectors within this function or make another function like `updateProbVectTerminalNode()`
    - `mergeVectors()`: combine two partial likelihood vectors
    - `getPartialVec()`: propagates probability vector along branch (this is basiclly the pruning u see in Felsenstein)

5. Sample Placement:
    - `findBestParentForNewSample()` : Finds where to put the new sample (CORE FUNCTION WOOHOO)
    - `appendProbNode()` : Calculate LK cost of appending node to parent
    - `placeSampleOnTree()` : actually executes the placement 

6. Branch optimization:
    - `estimateBranchLengthWithDerivative()` : optimizes branch lengths using derviates 

7. Output:
    - a simplified version of `createNewick()`


# Here's what the pseudo code could look like
```
# MAPLE v0.7.5 — CORE ALGORITHM ONLY (Sequential, No Optimizations)
# Minimal functions needed for basic phylogenetic tree building

def collectReference(fileName):
    """Read reference genome from FASTA file.
    
    Inputs: ['fileName']
    Returns: reference sequence as string/list
    """
    pass

def readConciseAlignment(fileName, extractReference, ref, onlyRef):
    """Read MAPLE concise alignment format (diff format).
    
    Inputs: ['fileName', 'extractReference', 'ref', 'onlyRef']
    Returns: (samples, mutation_lists, metadata)
    """
    pass

def updateSubMatrix(pseudoMutCounts, model, oldMutMatrix):
    """Compute substitution matrix Q from pseudo-counts.
    Start with JC69 model for simplicity.
    
    Inputs: ['pseudoMutCounts', 'model', 'oldMutMatrix']
    Returns: mutMatrix (transition probabilities)
    """
    pass

def probVectTerminalNode(diffs, tree, node):
    """Create terminal node probability vector from sample diffs.
    
    Inputs: ['diffs', 'tree', 'node']
    Returns: probability vector at terminal node
    """
    pass

def updateProbVectTerminalNode(probVect, numMinSeqs):
    """Normalize/post-process terminal node vector.
    
    Inputs: ['probVect', 'numMinSeqs']
    Returns: normalized probability vector
    """
    pass

def getPartialVec(i12, totLen, mutMatrix, errorRate, vect, upNode, flag):
    """Propagate probability vector along branch using Q matrix.
    Core of Felsenstein's pruning algorithm.
    
    Inputs: ['i12', 'totLen', 'mutMatrix', 'errorRate', 'vect', 'upNode', 'flag']
    Returns: propagated probability vector
    """
    pass

def mergeVectors(probVect1, bLen1, fromTip1, probVect2, bLen2, fromTip2, returnLK, isUpDown):
    """Combine two partial likelihood vectors at a node.
    This is the heart of Felsenstein's algorithm.
    
    Inputs: ['probVect1', 'bLen1', 'fromTip1', 'probVect2', 'bLen2', 'fromTip2', 
             'returnLK', 'isUpDown']
    Returns: merged vector (and optionally log-likelihood)
    """
    pass

def areVectorsDifferent(probVect1, probVect2):
    """Check if two probability vectors differ beyond threshold.
    Used to determine if updates are needed.
    
    Inputs: ['probVect1', 'probVect2']
    Returns: True if vectors differ significantly
    """
    pass

def findProbRoot(tree, node):
    """Find root position that maximizes likelihood.
    
    Inputs: ['tree', 'node']
    Returns: best root node and log-likelihood
    """
    pass

def rootVector(tree, node):
    """Compute likelihood vector at root by merging child partials.
    
    Inputs: ['tree', 'node']
    Returns: root probability vector and log-likelihood
    """
    pass

def estimateBranchLengthWithDerivative(probVectP, probVectC, fromTipP, fromTipC, 
                                       mutMatrix, minBL, maxBL, precision, errorRate,
                                       pseudoCountsGlobal, mutMatricesGlobal, 
                                       cumulativeRateGlobal):
    """Optimize branch length using likelihood derivatives.
    
    Inputs: ['probVectP', 'probVectC', 'fromTipP', 'fromTipC', 'mutMatrix', 
             'minBL', 'maxBL', 'precision', 'errorRate', 'pseudoCountsGlobal',
             'mutMatricesGlobal', 'cumulativeRateGlobal']
    Returns: optimized branch length
    """
    pass

def updateBLen(tree, cNode, addToList, nodeList):
    """Apply branch length change and update affected nodes.
    
    Inputs: ['tree', 'cNode', 'addToList', 'nodeList']
    Returns: updated tree structure
    """
    pass

def createNewick(tree, node, binary=False, namesInTree=None):
    """Serialize tree to Newick format (simplified version).
    
    Inputs: ['tree', 'node', 'binary', 'namesInTree']
    Returns: Newick string
    """
    pass
```