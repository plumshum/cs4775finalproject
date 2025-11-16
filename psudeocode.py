# MAPLE v0.7.5 — Core pipeline & model stubs (sequential version)
# This file mirrors the major sections/functions from MAPLE and documents
# expected inputs/outputs without implementing full logic.
#
# High‑level, sequential CORE pipeline (no parallelism):
#
# 1) Read input
#    - collectReference(fileName) -> reference string/list
#    - readConciseAlignment(fileName, extractReference, ref, onlyRef)
#         => returns (samples, per‑sample mutation lists vs reference, metadata)
#
# 2) Initialize model & likelihood structures
#    - updateSubMatrix(pseudoMutCounts, model, oldMutMatrix) -> mutMatrix
#    - (optionally) error‑rate setup; here we keep a scalar errorRate
#
# 3) Build/extend tree by incremental placement
#    For each sample:
#      a) Build terminal-node likelihood vector:
#         probVectTerminalNode(diffs, tree, node)
#         updateProbVectTerminalNode(probVect, numMinSeqs)
#      b) For each candidate branch:
#         getPartialVec(i12, totLen, mutMatrix, errorRate, vect, upNode, flag)
#         mergeVectors(probVect1,bLen1,fromTip1, probVect2,bLen2,fromTip2, returnLK,isUpDown)
#         (keep best placement by log-likelihood)
#      c) Insert sample; update local paths:
#         updateBLen(tree, cNode, addToList, nodeList)
#
# 4) Optimize branch lengths/topology (optional in 'core-only'):
#    - estimateBranchLengthWithDerivative(...)
#    - reRootTree(...), rootVector(...), findProbRoot(...)
#
# 5) Output
#    - createNewick(tree, root, ...)
#    - writeTSVfile(tree, root, file, namesInTree)
#
# Substitution models (used by updateSubMatrix):
#  - JC: single off‑diagonal rate; equal base freqs; normalize mean rate = 1.
#  - GTR: relative rate matrix r_{ij} (i<j) and base freqs π; Q = S*diag(π), rows sum to 0.
#  - UNREST: full 12‑rate model with independent r_{ij}; rows sum to 0, normalize.
#
# NOTE: This is a documentation scaffold. All functions 'pass'.

def collectReference(fileName):
    """ Read a FASTA-like reference from file and return reference sequence.

Inputs: ['fileName']
Outputs: see return docs in MAPLE source.
"""
    pass


def readConciseAlignment(fileName, extractReference, ref, onlyRef):
    """ Read MAPLE 'concise alignment' format; optionally extract embedded reference.

Inputs: ['fileName', 'extractReference', 'ref', 'onlyRef']
Outputs: see return docs in MAPLE source.
"""
    pass


# Constants needed for updeSubMatrix
from math import log
# rootFreqs=[0.25,0.25,0.25,0.25]
# rootFreqsLog=[log(0.25),log(0.25),log(0.25),log(0.25)]
THRESHOLD = 1e-6
def updateSubMatrix(model, oldMutMatrix):
    """ Re/compute substitution matrix Q (and derived transition matrix) from JC model
    - First update with mutation matrix
    - Normalize 
    - Return updated mutation matrix
    For now, model must be a JC (Jukes-Cantor)
    Note: JC model assumes equal base frequencies and equal substitution rates, so we do not need `pseudoCounts`, which is used by the original MAPLE algorithm

Inputs: `['model'="JC", 'oldMutMatrix']`
Outputs: bool: True if the mutation matrix `oldMutMatrix` was updated, False otherwise
Exception: Exception if the given model is not JC
"""
    mutMatrix = [[0.0] * len(oldMutMatrix) for _ in range(len(oldMutMatrix))]
    if model != "JC":
        print("Error: Only JC model is implemented.")
        raise Exception("exit")
    # Implement JC model update
    for i in range(len(oldMutMatrix)):
        for j in range(len(oldMutMatrix)):
            if i == j:
                #TODO: use math.pow
                mutMatrix[i][j] = 1 - 3 * 0.25 # fixed value for JC model
            else:
                mutMatrix[i][j] = 0.25
    print(f"MutMatrix after JC: {mutMatrix}")
    
    # Normalize 
    for i in range(len(mutMatrix)):
        row_sum = sum(mutMatrix[i])
        for j in range(len(mutMatrix)):
            mutMatrix[i][j] /= row_sum
    
    print(f"Normalized mut matrix: {mutMatrix}")
    
    # Update oldMutMatrix by checking if there are significant changes
    # We consider a significant change if the difference between mutMatrix and oldMutMatrix elements is greater than a threshold of THRESHOLD
    for i in range(len(mutMatrix)):
        for j in range(len(mutMatrix)):
            if abs(mutMatrix[i][j] - oldMutMatrix[i][j]) > THRESHOLD:
                oldMutMatrix[i][j] = mutMatrix[i][j]
                print(f"Updated oldMutMatrix at [{i}][{j}] to {mutMatrix[i][j]}")
                return True
    
    return False

def probVectTerminalNode(diffs, tree, node):
    """ Create a terminal-node probability vector from sample/reference diffs at a node.

Inputs: ['diffs', 'tree', 'node']
Outputs: see return docs in MAPLE source.
"""
    pass


def updateProbVectTerminalNode(probVect, numMinSeqs):
    """ Post-process/normalize terminal-node vector given min sequence counts.

Inputs: ['probVect', 'numMinSeqs']
Outputs: see return docs in MAPLE source.
"""
    pass


def getPartialVec(i12, totLen, mutMatrix, errorRate, vect, upNode, flag):
    """ Propagate a probability vector across a branch using the substitution model.

Inputs: ['i12', 'totLen', 'mutMatrix', 'errorRate', 'vect', 'upNode', 'flag']
Outputs: see return docs in MAPLE source.
"""
    pass


def mergeVectors(probVect1, bLen1, fromTip1, probVect2, bLen2, fromTip2, returnLK, isUpDown):
    """ Combine two partial-likelihood vectors meeting at a node/edge; optionally return LK.

Inputs: ['probVect1', 'bLen1', 'fromTip1', 'probVect2', 'bLen2', 'fromTip2', 'returnLK', 'isUpDown']
Outputs: see return docs in MAPLE source.
"""
    pass


def findProbRoot(tree, node):
    """ Search for root that maximizes overall likelihood given current vectors/BLens.

Inputs: ['tree', 'node']
Outputs: see return docs in MAPLE source.
"""
    pass


def rootVector(tree, node):
    """ Compute likelihood vector at (candidate) root by merging child partials.

Inputs: ['tree', 'node']
Outputs: see return docs in MAPLE source.
"""
    pass


def estimateBranchLengthWithDerivative(probVectP, probVectC, fromTipP, fromTipC, mutMatrix, minBL, maxBL, precision, errorRate, pseudoCountsGlobal, mutMatricesGlobal, cumulativeRateGlobal):
    """ Estimate branch length maximizing likelihood; optionally use derivatives.

Inputs: ['probVectP', 'probVectC', 'fromTipP', 'fromTipC', 'mutMatrix', 'minBL', 'maxBL', 'precision', 'errorRate', 'pseudoCountsGlobal', 'mutMatricesGlobal', 'cumulativeRateGlobal']
Outputs: see return docs in MAPLE source.
"""
    pass


def updateBLen(tree, cNode, addToList, nodeList):
    """ Commit a branch-length change and update impacted node lists/vectors.

Inputs: ['tree', 'cNode', 'addToList', 'nodeList']
Outputs: see return docs in MAPLE source.
"""
    pass


def areVectorsDifferent(probVect1, probVect2):
    """ Return True if two probability vectors differ beyond thresholds.

Inputs: ['probVect1', 'probVect2']
Outputs: see return docs in MAPLE source.
"""
    pass


def createNewick(tree, node, binary, namesInTree, includeMutationList, estimateMAT, networkOutput, sprtaOn, minSupport, count0BLenNodesOnce, includeSupports, keepInputIQtreeSupports, aBayesPlusOn, performLineageAssignmentByRefPlacement):
    """ Serialize tree to Newick/Nexus; optionally include supports, MAT, and metadata.

Inputs: ['tree', 'node', 'binary', 'namesInTree', 'includeMutationList', 'estimateMAT', 'networkOutput', 'sprtaOn', 'minSupport', 'count0BLenNodesOnce', 'includeSupports', 'keepInputIQtreeSupports', 'aBayesPlusOn', 'performLineageAssignmentByRefPlacement']
Outputs: see return docs in MAPLE source.
"""
    pass

# Hannah's NOTE: at this point most functions below aren't necessary

def writeTSVfile(tree, node, file, namesInTree):
    """ Emit a TSV of per-node attributes (mutations, supports, etc.).

Inputs: ['tree', 'node', 'file', 'namesInTree']
Outputs: see return docs in MAPLE source.
"""
    pass


def assignCoreNumbers(tree, root, numCores):
    """ Compute 'core' numbers for nodes (graph-theoretic metric) for downstream use.

Inputs: ['tree', 'root', 'numCores']
Outputs: see return docs in MAPLE source.
"""
    pass


def findPlacementsForSamples(tree, t1, distances, numCores):
    """ Evaluate candidate placements for a set of samples; no topology change if 'find-only'.

Inputs: ['tree', 't1', 'distances', 'numCores']
Outputs: see return docs in MAPLE source.
"""
    pass


def outputSamplePlacements(outputFile, tree, root):
    """ Write best placements (node, edge, distance, LK) for samples to file.

Inputs: ['outputFile', 'tree', 'root']
Outputs: see return docs in MAPLE source.
"""
    pass


def assignLineageByReferencePlacement(tree, t1, lineageRefData, numCores):
    """ Assign lineages by placing lineage-reference genomes onto tree.

Inputs: ['tree', 't1', 'lineageRefData', 'numCores']
Outputs: see return docs in MAPLE source.
"""
    pass


def seekPlacementOfLineageRefs(tree, t1, lineageRefData, numCores, findPlacementOnly):
    """ Find placements for lineage references; optionally multi-core.

Inputs: ['tree', 't1', 'lineageRefData', 'numCores', 'findPlacementOnly']
Outputs: see return docs in MAPLE source.
"""
    pass


def annotateLineageAssignments(tree, root):
    """ Annotate nodes with lineage assignment labels/metadata.

Inputs: ['tree', 'root']
Outputs: see return docs in MAPLE source.
"""
    pass


def outputLineageAssignments(outputFile, tree, root):
    """ Write lineage assignment TSV (node, lineage, support).

Inputs: ['outputFile', 'tree', 'root']
Outputs: see return docs in MAPLE source.
"""
    pass


def stringForNode(tree, nextNode, nameNode, distB, binary, nameInternalNode, count0BLenNodesOnce, namesInTree, performLineageAssignmentByRefPlacement):
    """ Return display name for node (tip/internal) honoring options and naming tables.

Inputs: ['tree', 'nextNode', 'nameNode', 'distB', 'binary', 'nameInternalNode', 'count0BLenNodesOnce', 'namesInTree', 'performLineageAssignmentByRefPlacement']
Outputs: see return docs in MAPLE source.
"""
    pass


def makeTreeBinary(tree, root):
    """ Convert multifurcations to a binary tree (introduce zero-length internals).

Inputs: ['tree', 'root']
Outputs: see return docs in MAPLE source.
"""
    pass


def reRootTree(tree, root, sample, reRootAtInternalNode):
    """ Find and set a new root (by name or by LK optimization).

Inputs: ['tree', 'root', 'sample', 'reRootAtInternalNode']
Outputs: see return docs in MAPLE source.
"""
    pass


def countTips(tree, node):
    """ Count number of descendant tips from node.

Inputs: ['tree', 'node']
Outputs: see return docs in MAPLE source.
"""
    pass


def getPoissonCoeff(BL, t, mu):
    """ Helper to compute Poisson-like coefficients used in time/LK calcs.

Inputs: ['BL', 't', 'mu']
Outputs: see return docs in MAPLE source.
"""
    pass


def getPartialVecTime(i12, totLen, mutMatrix, errorRate, vect, upNode, flag, BL, mu, tUp, tDown, minTime, maxTime):
    """ Time-aware vector propagation variant (when using sampling dates).

Inputs: ['i12', 'totLen', 'mutMatrix', 'errorRate', 'vect', 'upNode', 'flag', 'BL', 'mu', 'tUp', 'tDown', 'minTime', 'maxTime']
Outputs: see return docs in MAPLE source.
"""
    pass


def mergeVectorsTime(probVect1, bLen1, fromTip1, probVect2, bLen2, fromTip2, returnLK, isUpDown, mu, minTime, maxTime):
    """ Time-aware merge of partial-likelihood vectors.

Inputs: ['probVect1', 'bLen1', 'fromTip1', 'probVect2', 'bLen2', 'fromTip2', 'returnLK', 'isUpDown', 'mu', 'minTime', 'maxTime']
Outputs: see return docs in MAPLE source.
"""
    pass


def findProbRootTime(tree, node):
    """ Root search variant that accounts for sampling-time likelihood.

Inputs: ['tree', 'node']
Outputs: see return docs in MAPLE source.
"""
    pass


def rootVectorTime(tree, node):
    """ Compute root vector with sampling-time modifiers.

Inputs: ['tree', 'node']
Outputs: see return docs in MAPLE source.
"""
    pass


def passGenomeListThroughBranch(mutations, branch):
    """ Transform genome-level diffs passing through a branch (apply flips/accumulate).

Inputs: ['mutations', 'branch']
Outputs: see return docs in MAPLE source.
"""
    pass


def mergeMutationLists(mutListA, mutListB):
    """ Merge mutation lists across an edge; used when creating local references/MAT.

Inputs: ['mutListA', 'mutListB']
Outputs: see return docs in MAPLE source.
"""
    pass


def prepareTreeComparison(treeA, treeB):
    """ Pre-process two trees for RF distance (taxon sets, index maps).

Inputs: ['treeA', 'treeB']
Outputs: see return docs in MAPLE source.
"""
    pass


def RobinsonFouldsWithDay1985(prepA, prepB):
    """ Compute RF distance per Day 1985 algorithm.

Inputs: ['prepA', 'prepB']
Outputs: see return docs in MAPLE source.
"""
    pass

