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
import numpy as np

class Tree(object):
	def __init__(self):
		self.dist = []
		self.replacements=[]
		self.children = []
		self.mutations=[]
		self.up=[]
		self.dirty=[]
		self.name=[]
		self.minorSequences=[]
		self.probVect=[]
		self.probVectUpRight=[]
		self.probVectUpLeft=[]
		self.probVectTotUp=[]
		self.nDesc=[]
		#number of branches descending from node after collapsing 0-length branches
		self.nDesc0=[]
		self.probVectTime=[]
		self.probVectUpRightTime=[]
		self.probVectUpLeftTime=[]
		self.probVectTotUpTime=[]
		self.dateData=[]
	def __repr__(self):
		return "Tree object"
	def addNode(self,dirtiness=True):
		self.up.append(None)
		self.children.append([])
		self.dirty.append(dirtiness)
		self.name.append("")
		self.minorSequences.append([])
		self.mutations.append([])
		self.replacements.append(0)
		self.dist.append(0.0)
		self.probVect.append(None)
		self.probVectUpRight.append(None)
		self.probVectUpLeft.append(None)
		self.probVectTotUp.append(None)
		self.nDesc.append(0)
		# if HnZ:
		# 	self.nDesc0.append(1)

thresholdFoldChangeUpdate = 1.01
thresholdDiffForUpdate = 0.00001
def collectReference(fileName):
    """ Read a FASTA-like reference from file and return reference sequence.

Inputs: ['fileName']
Outputs: see return docs in MAPLE source.
"""
    ref_seq = ""
    with open(fileName) as file:
        for line in file:
            ref_seq += line.replace("\n", "")
    file.close()
    global lref 
    lref = len(ref_seq)
    return ref_seq


def readConciseAlignment(fileName, extractReference, ref, onlyRef):
    """ Read MAPLE 'concise alignment' format; optionally extract embedded reference.

Inputs: ['fileName', 'extractReference', 'ref', 'onlyRef']
Outputs: see return docs in MAPLE source.
"""
    if fileName.endswith(".gz"):
        import gzip
        fileI = gzip.open(fileName, 'rt')
    else:
        fileI = open(fileName)
    line = fileI.readline()
    if extractReference:
        line = fileI.readline()
        ref = ""
        while line != "" and line[0] != ">":
            ref += line.replace("\n", "")
            line = fileI.readline()
        ref = ref.lower()
    if onlyRef:
        return ref
    nSeqs = 0
    data = {}
    while line != "" and line != "\n":
        seqList = []
        name = line.replace(">", "").replace("\n", "")
        line = fileI.readline()
        pos = 0
        while line != "" and line != "\n" and line[0] != ">":
            linelist = line.split()
            if len(linelist) > 2:
                entry = (linelist[0].lower(), int(linelist[1]), int(linelist[2]))
            elif len(linelist) < 2:
                print(
                    "In input file " + fileName + " found line with only one column: \n" + line + "ERROR Please check for errors in the alignment format; if the reference is included at the top of the alignment, then please don't use option --reference.")
                raise Exception("exit")
            else:
                entry = (linelist[0].lower(), int(linelist[1]))
            if ref[entry[1] - 1] == entry[0] and entry[0] != "n" and entry[0] != "-":
                print("Mutation observed into reference nucleotide at position " + str(entry[1]) + " , nucleotide " +
                      entry[0] + ". Wrong reference and/or diff file?")
                raise Exception("exit")
            if entry[1] <= pos:
                print("WARNING, at sample number " + str(nSeqs + 1) + " found entry")
                print(line.replace("\n", ""))
                print("which is inconsistent since the position is already represented by another entry:")
                print(seqList[-1])
                raise Exception("exit")
            else:
                seqList.append(entry)
                if len(entry) == 2:
                    pos = entry[1]
                else:
                    pos = entry[1] + entry[2] - 1
            line = fileI.readline()
        data[name] = seqList
        nSeqs += 1
    fileI.close()
    print(str(nSeqs) + " sequences in diff file.")
    if extractReference:
        return ref, data
    else:
        return data


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
    n = len(oldMutMatrix)
    mutMatrix = np.full((n, n), 0.25)  # Fill with 0.25 (off-diagonal rate)
    np.fill_diagonal(mutMatrix, 0.25)  # Diagonal: 1 - 3*0.25 = 0.25 for JC
    if model != "JC":
        print("Error: Only JC model is implemented.")
        raise Exception("exit")
    # Implement JC model update
    # for i in range(len(oldMutMatrix)):
    #     for j in range(len(oldMutMatrix)):
    #         if i == j:
    #             #TODO: use math.pow
    #             mutMatrix[i][j] = 1 - 3 * 0.25 # fixed value for JC model
    #         else:
    #             mutMatrix[i][j] = 0.25
    print(f"MutMatrix after JC: {mutMatrix}")
    
    # Normalize using np array
    row_sums = mutMatrix.sum(axis=1, keepdims=True)
    mutMatrix = mutMatrix / row_sums
    
    print(f"Normalized mut matrix: {mutMatrix}")
    
    # Update oldMutMatrix by checking if there are significant changes
    # We consider a significant change if the difference between mutMatrix and oldMutMatrix elements is greater than `THRESHOLD`
    # Convert oldMutMatrix to numpy array for comparison
    oldMutMatrix_np = np.array(oldMutMatrix)
    
    # Check if there are significant changes
    if np.any(np.abs(mutMatrix - oldMutMatrix_np) > THRESHOLD):
        # Update oldMutMatrix in-place
        for i in range(n):
            for j in range(n):
                oldMutMatrix[i][j] = mutMatrix[i][j]
        print(f"Updated oldMutMatrix")
        return True
    
    return False

def convertLetterToNumber(letter):
    """ Convert a letter to a number. also checks validty"""
    if letter == "A":
        return 0
    elif letter == "C":
        return 1
    elif letter == "G":
        return 2
    elif letter == "T":
        return 3
    else:
        print("Error: Invalid letter")
        return None

def probVectTerminalNode(diffs, tree, node, ref_seq):
    """ Create a terminal-node probability vector from sample/reference diffs at a node.

Inputs: ['diffs', 'tree', 'node', ref_seq]
refseq: a list of numbers representing a sequence
Outputs: see return docs in MAPLE source.
output: prob vector is a list of tuples (code,start index, stop index)
code 1 : Exact Match
code 2: Mismatch
"""
    # todo: check about vairble ref  -> if numeric vesion of code
    # todo: ask is everything zero indexed? Note, tree uses mutation
    # set up varibles + base case
    probVect = [] # retunrs a list of trriples(code,start index, stop index(
    ref_numbers = [convertLetterToNumber(i) for i in ref_seq] # convert ref to numeric
    index = 0
    if (diffs == None or tree == None):
        print("Invalid call to probVectTerminalNode, empty arguments" )
        return None
    for (letter,position) in diffs:
        if (position > index):
            probVect.append((1,index,position))
            index = position# after we append, we shift our index
        else:
            letter_num = convertLetterToNumber(letter)
            sequence_num= ref_numbers[position]
            if (letter_num== sequence_num):
                probVect.append((1,index,position))
                index = position + 1
            else:
                probVect.append((2,position,position))
    return probVect

# Note: updateProbVectTerminalNode moved to archived functions


def getPartialVec(i12, totLen, mutMatrix, errorRate, vect, upNode, flag):
    """ Propagate a probability vector across a branch using the substitution model.

Inputs: ['i12', 'totLen', 'mutMatrix', 'errorRate', 'vect', 'upNode', 'flag']
Outputs: see return docs in MAPLE source.
"""
    pass


def mergeVectors(probVect1, bLen1, fromTip1, probVect2, bLen2, fromTip2, returnLK, isUpDown):
    """ Combine two partial-likelihood vectors meeting at a node/edge; optionally return LK.

Inputs: ['probVect1', 'bLen1', 'fromTip1', 'probVect2', 'bLen2', 'fromTip2', 'returnLK', 'isUpDown']
probVect1:
bLen1: branch length for the first probabilty vector 
fromTIp1:
probVect2:
bLen2: branch length for the second probabilty vector
fromTip2:
returnLK:
isUpDown: 
TODO: finish another day
Outputs: see return docs in MAPLE source.
"""
    pass


# def findProbRoot(tree, node):
#     """ Search for root that maximizes overall likelihood given current vectors/BLens.

# Inputs: ['tree', 'node']
# Outputs: see return docs in MAPLE source.
# """
#     pass


# TODO: 
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
    # store local variables
    parents = tree.up
    dirty = tree.dirty
    probDown = tree.probVect #up vectors (leaf to root)
    probUpLeft = tree.probVectUpLeft #down vectors (root to leaf)
    probUpRight = tree.probVectUpRight #down vectors (root to leaf)
    children = tree.children
    distances = tree.dist
    parent = parents[cNode]


    if cNode == children[parent][0] : #node is left child
         cIdx = 0
         vectDown = probUpRight[parent]
    else : #node is right child
         cIdx = 1
         vectDown = probUpLeft[parent]

    bestLength = estimateBranchLengthWithDerivative(vectDown,probDown[cNode],fromTipC=len(children[[cNode]] == 0))
    distances[cNode] = bestLength

    dirty[parent] = True
    dirty[cNode] = True
    if addToList: # need to schedule nodes for further 
         nodeList.append((cNode, 2, True, False))
         nodeList.append((parent, cIdx, True, False))


def compare_entry_type(e1,e2) :
    return e1[0] == e2[0]

def compare_entry_lengths(e1, e2):
    return len(e1) == len(e2)

def compare_simple_entry(entry1 ,entry2) :
    for i in range(2, len(entry1)) :
        if (abs(entry1[i] - entry2[i]) > THRESHOLD) :
            return True
    return False

def compare_six_entry(entry1, entry2) :
    if abs(entry1[2] - entry2[2]) > THRESHOLD :
        return True
    for i in range(4) :
        diffVal = abs(entry1[-1][i] - entry2[-1][i])
        if (diffVal > thresholdDiffForUpdate) :
            return True
        if (diffVal>THRESHOLD and ((diffVal/entry1[-1][i]>thresholdFoldChangeUpdate)  or  (diffVal/entry2[-1][i]>thresholdFoldChangeUpdate))):
            return True
    return False
        

def areVectorsDifferent(probVect1, probVect2):
    """ Return True if two probability vectors differ beyond thresholds.

Inputs: ['probVect1', 'probVect2']
Outputs: see return docs in MAPLE source.
"""
    if probVect1 == None or probVect2 == None :
        return False
    pos = 0
    for i in range (len(probVect1)) :
        entry1 = probVect1[i]
        entry2 = probVect2[i]
        if (not compare_entry_type(entry1, entry2)) :
            return True
        if (not compare_entry_lengths(entry1, entry2)) :
            return True
        if entry1[0] < 5: #types 0 - 4
            if (compare_simple_entry(entry1, entry2)):
                return True
            if entry1[0] <= 3: #types 0 to 3
                pos += 1
            else : #type 4
                pos = min(entry1[1], entry2[1])

        elif entry1[0] == 5: #type 5
             pos = min(entry1[1], entry2[1])

        elif entry1[0] == 6: # type 6
            if (compare_six_entry(entry1, entry2)) :
                return True
            pos += 1

        if pos == lref: # lref is length of reference sequence
            break
    return False


      
#createNewick() (and its helpers) starts here
# Helper to format branch length
def _bl_str(bl):
    if bl is None: return ""
    try:
        # format to a reasonable precision
        return ":" + ("{:.6f}".format(float(bl)).rstrip("0").rstrip(".") if bl != 0 else "0")
    except Exception:
        return ":" + str(bl)

# Helper to generate node label (name, support, mutations, lineage)
def _label(node, includeSupports, minSupport, includeMutationList, performLineageAssignmentByRefPlacement):
    parts = []
    name = node.name if hasattr(node, 'name') else None
    
    # name/tip label
    if name is not None:
        parts.append(str(name))

    # lineage annotation
    if performLineageAssignmentByRefPlacement and hasattr(node, 'lineage'):
        # Assuming only basic lineage is needed here, if more complex metadata is
        # required, use the stringForNode logic from your original implementation.
        if node.lineage:
            parts.append(f"[lineage={node.lineage}]")

    # supports for internal nodes
    if includeSupports and hasattr(node, "support"):
        support = node.support
        sup_float = float(support)
        if (minSupport is None) or (sup_float >= float(minSupport)):
            # Newick annotations usually use the & key-value format for internal nodes
            if abs(sup_float - int(sup_float)) < 1e-6:
                parts.append(f"support={int(sup_float)}")
            else:
                parts.append(f"support={sup_float:.3f}")

    # mutation list
    if includeMutationList and hasattr(node, 'mutations'):
        muts = node.mutations
        if muts:
            if isinstance(muts, (list, tuple)):
                mut_str = ",".join(map(str, muts))
            else:
                mut_str = str(muts)
            parts.append(f"mut={mut_str}")
    
    # Combine parts: tip name, then metadata in [&key=value,...]
    # This assumes the node is NOT the root of the entire output tree
    if not parts:
        return ""
    
    if len(parts) == 1 and name is not None and parts[0] == name:
        return name
    
    # If the first part is a name, it goes outside the [&...] block.
    # Otherwise, everything goes inside.
    name_part = parts[0] if name is not None and parts[0] == name else ""
    meta_parts = parts if name is None or parts[0] != name else parts[1:]

    if meta_parts:
        # Newick allows metadata in square brackets [&...] after a node name/group.
        meta_string = f"[&{','.join(meta_parts)}]"
        return f"{name_part}{meta_string}"
    
    return name_part


def createNewick(tree, root_node_id,
                 namesInTree=None,
                 includeMutationList=False,
                 minSupport=None,
                 count0BLenNodesOnce=False,
                 includeSupports=False,
                 performLineageAssignmentByRefPlacement=False):
    """
    Serialize subtree rooted at `root_node_id` to a Newick string using an 
    iterative depth first traversal approach.
    """

    nextNode = root_node_id
    stringList = []
    direction = 0  # 0: Down (left child), 1: Right, 2: Up (parent)

    # NOTE: The tree must be accessible with the indices, e.g., tree.up[nextNode]
    up = tree.up
    children = tree.children
    dist = tree.dist
    

    #  loop continues until the root node is fully processed and the cursor moves past it (nextNode becomes None).
    while nextNode is not None:
        is_leaf = not children[nextNode]

        if not is_leaf and direction == 0:
            # 1. down--entering a new internal node
            # Prepend '(' to start the group
            stringList.append("(")
            
            # Move to the first child
            nextNode = children[nextNode][0]
        
        elif not is_leaf and direction == 1:
            # 2. right--moving from first child to second child in a binary tree
            # Separator between children
            stringList.append(",")
            
            # Move to the second child (assuming binary tree structure!!)
            nextNode = children[nextNode][1]
            direction = 0 # Reset direction to 0 to process the new child
        
        else: # is_leaf or direction == 2 (Moving UP)
            # 3. up--leaving a node: a leaf, or an internal node after processing children; go back up to parent

            # get node label and branch length
            node_bl = dist[nextNode]
            
            # TODO: could check for the special case of 0-branch-length unary nodes (not implemented here)

            # If a leaf, append its label
            if is_leaf:
                label_str = _label(tree, nextNode, includeSupports, minSupport, includeMutationList, performLineageAssignmentByRefPlacement)
                stringList.append(label_str)

            # If internal, append ')' followed by the label (if any)
            if not is_leaf:
                stringList.append(")")
                label_str = _label(tree, nextNode, includeSupports, minSupport, includeMutationList, performLineageAssignmentByRefPlacement)
                stringList.append(label_str)

            # append branch length (always after label/group)
            stringList.append(_bl_str(node_bl))


            # move back up to parent and set new direction
            parent = up[nextNode]
            if parent is not None:
                # Check if this node was the first or second child of the parent
                if len(children[parent]) > 0 and children[parent][0] == nextNode:
                    # was the left child, next step is the right child
                    direction = 1
                else:
                    # was the right child, next step is the parent
                    direction = 2
            
            nextNode = parent # Move up one level
            
            # the root node has parent=None, so the loop correctly terminates when nextNode becomes None.


    newick = "".join(stringList) + ";"
    return newick

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

def updateProbVectTerminalNode(probVect, numMinSeqs):
    """ Post-process/normalize terminal-node vector given min sequence counts.

Inputs: ['probVect', 'numMinSeqs']
Outputs: see return docs in MAPLE source.
"""
    pass
    
def RobinsonFouldsWithDay1985(prepA, prepB):
    """ Compute RF distance per Day 1985 algorithm.

Inputs: ['prepA', 'prepB']
Outputs: see return docs in MAPLE source.
"""
    pass

if __name__ == "__main__":
    diffs = [('t', 313), ('g', 1832), ('c', 10029), ('c', 21618), ('t', 22917), ('c', 22995), ('a', 23063),
             ('c', 23604), ('a', 28271), ('g', 29742)] # from the sample from maple test
    node = 0
    tree = None
   