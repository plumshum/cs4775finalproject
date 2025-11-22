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


      

def createNewick(tree, node,
                 namesInTree=None,
                 includeMutationList=False,
                 minSupport=None,
                 count0BLenNodesOnce=False,
                 includeSupports=False,
                 performLineageAssignmentByRefPlacement=False):
    """
    Serialize subtree rooted at `node` to a Newick string.

    node may be an object with attributes or a dict.
    Returns: Newick string (ending with ';')
    """
    # idk what format our tree is rn so read node properties whether node is dict-like or attribute-like
    def _get(n, attr, default=None):
        if n is None:
            return default
        if hasattr(n, attr):
            return getattr(n, attr)
        if isinstance(n, dict) and attr in n:
            return n[attr]
        return default

    # convert branch length to string with desired formatting
    def _bl_str(nextNode):
        bl = tree.dist[nextNode]
        if bl is None:
            return ""
        try:
            # format to a reasonable precision
            return ":" + ("{:.6f}".format(float(bl)).rstrip("0").rstrip(".") if bl != 0 else "0")
        except Exception:
            return ":" + str(bl)

    # prepare label for a node depending on flags
    def _label(n):
        # tip name override via namesInTree mapping
        name = _get(n, "name", _get(n, "label", None))
        node_id = _get(n, "id", None)
        if namesInTree and node_id is not None and node_id in namesInTree:
            name = namesInTree[node_id]

        parts = []

        # If this is a tip and we have a name, use it
        if name is not None:
            parts.append(str(name))

        # lineage annotation
        if performLineageAssignmentByRefPlacement:
            lineage = _get(n, "lineage", None)
            if lineage:
                parts.append(f"[lineage={lineage}]")

        # supports for internal nodes
        if includeSupports:
            support = _get(n, "support", None)
            if support is not None:
                try:
                    sup_float = float(support)
                except Exception:
                    sup_float = None
                if sup_float is not None:
                    if (minSupport is None) or (sup_float >= float(minSupport)):
                        # show support either as integer if appropriate or float
                        if abs(sup_float - int(sup_float)) < 1e-6:
                            parts.append(f"[support={int(sup_float)}]")
                        else:
                            parts.append(f"[support={sup_float:.3f}]")

        # mutation list
        if includeMutationList:
            muts = _get(n, "mutations", _get(n, "mutation_list", None))
            if muts:
                # accept list or string
                if isinstance(muts, (list, tuple)):
                    mut_str = ",".join(map(str, muts))
                else:
                    mut_str = str(muts)
                parts.append(f"[mut={mut_str}]")

        # combine parts with spaces (or customize separator)
        if parts:
            return "".join(parts)
        else:
            return ""

    # handle zero-length branch printing only once if requested
    zero_printed = set()

    def _node_to_newick(n):
        children = _get(n, "children", _get(n, "child", []))
        if children is None:
            children = []
        # leaf node
        if not children:
            label = _label(n)
            if label == "":
                label = _get(n, "name", _get(n, "label", "")) or ""
            # branch length after name
            return f"{label}{_bl_str(n)}"
        
        # check branch length value
        bl_val = float(_get(n, "branch_length", 0))

        if count0BLenNodesOnce and bl_val == 0 and len(children) == 1:
            # This is a unary node with a 0 branch length.
            # Collapse it by just recursing on its child.
            # We don't add its label or branch length.
            return _node_to_newick(children[0])
        
        # internal node
        child_strs = []
        for c in children:
            child_newick = _node_to_newick(c)
            child_strs.append(child_newick)
        # join children with commas
        subtree = "(" + ",".join(child_strs) + ")"
        # internal node label (support, mutation, lineage) placed after )
        lab = _label(n)
        bls = _get(n, "branch_length", _get(n, "bl", None))
        # count0BLenNodesOnce: if branch length is zero, optionally only print once
        bl_suffix = ""
        if bls is not None:
            bl_val = float(bls)
            if count0BLenNodesOnce and bl_val == 0:
                # use node id to suppress duplicates
                nid = _get(n, "id", None)
                if nid is None or nid not in zero_printed:
                    zero_printed.add(nid)
                    bl_suffix = _bl_str(n)
                else:
                    bl_suffix = ""  # skip printing 0 again
            else:
                bl_suffix = _bl_str(n)
        # if lab empty and we have no special flags, avoid empty []
        return f"{subtree}{lab}{bl_suffix}"

    # produce the string and append semicolon
    newick_body = _node_to_newick(node)
    newick = newick_body + ";"



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
   