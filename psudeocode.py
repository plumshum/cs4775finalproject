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
import sys

# for memory
from memory_profiler import memory_usage
import matplotlib.pyplot as plt

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

# Allele Data
allelesDict = {"A":0,"C":1,"G":2,"T":3}

# Global mut matrix. in MAPLE it originally has different mut matrix global variables based on the different modles used, but in this case we are only using JC model
mutMatrixGlobal=[[-1.0,1.0/3,1.0/3,1.0/3],[1.0/3,-1.0,1.0/3,1.0/3],[1.0/3,1.0/3,-1.0,1.0/3],[1.0/3,1.0/3,1.0/3,-1.0]]

# Error Rate Constants that are fixed 
errorRates=None
errorRateGlobal=None
cumulativeRate=[0.0]
refIndeces = []

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
    Outputs: if extractReference is True, returns (ref, data); else returns str data.
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
rootFreqs=[0.25,0.25,0.25,0.25]
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
    # IMPORTANT: MAPLE uses a continuous-time substitution *rate* matrix Q
    # (rows sum to 0, diagonal is negative). Our likelihood/derivative code
    # assumes this convention.
    n = len(oldMutMatrix)
    mutMatrix = np.full((n, n), 1.0 / 3.0)
    np.fill_diagonal(mutMatrix, -1.0)
    if model != "JC":
        print("Error: Only JC model is implemented.")
        raise Exception("exit")
    print(f"JC rate matrix Q: {mutMatrix}")
    
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
    letter = letter.upper()
    if letter == "A":
        return 0
    elif letter == "C":
        return 1
    elif letter == "G":
        return 2
    elif letter == "T":
        return 3
    else:
        return None


def _iter_diff_records(diffs):
    """Yield normalized diff records as (letter_lower, pos_1idx, length>=1)."""
    if diffs is None:
        return
    for rec in diffs:
        if len(rec) == 2:
            letter, pos = rec
            length = 1
        elif len(rec) == 3:
            letter, pos, length = rec
        else:
            raise ValueError(f"Unexpected diff record (len={len(rec)}): {rec}")
        if length is None:
            length = 1
        length = int(length)
        if length <= 0:
            raise ValueError(f"Invalid diff length {length} for record {rec}")
        yield (str(letter).lower(), int(pos), length)

def probVectTerminalNode(diffs, tree, node, ref_seq):
    """
    MAPLE-faithful construction of a terminal genome list.
    """
    global lref

    if diffs is None or ref_seq is None:
        raise ValueError("probVectTerminalNode requires diffs and ref_seq")

    lref = len(ref_seq)

    probVect = []
    pos = 1  # 1-indexed genome position

    for (letter, start_pos, length) in _iter_diff_records(diffs):
        if start_pos < pos:
            raise ValueError(
                f"Diff records must be sorted and non-overlapping "
                f"(start {start_pos} < current pos {pos})"
            )

        # --- reference stretch before mutation ---
        if start_pos > pos:
            probVect.append((4, start_pos - 1))
            pos = start_pos

        end_pos = start_pos + length - 1

        # --- N / missing data ---
        if letter in ("n", "-"):
            probVect.append((5, end_pos))
            pos = end_pos + 1
            continue

        state = convertLetterToNumber(letter)
        if state is None:
            probVect.append((5, end_pos))
            pos = end_pos + 1
            continue

        # --- emit single-site mutations ONLY ---
        for p in range(start_pos, end_pos + 1):
            ref_nuc = convertLetterToNumber(ref_seq[p - 1])
            if ref_nuc is None:
                ref_nuc = 0
            probVect.append((state, ref_nuc))
            pos += 1

    # --- close remaining reference ---
    if pos <= lref:
        _append_stretch(probVect, 4, lref)

    return probVect

def _append_stretch(probVect, typ, end_pos):
    if probVect and probVect[-1][0] == typ:
        # Extend previous stretch
        probVect[-1] = (typ, end_pos)
    else:
        probVect.append((typ, end_pos))


# Note: updateProbVectTerminalNode moved to archived functions


def propagatePartialVecHelper(mutMatrix, vect, totLen):
    vect = np.array(vect, dtype = float)
    changeVec = np.dot(mutMatrix, vect)
    changeVecScaled = totLen * changeVec
    ansVec = vect + changeVecScaled
    if (ansVec < 0).any():
        return [0.25,0.25,0.25,0.25]
    else:
        return ansVec.tolist()

usingErrorRate = False
def getPartialVec(i12, totLen, mutMatrix, errorRate, vect=None, upNode=False, flag=False):
    if i12==6:
        if (not totLen) and (vect):
            return list(vect)
        mutMatrixToUse = mutMatrix.T if upNode else mutMatrix
        newVect = propagatePartialVecHelper(mutMatrixToUse, vect, totLen)
        return newVect
    elif usingErrorRate and flag:
        newVect = [errorRate*0.33333] * 4
        newVect[i12] = 1.0 - errorRate
        if (not totLen):
            return newVect
        mutatedPartialVec = propagatePartialVecHelper(mutMatrix.T, newVect, totLen)
        return mutatedPartialVec
    else:
        if (not totLen):
            defVect=[0.0,0.0,0.0,0.0]
            defVect[i12]+=1.0
            return defVect
        newVect = mutMatrix[i12, :] * totLen if upNode else mutMatrix[:, i12] * totLen
        newVect[i12]+=1.0
        if newVect[i12] < 0:
            return [0.25,0.25,0.25,0.25]
        return newVect.tolist()

# CONSTANTS FOR FLAGS
useRateVariation = False
errorRateSiteSpecific = False 
minimumCarryOver=sys.float_info.min*(1e50)
totError = None
globalTotRate=0.0# remark: orignial -float(lref)
# TODO: but prob should do lref*0.75

# beginning of merge vectors code
def determineState(probVec, refNuc):
    """
    Helper func
    Determine the state code from a probability vector.
    Returns:
        - 0-3: specific nucleotide if one has probability ~1
        - 4: Reference if all low/equal
        - 6: "O" (other/ambiguous) if mixed probabilities
    """
    maxProb = max(probVec)
    maxIdx = probVec.index(maxProb)
    
    if maxProb > 0.99:  # Nearly certain
        return maxIdx  # Return 0-3 (A, C, G, T)
    elif maxProb < 0.3:  # All roughly equal
        return 4  # Reference
    else:
        return 6  # Mixed/"O"

def handler_helper(ctx):
    if ctx["returnLK"]:
        ctx["cumulPartLk"] += (ctx["bLen1"] + ctx["bLen2"]) * (ctx["cumulativeRateUsed"][ctx["pos"]] - ctx["cumulativeRateUsed"][ctx["newPos"]])
        if ctx["usingErrorRate"]:
            if ctx["fromTip1"] or ctx["fromTip2"]:
                if ctx["errorRateSiteSpecific"]:
                    ctx["cumErrorRate"] = ctx["cumulativeErrorRateUsed"][ctx["newPos"]] - ctx["cumulativeErrorRateUsed"][ctx["pos"]]
                else:
                    ctx["cumErrorRate"] = ctx["errorRate"] * (ctx["newPos"] - ctx["pos"])
            if ctx["fromTip1"]:
                ctx["cumulPartLk"] += ctx["cumErrorRate"]
            if ctx["fromTip2"]:
                ctx["cumulPartLk"] += ctx["cumErrorRate"]
    ctx["pos"] = ctx["newPos"]

def handler_5_5(ctx):
    ctx["newPos"] = min(ctx["entry1"][1], ctx["entry2"][1])
    ctx["probVect"].append((5, ctx["newPos"]))
    handler_helper(ctx)

def handler_5_L5(ctx):
    if ctx["entry2"][0] < 4:
        ctx["newPos"] = ctx["pos"] + 1
        newEl = ctx["entry2"][1]
    else:
        ctx["newPos"] = min(ctx["entry1"][1], ctx["entry2"][1])
        newEl = ctx["newPos"]
    if ctx["isUpDown"]:
        if ctx["usingErrorRate"]:
            if len(ctx["entry2"]) == 2:
                if ctx["bLen2"] or ctx["fromTip2"]:
                    ctx["probVect"].append((ctx["entry2"][0], newEl, ctx["bLen2"], 0.0, ctx["fromTip2"]))
                else:
                    ctx["probVect"].append((ctx["entry2"][0], newEl))
            elif len(ctx["entry2"]) == 3:
                ctx["probVect"].append((ctx["entry2"][0], newEl, ctx["bLen2"], 0.0, ctx["entry2"][3]))
            else:
                ctx["probVect"].append((ctx["entry2"][0], newEl, ctx["entry2"][2] + ctx["bLen2"], 0.0, ctx["entry2"][3]))
        else:
            if len(ctx["entry2"]) > 2:
                ctx["probVect"].append((ctx["entry2"][0], newEl, ctx["entry2"][2] + ctx["bLen2"], 0.0))
            else:
                if ctx["bLen2"]:
                    ctx["probVect"].append((ctx["entry2"][0], newEl, ctx["bLen2"], 0.0))
                else:
                    ctx["probVect"].append((ctx["entry2"][0], newEl))
    handler_helper(ctx)

def handler_5_G5(ctx):
    ctx["newPos"] = ctx["pos"] + 1
    if ctx["isUpDown"]:
        if ctx["useRateVariation"]:
            ctx["mutMatrix"] = ctx["mutMatricesUsed"][ctx["pos"]]
        totBLen = ctx["bLen2"]
        if len(ctx["entry2"]) > 3:
            totBLen += ctx["entry2"][2]
        if totBLen:
            ctx["newVec"] = getPartialVec(6, totBLen, ctx["mutMatrix"], 0, vect=ctx["entry2"][-1])
        else:
            ctx["newVec"] = list(ctx["entry2"][-1])
        for i in range(4):
            ctx["newVec"][i] *= ctx["rootFreqs"][i]
        ctx["totSum"] = sum(ctx["newVec"])
        for i in range(4):
            ctx["newVec"][i] /= ctx["totSum"]
        ctx["probVect"].append((6, ctx["entry2"][1], ctx["newVec"]))
    else:
        if len(ctx["entry2"]) > 3:
            ctx["probVect"].append((6, ctx["entry2"][1], ctx["entry2"][2] + ctx["bLen2"], ctx["entry2"][3]))
        else:
            if ctx["bLen2"]:
                ctx["probVect"].append((6, ctx["entry2"][1], ctx["bLen2"], ctx["entry2"][2]))
            else:
                ctx["probVect"].append((6, ctx["entry2"][1], ctx["entry2"][2]))
    handler_helper(ctx)

def handler_L5_5(ctx):
    entry1 = ctx["entry1"]
    entry2 = ctx["entry2"]

    if entry1[0] < 4:
        ctx["newPos"] = ctx["pos"] + 1
        newEl = entry1[1]
    else:
        ctx["newPos"] = min(entry1[1], entry2[1])
        newEl = ctx["newPos"]

    if ctx["isUpDown"]:
        if ctx["usingErrorRate"]:
            if len(entry1) == 2:
                if ctx["bLen1"]:
                    ctx["probVect"].append((entry1[0], newEl, ctx["bLen1"], False))
                else:
                    ctx["probVect"].append((entry1[0], newEl))
            elif len(entry1) == 3:
                ctx["probVect"].append((entry1[0], newEl, ctx["bLen1"], entry1[2]))
            elif len(entry1) == 4:
                ctx["probVect"].append((entry1[0], newEl, entry1[2] + ctx["bLen1"], entry1[3]))
            else:
                ctx["probVect"].append((entry1[0], newEl, entry1[2], entry1[3] + ctx["bLen1"], entry1[4]))
        else:
            if len(entry1) == 2:
                if ctx["bLen1"]:
                    ctx["probVect"].append((entry1[0], newEl, ctx["bLen1"]))
                else:
                    ctx["probVect"].append((entry1[0], newEl))
            elif len(entry1) == 3:
                ctx["probVect"].append((entry1[0], newEl, entry1[2] + ctx["bLen1"]))
            else:
                ctx["probVect"].append((entry1[0], newEl, entry1[2], entry1[3] + ctx["bLen1"]))

    else:
        if ctx["usingErrorRate"]:
            if len(entry1) == 2:
                if ctx["bLen1"] or ctx["fromTip1"]:
                    ctx["probVect"].append((entry1[0], newEl, ctx["bLen1"], ctx["fromTip1"]))
                else:
                    ctx["probVect"].append((entry1[0], newEl))
            elif len(entry1) == 3:
                if ctx["bLen1"]:
                    ctx["probVect"].append((entry1[0], newEl, ctx["bLen1"], entry1[3]))
                else:
                    ctx["probVect"].append((entry1[0], newEl, entry1[3]))
            else:
                ctx["probVect"].append((entry1[0], newEl, entry1[2] + ctx["bLen1"], entry1[3]))
        else:
            if len(entry1) > 2:
                ctx["probVect"].append((entry1[0], newEl, entry1[2] + ctx["bLen1"]))
            else:
                if ctx["bLen1"]:
                    ctx["probVect"].append((entry1[0], newEl, ctx["bLen1"]))
                else:
                    ctx["probVect"].append((entry1[0], newEl))

    handler_helper(ctx)

def handler_G5_5(ctx):
    entry1 = ctx["entry1"]
    ctx["newPos"] = ctx["pos"] + 1
    ctx["refNucToPass"] = entry1[1]

    if ctx["isUpDown"] and ((len(entry1) == 4 and entry1[2] > 0) or ctx["bLen1"]):
        if ctx["useRateVariation"]:
            ctx["mutMatrix"] = ctx["mutMatricesUsed"][ctx["pos"]]

        totBLen = ctx["bLen1"]
        if len(entry1) > 3:
            totBLen += entry1[2]

        if totBLen:
            ctx["newVec"] = getPartialVec(6, totBLen, ctx["mutMatrix"], 0, vect=entry1[-1], upNode=True)
        else:
            ctx["newVec"] = list(entry1[-1])

        totSum = sum(ctx["newVec"])
        for i in range(4):
            ctx["newVec"][i] /= totSum

        ctx["probVect"].append((6, entry1[1], ctx["newVec"]))

    else:
        if len(entry1) > 3:
            ctx["probVect"].append((6, entry1[1], entry1[2] + ctx["bLen1"], entry1[3]))
        else:
            if ctx["bLen1"]:
                ctx["probVect"].append((6, entry1[1], ctx["bLen1"], entry1[2]))
            else:
                ctx["probVect"].append((6, entry1[1], entry1[2]))

    handler_helper(ctx)

def handler_other(ctx):
    ctx["totLen1"] = ctx["bLen1"]
    if ctx["entry1"][0] == 6:
        if len(ctx["entry1"]) > 3:
            ctx["totLen1"] += ctx["entry1"][2]
    elif len(ctx["entry1"]) > (2 + (1 if ctx["usingErrorRate"] else 0)):
        ctx["totLen1"] += ctx["entry1"][2]
        if len(ctx["entry1"]) > (3 + (1 if ctx["usingErrorRate"] else 0)):
            ctx["totLen1"] += ctx["entry1"][3]

    ctx["totLen2"] = ctx["bLen2"]
    if len(ctx["entry2"]) > (2 + (ctx["usingErrorRate"] or ctx["entry2"][0] == 6)):
        ctx["totLen2"] += ctx["entry2"][2]

    ctx["flag1"] = (ctx["usingErrorRate"] and (ctx["entry1"][0] != 6) and ((len(ctx["entry1"]) > 2 and ctx["entry1"][-1]) or ctx["fromTip1"]))
    ctx["flag2"] = (ctx["usingErrorRate"] and (ctx["entry2"][0] != 6) and ((len(ctx["entry2"]) > 2 and ctx["entry2"][-1]) or ctx["fromTip2"]))

    if ctx["entry1"][0] == 4 and ctx["entry2"][0] == 4:
        ctx["newPos"] = min(ctx["entry1"][1], ctx["entry2"][1])
    else:
        ctx["newPos"] = ctx["pos"] + 1

    if ctx["returnLK"]:
        if ctx["entry1"][0] == 4 and ctx["entry2"][0] == 4:
            if ctx["totLen2"] > ctx["bLen2"] or ctx["totLen1"] > ctx["bLen1"]:
                ctx["cumulPartLk"] += (ctx["totLen2"] - ctx["bLen2"] + ctx["totLen1"] - ctx["bLen1"]) * (ctx["cumulativeRateUsed"][ctx["newPos"]] - ctx["cumulativeRateUsed"][ctx["pos"]])
                if ctx["usingErrorRate"]:
                    if ((not ctx["fromTip1"]) and ctx["flag1"]) or ((not ctx["fromTip2"]) and ctx["flag2"]):
                        if ctx["errorRateSiteSpecific"]:
                            ctx["cumErrorRate"] = ctx["cumulativeErrorRateUsed"][ctx["pos"]] - ctx["cumulativeErrorRateUsed"][ctx["newPos"]]
                        else:
                            ctx["cumErrorRate"] = ctx["errorRate"] * (ctx["pos"] - ctx["newPos"])
                        if ((not ctx["fromTip1"]) and ctx["flag1"]):
                            ctx["cumulPartLk"] += ctx["cumErrorRate"]
                        if ((not ctx["fromTip2"]) and ctx["flag2"]):
                            ctx["cumulPartLk"] += ctx["cumErrorRate"]
        else:
            if ctx["entry1"][0] != 4:
                ctx["refNucToPass"] = ctx["entry1"][1]
            else:
                ctx["refNucToPass"] = ctx["entry2"][1]
            if ctx["useRateVariation"]:
                ctx["cumulPartLk"] -= ctx["mutMatricesUsed"][ctx["pos"]][ctx["refNucToPass"]][ctx["refNucToPass"]] * (ctx["bLen2"] + ctx["bLen1"])
            else:
                ctx["cumulPartLk"] -= ctx["mutMatrix"][ctx["refNucToPass"]][ctx["refNucToPass"]] * (ctx["bLen2"] + ctx["bLen1"])
            if ctx["usingErrorRate"] and ((ctx["entry1"][0] != ctx["entry2"][0]) or ctx["entry1"][0] == 6) and (ctx["fromTip1"] or ctx["fromTip2"]):
                if ctx["errorRateSiteSpecific"]:
                    ctx["cumErrorRate"] = ctx["errorRatesUsed"][ctx["pos"]]
                else:
                    ctx["cumErrorRate"] = ctx["errorRate"]
                if ctx["fromTip1"]:
                    ctx["cumulPartLk"] += ctx["cumErrorRate"]
                if ctx["fromTip2"]:
                    ctx["cumulPartLk"] += ctx["cumErrorRate"]

    if ctx["entry2"][0] == ctx["entry1"][0] and ctx["entry2"][0] < 5:
        if ctx["entry1"][0] == 4:
            ctx["probVect"].append((4, ctx["newPos"]))
        else:
            ctx["probVect"].append((ctx["entry1"][0], ctx["entry1"][1]))
            if ctx["returnLK"]:
                if ctx["useRateVariation"]:
                    ctx["cumulPartLk"] += ctx["mutMatricesUsed"][ctx["pos"]][ctx["entry1"][0]][ctx["entry1"][0]] * (ctx["totLen1"] + ctx["totLen2"])
                else:
                    ctx["cumulPartLk"] += ctx["mutMatrix"][ctx["entry1"][0]][ctx["entry1"][0]] * (ctx["totLen1"] + ctx["totLen2"])
                if ctx["usingErrorRate"]:
                    if ((not ctx["fromTip1"]) and ctx["flag1"]) or ((not ctx["fromTip2"]) and ctx["flag2"]):
                        if ctx["errorRateSiteSpecific"]:
                            ctx["cumErrorRate"] = ctx["errorRatesUsed"][ctx["pos"]]
                        else:
                            ctx["cumErrorRate"] = ctx["errorRate"]
                        if ((not ctx["fromTip1"]) and ctx["flag1"]):
                            ctx["cumulPartLk"] -= ctx["cumErrorRate"]
                        if ((not ctx["fromTip2"]) and ctx["flag2"]):
                            ctx["cumulPartLk"] -= ctx["cumErrorRate"]

    elif (not ctx["totLen1"]) and (not ctx["totLen2"]) and ctx["entry1"][0] < 5 and ctx["entry2"][0] < 5 and (not ctx["flag1"]) and (not ctx["flag2"]):
        if ctx["returnLK"]:
            print("mergeVectors() returning None 1")
            raise Exception("exit")
        else:
            return None
    else:
        if ctx["usingErrorRate"] and ctx["errorRateSiteSpecific"]:
            ctx["errorRate"] = ctx["errorRatesUsed"][ctx["pos"]]
        if ctx["useRateVariation"]:
            ctx["mutMatrix"] = ctx["mutMatricesUsed"][ctx["pos"]]

        if ctx["entry1"][0] == 4:
            ctx["refNucToPass"] = ctx["entry2"][1]
            ctx["i1"] = ctx["refNucToPass"]
        else:
            ctx["refNucToPass"] = ctx["entry1"][1]
            ctx["i1"] = ctx["entry1"][0]
        if ctx["i1"] <= 4:
            if ctx["totLen1"] or ctx["flag1"]:
                if ctx["isUpDown"] and len(ctx["entry1"]) > 3 + ctx["usingErrorRate"]:
                    ctx["newVec"] = getPartialVec(ctx["i1"], ctx["entry1"][2], ctx["mutMatrix"], ctx["errorRate"], flag=ctx["flag1"])
                    for ctx["i"] in range(4):
                        ctx["newVec"][ctx["i"]] *= ctx["rootFreqs"][ctx["i"]]
                    if ctx["entry1"][3] + ctx["bLen1"]:
                        ctx["newVec"] = getPartialVec(6, ctx["entry1"][3] + ctx["bLen1"], ctx["mutMatrix"], 0, vect=ctx["newVec"], upNode=True)
                else:
                    ctx["newVec"] = getPartialVec(ctx["i1"], ctx["totLen1"], ctx["mutMatrix"], ctx["errorRate"], flag=ctx["flag1"], upNode=ctx["isUpDown"])
            else:
                ctx["newVec"] = [0.0, 0.0, 0.0, 0.0]
                ctx["newVec"][ctx["i1"]] = 1.0
        else:
            if ctx["totLen1"]:
                ctx["newVec"] = getPartialVec(6, ctx["totLen1"], ctx["mutMatrix"], 0, vect=ctx["entry1"][-1], upNode=ctx["isUpDown"])
            else:
                ctx["newVec"] = list(ctx["entry1"][-1])

        if ctx["entry2"][0] == 4:
            ctx["i2"] = ctx["refNucToPass"]
        else:
            ctx["i2"] = ctx["entry2"][0]
        if ctx["i2"] == 6:
            if ctx["totLen2"]:
                ctx["newVec2"] = getPartialVec(6, ctx["totLen2"], ctx["mutMatrix"], 0, vect=ctx["entry2"][-1])
            else:
                ctx["newVec2"] = ctx["entry2"][-1]
        else:
            if ctx["totLen2"] or ctx["flag2"]:
                ctx["newVec2"] = getPartialVec(ctx["i2"], ctx["totLen2"], ctx["mutMatrix"], ctx["errorRate"], flag=ctx["flag2"])
            else:
                ctx["newVec2"] = [0.0, 0.0, 0.0, 0.0]
                ctx["newVec2"][ctx["i2"]] = 1.0

        for ctx["j"] in range(4):
            ctx["newVec"][ctx["j"]] *= ctx["newVec2"][ctx["j"]]
        ctx["totSum"] = sum(ctx["newVec"])
        if not ctx["totSum"]:
            if ctx["returnLK"]:
                print("mergeVectors() returning None 2")
                raise Exception("exit")
            else:
                return None
        for ctx["i"] in range(4):
            ctx["newVec"][ctx["i"]] /= ctx["totSum"]

        ctx["state"] = determineState(ctx["newVec"], ctx["refNucToPass"])
        if ctx["state"] == 6:
            ctx["probVect"].append((6, ctx["refNucToPass"], ctx["newVec"]))
        else:
            if ctx["state"] == 4:
                ctx["probVect"].append((4, ctx["newPos"]))
            else:
                ctx["probVect"].append((ctx["state"], ctx["refNucToPass"]))

        if ctx["returnLK"]:
            ctx["totalFactor"] *= ctx["totSum"]

    ctx["pos"] = ctx["newPos"]
    
def advance_entry_if_needed(ctx, probVect, which):
    indexKey = f"indexEntry{which}"
    entryKey = f"entry{which}"

    entry = ctx[entryKey]

    # Single-site entries
    if entry[0] < 4 or entry[0] == 6:
        ctx[indexKey] += 1

    # Stretch entries
    elif ctx["pos"] == entry[1]:
        ctx[indexKey] += 1

    if ctx[indexKey] >= len(probVect):
        return False

    ctx[entryKey] = probVect[ctx[indexKey]]
    return True




def mergeVectors(probVect1, bLen1, fromTip1, probVect2, bLen2, fromTip2,
                 returnLK=False, isUpDown=False, numMinor1=0, numMinor2=0,
                 errorRateGlobalPassed=None, mutMatrixGlobalPassed=None,
                 errorRatesGlobal=None, mutMatricesGlobal=None,
                 cumulativeRateGlobal=None, cumulativeErrorRateGlobal=None):
    global lref
    # propagating handler map
    handler_map = {}
    handler_map[(5, 5)] = handler_5_5
    for i in range(5):
        handler_map[(5, i)] = handler_5_L5
    handler_map[(5, 6)] = handler_5_G5
    for i in range(5):
        handler_map[(i, 5)] = handler_L5_5
    handler_map[(6, 5)] = handler_G5_5
    
    ctx = {
		"pos": 0,
		"newPos": 0,

		"probVect": [],
		"newVec": [],
		"newVec2": [],

		"entry1": probVect1[0],
		"entry2":  probVect2[0],
		"indexEntry1": 0,
		"indexEntry2": 0,

		"totLen1": 0.0,
		"totLen2": 0.0,
		"refNucToPass": -1,

		"cumulPartLk": 0.0,
		"totalFactor": 1.0,

		"mutMatrix": None,
		"mutMatricesUsed": None,
		"errorRate": None,
		"errorRatesUsed": None,
		"cumulativeRateUsed": None,
		"cumulativeErrorRateUsed": None,

		"useRateVariation": useRateVariation,
		"usingErrorRate": usingErrorRate,
		"errorRateSiteSpecific": errorRateSiteSpecific,
		"globalTotRate": globalTotRate,
		"totError": totError, #set constnat right above mergeVector handlers
		"minimumCarryOver": minimumCarryOver, # constant set right above mergeVector handlers
		"lRef": lref,
		"rootFreqs": rootFreqs, #above mutmatrix function
        "fromTip1": fromTip1,
		"fromTip2": fromTip2,

		"bLen1": bLen1,
		"bLen2": bLen2,

		"numMinor1": numMinor1,
		"numMinor2": numMinor2,

		"errorRateGlobalPassed": errorRateGlobalPassed,
		"mutMatrixGlobalPassed": mutMatrixGlobalPassed,
		"errorRatesGlobal": errorRatesGlobal,
		"mutMatricesGlobal": mutMatricesGlobal,
		"cumulativeRateGlobal": cumulativeRateGlobal,
		"cumulativeErrorRateGlobal": cumulativeErrorRateGlobal,

		"returnLK": returnLK,
		"isUpDown": isUpDown,

		"totSum": 0.0,
		"cumErrorRate": 0.0,
		"i": 0,
		"j": 0,
		"i1": 0,
		"i2": 0,
        
		"flag1": False,
        "flag2": False,
	}
    
    # Initialize mutation matrices
    if useRateVariation:
        ctx["mutMatricesUsed"] = ctx["mutMatricesGlobal"] if ctx["mutMatricesGlobal"] is not None else mutMatricesGlobal
    else:
        # Fall back to the global matrix if none is passed.
        ctx["mutMatrix"] = (
            ctx["mutMatrixGlobalPassed"]
            if ctx["mutMatrixGlobalPassed"] is not None
            else mutMatrixGlobal
        )

    if usingErrorRate and errorRateSiteSpecific:
        ctx["errorRatesUsed"] = ctx["errorRatesGlobal"] if ctx["errorRatesGlobal"] is not None else errorRatesGlobal
    else:
        ctx["errorRate"] = (
            ctx["errorRateGlobalPassed"]
            if ctx["errorRateGlobalPassed"] is not None
            else errorRateGlobal
        )

    if returnLK:
        # Ensure we always have a cumulative-rate array for LK accounting.
        # If the caller doesn't provide one, fall back to module-level `cumulativeRate`.
        rate_used = (
            ctx["cumulativeRateGlobal"]
            if ctx["cumulativeRateGlobal"] is not None
            else cumulativeRateGlobal
        )
        if rate_used is None:
            rate_used = cumulativeRate
        # Final fallback: constant per-site rate (cumulative[k] = k).
        if rate_used is None or len(rate_used) < (lref + 1):
            rate_used = np.arange(lref + 1, dtype=float).tolist()
        ctx["cumulativeRateUsed"] = rate_used
        if usingErrorRate and errorRateSiteSpecific:
            ctx["cumulativeErrorRateUsed"] = ctx["cumulativeErrorRateGlobal"] if ctx["cumulativeErrorRateGlobal"] is not None else cumulativeErrorRateGlobal
        ctx["cumulPartLk"] = (ctx["bLen1"] + ctx["bLen2"]) * globalTotRate
        if usingErrorRate:
            if fromTip1 or ctx["numMinor1"]:
                ctx["cumulPartLk"] += totError * (1 + ctx["numMinor1"])
            if ctx["fromTip2"] or ctx["numMinor2"]:
                ctx["cumulPartLk"] += totError * (1 + ctx["numMinor2"])
	
    # main loop
    while True:
        entries_tuple = ctx["entry1"][0], ctx["entry2"][0]
        handler_fxn = handler_map.get(entries_tuple, handler_other)

        # update call
        handler_fxn(ctx)
        
        # AFTER handler and termination check
        if not advance_entry_if_needed(ctx, probVect1, 1):
            break

        if not advance_entry_if_needed(ctx, probVect2, 2):
            break

        if ctx["pos"] == ctx["lRef"]:
            break

        if ctx["returnLK"] and ctx["totalFactor"] <= ctx["minimumCarryOver"]:
            try:
                if ctx["totalFactor"] < sys.float_info.min:
                    print("In mergeVectors() too small LK")
                    raise Exception("exit")
            except:
                print("In mergeVectors() value error")
                raise Exception("exit")
            ctx["cumulPartLk"] += log(ctx["totalFactor"])
            ctx["totalFactor"] = 1.0

        if ctx["pos"] == ctx["lRef"]:
            break

        if ctx["indexEntry1"] >= len(probVect1):
            raise RuntimeError(
                f"mergeVectors overflow: pos={ctx['pos']}, "
                f"index1={ctx['indexEntry1']}, len1={len(probVect1)}"
            )

        if ctx["entry1"][0] < 4 or ctx["entry1"][0] == 6:
            ctx["indexEntry1"] += 1
            
            ctx["entry1"] = probVect1[ctx["indexEntry1"]]
        elif ctx["pos"] == ctx["entry1"][1]:
            ctx["indexEntry1"] += 1
            # if ctx["indexEntry1"] >= len(probVect1):
            #     raise RuntimeError(
            #         f"mergeVectors overflow: pos={ctx['pos']}, "
            #         f"index1={ctx['indexEntry1']}, len1={len(probVect1)}"
            #     )
            ctx["entry1"] = probVect1[ctx["indexEntry1"]]

        if ctx["entry2"][0] < 4 or ctx["entry2"][0] == 6:
            ctx["indexEntry2"] += 1
            ctx["entry2"] = probVect2[ctx["indexEntry2"]]
        elif ctx["pos"] == ctx["entry2"][1]:
            ctx["indexEntry2"] += 1
            ctx["entry2"] = probVect2[ctx["indexEntry2"]]

    if ctx["returnLK"]:
        return ctx["probVect"], ctx["cumulPartLk"] + log(ctx["totalFactor"])
    else:
        return ctx["probVect"]
# def findProbRoot(tree, node):
#     """ Search for root that maximizes overall likelihood given current vectors/BLens.

# Inputs: ['tree', 'node']
# Outputs: see return docs in MAPLE source.
# """
#     pass


# TODO: don't need
def rootVector(tree, node):
    """ Compute likelihood vector at (candidate) root by merging child partials.

Inputs: ['tree', 'node']
Outputs: see return docs in MAPLE source.
"""
    pass

def advance_entry_if_needed_deriv(ctx, probVect, which):
    """
    Advance entry index if the current position has been fully processed.
    Used specifically in estimateBranchLengthWithDerivative.
    
    Args:
        ctx: Dictionary containing context (entry1, entry2, indexEntry1, indexEntry2, pos)
        probVect: The probability vector to advance through
        which: 1 for entry1, 2 for entry2
        
    Returns:
        bool: True if we can continue, False if we've reached the end
    """
    indexKey = f"indexEntry{which}"
    entryKey = f"entry{which}"
    
    entry = ctx[entryKey]
    
    # Single-site entries (nucleotides or type O)
    if entry[0] < 4 or entry[0] == 6:
        ctx[indexKey] += 1
    
    # Stretch entries (Reference or N-type)
    elif ctx["pos"] == entry[1]:
        ctx[indexKey] += 1
    
    # Check if we've gone past the end
    if ctx[indexKey] >= len(probVect):
        return False
    
    # Update the entry
    ctx[entryKey] = probVect[ctx[indexKey]]
    return True


def estimateBranchLengthWithDerivative(probVectP, probVectC, fromTipC=False, 
                                        errorRateGlobalPassed=None, 
                                        mutMatrixGlobalPassed=None, 
                                        errorRatesGlobal=None, 
                                        mutMatricesGlobal=None, 
                                        cumulativeRateGlobal=None):
    """
    Estimate branch length maximizing likelihood; optionally use derivatives.

    Inputs: ['probVectP', 'probVectC', 'fromTipP', 'fromTipC', 'mutMatrix', 'minBL', 'maxBL', 'precision', 'errorRate', 'pseudoCountsGlobal', 'mutMatricesGlobal', 'cumulativeRateGlobal']
    Outputs: see return docs in MAPLE source.
    """
    global lref
    mutMatricesUsed = None  # because rateLimiter is false, this is not used
    if mutMatrixGlobalPassed != None:
        mutMatrix = mutMatrixGlobalPassed
    else:
        mutMatrix = mutMatrixGlobal

    # Note: usingErrorRate and errorRateSiteSpecific always set to False
    if usingErrorRate and errorRateSiteSpecific:
        if errorRatesGlobal != None:
            errorRatesUsed = errorRatesGlobal
        else:
            errorRatesUsed = errorRates
    else:
        if errorRateGlobalPassed == None:
            errorRate = errorRateGlobal
        else:
            errorRate = errorRateGlobalPassed
    
    if cumulativeRateGlobal != None:
        cumulativeRateUsed = cumulativeRateGlobal
    else:
        cumulativeRateUsed = cumulativeRate
        
    if not (usingErrorRate and errorRateSiteSpecific):
        errorRate = errorRateGlobal
    if not useRateVariation:
        mutMatrix = mutMatrixGlobal
  
    c1 = globalTotRate
    ais = []
    
    # Initialize context dictionary
    ctx = {
        "indexEntry1": 0,
        "indexEntry2": 0,
        "pos": 0,
        "entry1": probVectP[0],
        "entry2": probVectC[0]
    }
    
    contribLength, nZeros = 0.0, 0
    
    while True:
        entry1 = ctx["entry1"]
        entry2 = ctx["entry2"]
        
        if entry2[0] == 5:
            if entry1[0] == 4 or entry1[0] == 5:
                end = min(entry1[1], entry2[1])
            else:
                end = ctx["pos"] + 1
            c1 += (cumulativeRateUsed[ctx["pos"]] - cumulativeRateUsed[end])
            ctx["pos"] = end
            
        elif entry1[0] == 5:  # case entry1 or entry2 is N
            # if parent node is type "N", in theory we might have to calculate the contribution of root nucleotides; 
            # however, if this node is "N" then every other node in the current tree is "N", so we can ignore this since this contribution cancels out in relative terms.
            if entry2[0] == 4:
                end = min(entry1[1], entry2[1])
            else:
                end = ctx["pos"] + 1
            c1 += (cumulativeRateUsed[ctx["pos"]] - cumulativeRateUsed[end])
            ctx["pos"] = end
            
        else:
            # below, when necessary, we represent the likelihood as coeff0*l +coeff1, where l is the branch length to be optimized.
            if entry1[0] == 4 and entry2[0] == 4:  # case entry1 and entry2 are R
                ctx["pos"] = min(entry1[1], entry2[1])
            else:
                if useRateVariation and mutMatricesUsed is not None:
                    mutMatrix = mutMatricesUsed[ctx["pos"]]
                
                if entry1[0] == 4:
                    c1 -= mutMatrix[entry2[1]][entry2[1]]
                else:
                    c1 -= mutMatrix[entry1[1]][entry1[1]]
                    
                flag1 = (usingErrorRate and (entry1[0] != 6) and len(entry1) > 2 and entry1[-1])  # flag1 true if error rate applies to entry1
                flag2 = (usingErrorRate and (entry2[0] != 6) and (fromTipC or (len(entry2) > 2 and entry2[-1])))
                if usingErrorRate and errorRateSiteSpecific and errorRatesUsed is not None: 
                    errorRate = errorRatesUsed[ctx["pos"]]

                # contribLength will be here the total length from the root or from the upper node, down to the down node.
                contribLength = False
                if entry1[0] < 5:
                    if len(entry1) == 3 + usingErrorRate:
                        contribLength = entry1[2]
                    elif len(entry1) == 4 + usingErrorRate:
                        contribLength = entry1[3]
                else:
                    if len(entry1) > 3:
                        contribLength = entry1[2]
                if entry2[0] < 5:
                    if len(entry2) > 2 + usingErrorRate:
                        contribLength += entry2[2]
                else:
                    if len(entry2) > 3:
                        contribLength += entry2[2]

                if entry1[0] == 4:
                    # entry1 is reference and entry2 is of type "O"
                    if entry2[0] == 6:
                        i1 = entry2[1]
                        if len(entry1) == (4 + usingErrorRate):
                            coeff0 = rootFreqs[i1] * entry2[-1][i1]
                            coeff1 = 0.0
                            for i in range(4):
                                coeff0 += rootFreqs[i] * mutMatrix[i][i1] * entry1[2] * entry2[-1][i]
                                coeff1 += mutMatrix[i1][i] * entry2[-1][i]
                            coeff1 *= rootFreqs[i1]
                            if contribLength:
                                coeff0 += coeff1 * contribLength
                            if flag1 and errorRate is not None:
                                coeff0 -= 1.33333 * errorRate * rootFreqs[i1] * entry2[-1][i1]
                                for i in range(4):
                                    coeff0 += rootFreqs[i] * entry2[-1][i] * 0.33333 * errorRate
                        else:
                            coeff0 = entry2[-1][i1]
                            coeff1 = 0.0
                            for j in range(4):
                                coeff1 += mutMatrix[i1][j] * entry2[-1][j]
                            if contribLength:
                                coeff0 += coeff1 * contribLength
                        if coeff1 < 0.0:
                            c1 += coeff1 / coeff0
                        elif coeff1:
                            coeff0 = coeff0 / coeff1
                            ais.append(coeff0)
                        ctx["pos"] += 1

                    else:  # entry1 is R and entry2 is a different but single nucleotide
                        if len(entry1) == 4 + usingErrorRate:
                            i1 = entry2[1]
                            i2 = entry2[0]
                            coeff0 = rootFreqs[i2] * mutMatrix[i2][i1] * entry1[2]
                            if contribLength:
                                coeff0 += rootFreqs[i1] * mutMatrix[i1][i2] * contribLength
                            if flag2:
                                coeff0 += rootFreqs[i1] * 0.33333 * errorRate
                            if flag1:
                                coeff0 += rootFreqs[i2] * 0.33333 * errorRate
                            coeff1 = rootFreqs[i1] * mutMatrix[i1][i2]
                            if coeff1:
                                coeff0 = coeff0 / coeff1
                            else:
                                coeff0 = None
                        else:
                            coeff0 = contribLength
                            if flag2 and errorRate is not None:
                                if mutMatrix[entry2[1]][entry2[0]]:
                                    coeff0 += errorRate * 0.33333 / mutMatrix[entry2[1]][entry2[0]]
                                else:
                                    coeff0 = None
                        if coeff0 != None:
                            if coeff0:
                                ais.append(coeff0)
                            else:
                                nZeros += 1
                        ctx["pos"] += 1

                # entry1 is of type "O"
                elif entry1[0] == 6:
                    if entry2[0] == 6:
                        coeff0 = entry1[-1][0] * entry2[-1][0] + entry1[-1][1] * entry2[-1][1] + entry1[-1][2] * entry2[-1][2] + entry1[-1][3] * entry2[-1][3]
                        coeff1 = 0.0
                        for i in range(4):
                            for j in range(4):
                                coeff1 += entry1[-1][i] * entry2[-1][j] * mutMatrix[i][j]
                        if contribLength:
                            coeff0 += coeff1 * contribLength
                    else:  # entry1 is "O" and entry2 is a nucleotide
                        if entry2[0] == 4:
                            i2 = entry1[1]
                        else:
                            i2 = entry2[0]
                        coeff0 = entry1[-1][i2]
                        coeff1 = 0.0
                        for i in range(4):
                            coeff1 += entry1[-1][i] * mutMatrix[i][i2]
                        if contribLength:
                            coeff0 += coeff1 * contribLength
                        if flag2 and errorRate is not None:
                            coeff0 += errorRate * 0.33333
                    if coeff1 < 0.0:
                        c1 += coeff1 / coeff0
                    elif coeff1:
                        coeff0 = coeff0 / coeff1
                        ais.append(coeff0)
                    ctx["pos"] += 1
                    
                else:  # entry1 is a non-ref nuc
                    if entry2[0] == entry1[0]:
                        c1 += mutMatrix[entry1[0]][entry1[0]]
                    else:  # entry1 is a nucleotide and entry2 is not the same as entry1
                        i1 = entry1[0]
                        if entry2[0] < 5:  # entry2 is a nucleotide
                            if entry2[0] == 4:
                                i2 = entry1[1]
                            else:
                                i2 = entry2[0]

                            if len(entry1) == 4 + usingErrorRate:
                                coeff0 = rootFreqs[i2] * mutMatrix[i2][i1] * entry1[2]
                                if contribLength:
                                    coeff0 += rootFreqs[i1] * mutMatrix[i1][i2] * contribLength
                                if flag2:
                                    coeff0 += rootFreqs[i1] * 0.33333 * errorRate
                                if flag1:
                                    coeff0 += rootFreqs[i2] * 0.33333 * errorRate
                                coeff1 = rootFreqs[i1] * mutMatrix[i1][i2]
                                if coeff1:
                                    coeff0 = coeff0 / coeff1
                                else:
                                    coeff0 = None
                            else:
                                coeff0 = contribLength
                                if flag2 and errorRate is not None:
                                    coeff0 += errorRate * 0.33333 / mutMatrix[i1][i2]
                            if coeff0 != None:
                                if coeff0:
                                    ais.append(coeff0)
                                else:
                                    nZeros += 1

                        else:  # entry1 is a nucleotide and entry2 is of type "O"
                            if len(entry1) == 4 + usingErrorRate:
                                coeff0 = rootFreqs[i1] * entry2[-1][i1]
                                coeff1 = 0.0
                                for i in range(4):
                                    coeff0 += rootFreqs[i] * mutMatrix[i][i1] * entry1[2] * entry2[-1][i]
                                    coeff1 += mutMatrix[i1][i] * entry2[-1][i]
                                coeff1 *= rootFreqs[i1]
                                if contribLength:
                                    coeff0 += coeff1 * contribLength
                                if flag1 and errorRate is not None:
                                    coeff0 -= 1.33333 * errorRate * rootFreqs[i1] * entry2[-1][i1]
                                    for i in range(4):
                                        coeff0 += rootFreqs[i] * entry2[-1][i] * 0.33333 * errorRate
                            else:
                                coeff0 = entry2[-1][i1]
                                coeff1 = 0.0
                                for j in range(4):
                                    coeff1 += mutMatrix[i1][j] * entry2[-1][j]
                                if contribLength:
                                    coeff0 += coeff1 * contribLength
                            if coeff1 < 0.0:
                                c1 += coeff1 / coeff0
                            elif coeff1:
                                coeff0 = coeff0 / coeff1
                                ais.append(coeff0)
                        ctx["pos"] += 1

        # Check if we've reached the end
        if ctx["pos"] == lref:
            break
            
        # Advance entries using the new helper function
        if not advance_entry_if_needed_deriv(ctx, probVectP, 1):
            break
        if not advance_entry_if_needed_deriv(ctx, probVectC, 2):
            break

    # now optimized branch length based on coefficients
    c1 = -c1
    n = len(ais) + nZeros
    if n == 0:
        return False
    else:
        if len(ais):
            minAis = min(ais)
        else:
            minAis = 0.0
        if nZeros:
            minAis = min(0.0, minAis)
        if minAis < 0.0:
            return 0.1
        tDown = min(0.1, n / c1 - minAis)
        if tDown <= 0.0:
            return False
        if nZeros:
            vDown = nZeros / tDown
        else:
            vDown = 0.0
        for ai in ais:
            vDown += 1.0 / (ai + tDown)
        if len(ais):
            maxAis = max(ais)
        else:
            maxAis = 0.0
        tUp = min(0.1, n / c1 - maxAis)
        if tUp >= 0.1:
            return 0.1
        if tUp <= minBLenSensitivity:
            if minAis:
                tUp = 0.0
            else:
                tUp = minBLenSensitivity
        if nZeros:
            vUp = nZeros / tUp
        else:
            vUp = 0.0
        for ai in ais:
            vUp += 1.0 / (ai + tUp)
    if vDown > c1 + minBLenSensitivity or vUp < c1 - minBLenSensitivity:
        if vUp < c1 - minBLenSensitivity and (not tUp):
            return False
        if (vDown > c1 + minBLenSensitivity) and tDown >= 0.1:
            return 0.1
        print("Initial border parameters don't fit expectations")

    while tDown - tUp > minBLenSensitivity:
        tMiddle = (tUp + tDown) / 2
        if nZeros:
            vMiddle = nZeros / tMiddle
        else:
            vMiddle = 0.0
        for ai in ais:
            vMiddle += 1.0 / (ai + tMiddle)
        if vMiddle > c1:
            tUp = tMiddle
        else:
            tDown = tMiddle

    return tUp


def updateBLen(tree, cNode, addToList, nodeList):
    """ Commit a branch-length change and update impacted node lists/vectors.

    Inputs: ['tree', 'cNode', 'addToList', 'nodeList']
    Outputs: see return docs in MAPLE source.
    """
    # store local variables
    def _min_blen():
        # minBLenSensitivity is set in main(); fall back to a small value for safety.
        try:
            return float(minBLenSensitivity) if minBLenSensitivity else 1e-8
        except Exception:
            return 1e-8

    def _max_blen():
        # Hard cap is intentionally generous; we rely on a soft prior below.
        # (MAPLE's derivative estimator itself is bounded at 0.1 in this scaffold.)
        return 0.1

    def _prior_params():
        # Soft exponential prior to discourage runaway branch lengths while still
        # allowing the optimizer to exceed the old 2e-3 cap when the data supports it.
        try:
            base = float(oneMutBLen) if oneMutBLen else 3e-5
        except Exception:
            base = 3e-5
        # Scale around typical 1-mutation length, but not too tiny.
        prior_scale = float(np.clip(50.0 * base, 2e-4, 5e-3))
        prior_strength = 10.0
        return prior_strength, prior_scale

    def _sanitize_blen(x, fallback=None):
        if fallback is None:
            fallback = _min_blen()
        if x is None or x is False:
            return float(fallback)
        try:
            xf = float(x)
        except Exception:
            return float(fallback)
        if not np.isfinite(xf) or xf <= 0.0:
            return float(fallback)
        if xf < _min_blen():
            return float(_min_blen())
        if xf > _max_blen():
            return float(_max_blen())
        return xf

    def _bLen_candidates():
        # Candidate grid for fallback/regularization. Include lengths beyond the
        # old 2e-3 cap; the soft prior handles preference for shorter lengths.
        try:
            base = float(oneMutBLen) if oneMutBLen else 3e-5
        except Exception:
            base = 3e-5
        mins = _min_blen()
        cands = [
            mins,
            1e-7,
            3e-7,
            1e-6,
            3e-6,
            1e-5,
            3e-5,
            base,
            base * 3.0,
            base * 10.0,
            1e-4,
            3e-4,
            1e-3,
            2e-3,
            3e-3,
            5e-3,
            1e-2,
            2e-2,
            5e-2,
        ]
        # sanitize + dedupe + sort
        out = []
        seen = set()
        for v in cands:
            vv = _sanitize_blen(v)
            key = round(vv, 16)
            if key in seen:
                continue
            seen.add(key)
            out.append(vv)
        out.sort()
        return out

    parents = tree.up
    dirty = tree.dirty
    probDown = tree.probVect
    probUpLeft = tree.probVectUpLeft
    probUpRight = tree.probVectUpRight
    children = tree.children
    distances = tree.dist
    parent = parents[cNode]
    if parent is None:
        return

    # Determine the correct "up" vector at the parent excluding this child.
    cIdx = 0
    if len(children[parent]) == 1:
        vectUp = tree.probVectTotUp[parent]
    else:
        if cNode == children[parent][0]:
            cIdx = 0
            vectUp = probUpRight[parent]
        else:
            cIdx = 1
            vectUp = probUpLeft[parent]

    vectDown = probDown[cNode]
    if vectUp is None or vectDown is None:
        return

    fromTipC = (len(children[cNode]) == 0)

    # Use derivative estimator as a proposal, but always score with a soft prior.
    proposed = estimateBranchLengthWithDerivative(vectUp, vectDown, fromTipC=fromTipC)
    proposed = None if (proposed is False or proposed is None) else _sanitize_blen(proposed)

    prior_strength, prior_scale = _prior_params()

    bestScore = float('-inf')
    bestLen = None
    cand_set = list(_bLen_candidates())
    if proposed is not None:
        cand_set.append(proposed)

    for cand in cand_set:
        cand = _sanitize_blen(cand)
        try:
            merged = mergeVectors(
                probVect1=vectUp,
                bLen1=0.0,
                fromTip1=False,
                probVect2=vectDown,
                bLen2=cand,
                fromTip2=fromTipC,
                returnLK=True,
                isUpDown=True,
            )
        except Exception:
            continue
        if merged is None:
            continue
        _, lk = merged
        score = lk - prior_strength * (cand / prior_scale)
        if score > bestScore:
            bestScore = score
            bestLen = cand

    distances[cNode] = _sanitize_blen(bestLen, fallback=distances[cNode] if distances[cNode] else None)

    dirty[parent] = True
    dirty[cNode] = True
    if addToList:
        nodeList.append((cNode, 2, True, False))
        nodeList.append((parent, cIdx, True, False))
    


def compare_entry_type(e1,e2) :
    return e1[0] == e2[0]

def compare_entry_lengths(e1, e2):
    return len(e1) == len(e2)

def compare_ACGTR_entry(entry1 ,entry2) :
    """Compare A/C/G/T/R entries.

    Many genome-list entries are only 2-tuples like (state, posOrEnd). Branch-length
    fields are optional, so only compare them when present.
    """
    if len(entry1) < 3 or len(entry2) < 3:
        return False
    if abs(entry1[2] - entry2[2]) > THRESHOLD:  # compare branch lengths
        return True
    return False

def compare_O_entry(entry1, entry2) :
    def _is_number(x):
        return isinstance(x, (int, float, np.floating))

    def _is_prob_vec(x):
        return (
            isinstance(x, (list, tuple, np.ndarray))
            and len(x) == 4
            and all(_is_number(v) for v in x)
        )

    # Some (6, ...) entries are (6, refNuc, probVec) where entry[2] is a vector.
    # Others carry branch-length fields and put the vector at the end.
    if len(entry1) >= 3 and len(entry2) >= 3 and _is_number(entry1[2]) and _is_number(entry2[2]):
        if abs(entry1[2] - entry2[2]) > THRESHOLD:
            return True

    v1 = entry1[-1] if len(entry1) >= 3 else None
    v2 = entry2[-1] if len(entry2) >= 3 else None
    if _is_prob_vec(v1) and _is_prob_vec(v2):
        for i in range(4):
            diffVal = abs(float(v1[i]) - float(v2[i]))
            if diffVal > thresholdDiffForUpdate:
                return True
            denom1 = float(v1[i]) if float(v1[i]) != 0.0 else None
            denom2 = float(v2[i]) if float(v2[i]) != 0.0 else None
            if diffVal > THRESHOLD:
                if denom1 is not None and (diffVal / denom1) > thresholdFoldChangeUpdate:
                    return True
                if denom2 is not None and (diffVal / denom2) > thresholdFoldChangeUpdate:
                    return True
        return False

    # Fallback: if shapes/payloads differ and we can't compare numerically, treat as different.
    return entry1 != entry2

def update_singular_pos(pos) :
    return pos + 1

def update_contiguous_pos(entry1, entry2) :
    return min(entry1[1], entry2[1])


        

def areVectorsDifferent(probVect1, probVect2):
    """ Return True if two probability vectors differ beyond thresholds.

Inputs: ['probVect1', 'probVect2']
Outputs: see return docs in MAPLE source.
"""
    if probVect1 is probVect2:
        return False
    if probVect1 is None or probVect2 is None :
        return False
    pos = 0
    for i in range (len(probVect1)) :
        entry1 = probVect1[i]
        entry2 = probVect2[i]
        if (not compare_entry_type(entry1, entry2)) :
            return True
        if (not compare_entry_lengths(entry1, entry2)) :
            return True
        if entry1[0] < 5: #types ACGTR
            # For 2-tuples, the payload differs by type:
            # - A/C/G/T: (state, refNucAtSite)
            # - R: (4, endPos)
            if len(entry1) == 2:
                if entry1[1] != entry2[1]:
                    return True
            else:
                if (compare_ACGTR_entry(entry1, entry2)):
                    return True
            if entry1[0] <= 3: #ACGT
                pos = update_singular_pos(pos)
            else : #type R
                pos = update_contiguous_pos(entry1, entry2)

        elif entry1[0] == 5: #type N
               if entry1[1] != entry2[1]:
                  return True
               pos = update_contiguous_pos(entry1, entry2)

        elif entry1[0] == 6: # type 6
            if (compare_O_entry(entry1, entry2)) :
                return True
            pos = update_contiguous_pos(entry1, entry2)

        if pos == lref: # lref is length of reference sequence
            break
    return False


def _find_root_index(tree):
    for i, p in enumerate(tree.up):
        if p is None:
            return i
    return 0


def _all_N_vector():
    global lref
    if lref is None:
        raise ValueError("lref must be set before updating partials")
    return [(5, lref)]


def _is_tip(tree, node_idx):
    return len(tree.children[node_idx]) == 0




def _preorder_nodes(tree, root_idx):
    order = []
    stack = [root_idx]
    while stack:
        node = stack.pop()
        order.append(node)
        children = tree.children[node]
        for ch in reversed(children):
            stack.append(ch)
    return order


def _recompute_down_at_node(tree, node_idx):
    """Recompute tree.probVect[node_idx] from children (if internal).

    Assumes children probVects are already up-to-date.
    """
    children = tree.children[node_idx]
       # DO NOT recompute until all children have valid probVect
    for ch in children:
        if tree.probVect[ch] is None:
            return None
    if len(children) == 0:
        return tree.probVect[node_idx]
    if len(children) == 1:
        ch = children[0]
        return mergeVectors(
            probVect1=tree.probVect[ch],
            bLen1=tree.dist[ch],
            fromTip1=_is_tip(tree, ch),
            probVect2=_all_N_vector(),
            bLen2=0.0,
            fromTip2=False,
            returnLK=False,
            isUpDown=False,
        )
    if len(children) != 2:
        raise ValueError(f"updatePartials currently supports binary trees; node {node_idx} has {len(children)} children")
    left, right = children[0], children[1]
    return mergeVectors(
        probVect1=tree.probVect[left],
        bLen1=tree.dist[left],
        fromTip1=_is_tip(tree, left),
        probVect2=tree.probVect[right],
        bLen2=tree.dist[right],
        fromTip2=_is_tip(tree, right),
        returnLK=False,
        isUpDown=False,
    )

from collections import deque
def updatePartials(tree, nodeList=None, force=False):
    """MAPLE-style partial update.

    Maintains:
    - tree.probVect: downward/subtree partials at each node
    - tree.probVectTotUp: upward partials from outside the node's subtree
    - tree.probVectUpRight / tree.probVectUpLeft: per-parent vectors excluding each child

    Args:
        tree: Tree
        nodeList: optional list of (nodeIdx, childIdx, updateDown, updateUp) tuples
        force: if True, recompute all partials regardless of dirty flags
    """
    root_idx = _find_root_index(tree)

    # ---- Bottom-up: update downward partials where needed ----
    update_down_nodes = set()
    if nodeList:
        for entry in nodeList:
            if not entry:
                continue
            node_idx = entry[0]
            update_down = True
            if len(entry) >= 3:
                update_down = bool(entry[2])
            if update_down:
                update_down_nodes.add(node_idx)

    if force:
        update_down_nodes = set(range(len(tree.up)))
    else:
        for i, is_dirty in enumerate(tree.dirty):
            if is_dirty:
                update_down_nodes.add(i)

    # ---- Incremental bottom-up update driven by changes ----
    work = deque(update_down_nodes)
    scheduled = set(work)
    changed = set()

    while work:
        node_idx = work.pop()
        scheduled.discard(node_idx)
        old_vect = tree.probVect[node_idx]
        new_vect = _recompute_down_at_node(tree, node_idx)
        if new_vect is None:
        # children not ready yet → retry later
            work.appendleft(node_idx)
            continue

        if old_vect is None or areVectorsDifferent(old_vect, new_vect):
            tree.probVect[node_idx] = new_vect
            tree.dirty[node_idx] = False
            changed.add(node_idx)

            parent = tree.up[node_idx]
            if parent is not None:
                if parent not in scheduled:
                    work.append(parent)
                    scheduled.add(parent)
    # Nodes whose upward vectors may be affected
    affected = set(changed)
    for n in list(affected):
        p = tree.up[n]
        if p is not None:
            affected.add(p)

    # ---- Top-down: recompute upward partials and per-child excluding vectors ----
    # We keep this step global for correctness; it is still MAPLE-style (totUp + per-child up vectors).
    tree.probVectTotUp[root_idx] = _all_N_vector()
    tree.probVectUpLeft[root_idx] = None
    tree.probVectUpRight[root_idx] = None

    for node_idx in _preorder_nodes(tree, root_idx):
        if node_idx not in affected:
            continue
        children = tree.children[node_idx]
        if len(children) == 0:
            continue
        if len(children) == 1:
            ch = children[0]
            # No sibling contribution.
            up_excl_child_at_node = tree.probVectTotUp[node_idx]
            tree.probVectUpRight[node_idx] = up_excl_child_at_node
            tree.probVectUpLeft[node_idx] = up_excl_child_at_node
            # Propagate to child through the branch (N-vector is neutral).
            tree.probVectTotUp[ch] = mergeVectors(
                probVect1=up_excl_child_at_node,
                bLen1=tree.dist[ch],
                fromTip1=False,
                probVect2=_all_N_vector(),
                bLen2=0.0,
                fromTip2=False,
                returnLK=False,
                isUpDown=True,
            )
            continue
        if len(children) != 2:
            raise ValueError(f"updatePartials currently supports binary trees; node {node_idx} has {len(children)} children")

        left, right = children[0], children[1]

        # Vector at node excluding LEFT child (uses RIGHT sibling).
        up_excl_left_at_node = mergeVectors(
            probVect1=tree.probVectTotUp[node_idx],
            bLen1=0.0,
            fromTip1=False,
            probVect2=tree.probVect[right],
            bLen2=tree.dist[right],
            fromTip2=_is_tip(tree, right),
            returnLK=False,
            isUpDown=True,
        )
        # Vector at node excluding RIGHT child (uses LEFT sibling).
        up_excl_right_at_node = mergeVectors(
            probVect1=tree.probVectTotUp[node_idx],
            bLen1=0.0,
            fromTip1=False,
            probVect2=tree.probVect[left],
            bLen2=tree.dist[left],
            fromTip2=_is_tip(tree, left),
            returnLK=False,
            isUpDown=True,
        )

        # Store per-child excluding vectors at this internal node.
        # Naming matches updateBLen(): for left child we need the "right"-side vector, and vice-versa.
        tree.probVectUpRight[node_idx] = up_excl_left_at_node
        tree.probVectUpLeft[node_idx] = up_excl_right_at_node

        # Propagate totUp to each child through its branch.
        tree.probVectTotUp[left] = mergeVectors(
            probVect1=up_excl_left_at_node,
            bLen1=tree.dist[left],
            fromTip1=False,
            probVect2=_all_N_vector(),
            bLen2=0.0,
            fromTip2=False,
            returnLK=False,
            isUpDown=True,
        )
        tree.probVectTotUp[right] = mergeVectors(
            probVect1=up_excl_right_at_node,
            bLen1=tree.dist[right],
            fromTip1=False,
            probVect2=_all_N_vector(),
            bLen2=0.0,
            fromTip2=False,
            returnLK=False,
            isUpDown=True,
        )

    return True


      
#createNewick() (and its helpers) starts here
# Helper to format branch length
def _bl_str(bl):
    if bl is None: return ""
    try:
        blf = float(bl)
        # Preserve small-but-nonzero branch lengths.
        if blf == 0.0:
            return ":0"
        if abs(blf) < 1e-6:
            return ":" + ("{:.12g}".format(blf))
        return ":" + ("{:.8f}".format(blf).rstrip("0").rstrip("."))
    except Exception:
        return ":" + str(bl)

# Helper to generate node label (name, support, mutations, lineage)
def _label(tree, nodeIdx, includeSupports, minSupport, includeMutationList, performLineageAssignmentByRefPlacement):
    """
    Generate label string for a node in Newick format.
    
    Args:
        tree: Tree object with list-based node data
        nodeIdx: Integer index of the node
        includeSupports: Whether to include support values
        minSupport: Minimum support value to display
        includeMutationList: Whether to include mutation annotations
        performLineageAssignmentByRefPlacement: Whether to include lineage info
    
    Returns:
        String label for the node (may be empty for internal nodes without annotations)
    """
    parts = []
    
    # 1. Get node name (for tips/leaves)
    name = tree.name[nodeIdx] if nodeIdx < len(tree.name) else None
    
    # Add name if it exists and is non-empty
    if name is not None and name != "":
        parts.append(str(name))
    
    # 2. Lineage annotation (if applicable)
    if performLineageAssignmentByRefPlacement:
        # Check if tree has a lineage attribute/list
        if hasattr(tree, 'lineage') and nodeIdx < len(tree.lineage):
            lineage = tree.lineage[nodeIdx]
            if lineage:  # Only add if lineage exists
                parts.append(f"lineage={lineage}")
    
    # 3. Support values for internal nodes
    if includeSupports:
        # Check if tree has support values
        if hasattr(tree, 'support') and nodeIdx < len(tree.support):
            support = tree.support[nodeIdx]
            
            # Only include if support meets minimum threshold
            if support is not None:
                try:
                    sup_float = float(support)
                    
                    # Check against minimum support threshold
                    if minSupport is None or sup_float >= float(minSupport):
                        # Format as integer if it's a whole number
                        if abs(sup_float - int(sup_float)) < 1e-6:
                            parts.append(f"support={int(sup_float)}")
                        else:
                            parts.append(f"support={sup_float:.3f}")
                except (ValueError, TypeError):
                    # If conversion fails, skip support
                    pass
    
    # 4. Mutation list
    if includeMutationList:
        # Check if tree has mutations
        if hasattr(tree, 'mutations') and nodeIdx < len(tree.mutations):
            muts = tree.mutations[nodeIdx]
            
            if muts:  # Only add if mutations exist
                # Format mutation list
                if isinstance(muts, (list, tuple)) and len(muts) > 0:
                    # Convert mutation entries to strings
                    # Assuming mutations are like [(pos, from_base, to_base), ...]
                    mut_str = ",".join(str(m) for m in muts)
                    parts.append(f"mut={mut_str}")
                elif isinstance(muts, str):
                    parts.append(f"mut={muts}")
    
    # 5. Assemble the final label
    if not parts:
        return ""  # No label needed
    
    # Separate name from metadata
    if name is not None and name != "" and len(parts) > 0 and parts[0] == str(name):
        # First part is the name
        name_part = parts[0]
        meta_parts = parts[1:]
    else:
        # No name, all parts are metadata
        name_part = ""
        meta_parts = parts
    
    # Format: "name[&key=value,key=value,...]"
    if meta_parts:
        meta_string = f"[&{','.join(meta_parts)}]"
        return f"{name_part}{meta_string}"
    else:
        return name_part


def createNewick(tree, root_node_id,
                 namesInTree=None,
                 includeMutationList=False,
                 minSupport=None,
                 count0BLenNodesOnce=False,
                 includeSupports=False,
                 performLineageAssignmentByRefPlacement=False):
    """
    Serialize subtree rooted at `root_node_id` to a Newick string.

    NOTE: The earlier implementation assumed the tree was strictly binary and
    would crash if an internal node had != 2 children. This version supports
    arbitrary out-degree (0/1/2+), which is valid Newick.
    """

    children = tree.children
    dist = tree.dist

    def _subtree_to_newick(nodeIdx):
        node_children = children[nodeIdx] if nodeIdx < len(children) else []
        is_leaf = not node_children

        label_str = _label(
            tree,
            nodeIdx,
            includeSupports,
            minSupport,
            includeMutationList,
            performLineageAssignmentByRefPlacement,
        )

        if is_leaf:
            body = label_str
        else:
            body = "(" + ",".join(_subtree_to_newick(c) for c in node_children) + ")" + label_str

        node_bl = dist[nodeIdx] if nodeIdx < len(dist) else None
        return body + _bl_str(node_bl)

    return _subtree_to_newick(root_node_id) + ";"

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


### Hannah's Note: for main function we need a few more functions.
# findBestParentForNewSample()
def findBestParentForNewSample(tree, newSampleDiffs, ref_seq, mutMatrix):
    """
    -Find the best node in the tree to append a new sample.
    -The algorithm traverses the tree and tries to append the sampe at eahc node and mid-branch node.
    -It will start traversing when certain criteria are met
    
    :param tree: Description
    :param newSampleDiffs: Description
    :param ref_seq: Description
    :param mutMatrix: 4x4 subtituion matrix
    
    returns:
    (bestNode, bestBranchLen, bestLikelilhood)
    - bestNode: index of node where sample should attach 
    - bestBranchLen: optimal length of branch to attach
    - bestlikelihood: log-likelihood of best placecment 
    """
    """Edge-based placement: choose the best existing edge (parent->child) to split.

    Returns:
        (bestParent, bestChild, lenParentToInternal, lenInternalToChild, newLeafLen, bestLogLK)
    """

    def _min_blen():
        try:
            return float(minBLenSensitivity) if minBLenSensitivity else 1e-8
        except Exception:
            return 1e-8

    def _max_blen():
        # generous bound; soft priors drive the effective range
        return 0.1

    def _prior_params():
        try:
            base = float(oneMutBLen) if oneMutBLen else 3e-5
        except Exception:
            base = 3e-5
        prior_scale = float(np.clip(50.0 * base, 2e-4, 5e-3))
        prior_strength = 10.0
        # penalize changing total split length too far from the current edge length
        sum_dev_strength = 5.0
        return prior_strength, prior_scale, sum_dev_strength

    def _sanitize_blen(x, fallback=None):
        if fallback is None:
            fallback = _min_blen()
        if x is None or x is False:
            return float(fallback)
        try:
            xf = float(x)
        except Exception:
            return float(fallback)
        if not np.isfinite(xf) or xf <= 0.0:
            return float(fallback)
        if xf < _min_blen():
            return float(_min_blen())
        if xf > _max_blen():
            return float(_max_blen())
        return xf

    def _bLen_candidates():
        try:
            base = float(oneMutBLen) if oneMutBLen else 3e-5
        except Exception:
            base = 3e-5
        mins = _min_blen()
        cands = [
            mins,
            1e-7,
            3e-7,
            1e-6,
            3e-6,
            1e-5,
            3e-5,
            base,
            base * 3.0,
            base * 10.0,
            1e-4,
            3e-4,
            1e-3,
            2e-3,
            3e-3,
            5e-3,
            1e-2,
        ]
        out = []
        seen = set()
        for v in cands:
            vv = _sanitize_blen(v)
            key = round(vv, 16)
            if key in seen:
                continue
            seen.add(key)
            out.append(vv)
        out.sort()
        return out

    bestParent = None
    bestChild = None
    bestLenP = None
    bestLenC = None
    bestNewLen = None
    bestLikelihood = float('-inf')
    bestScoreGlobal = float('-inf')

    # Likelihood vector for the new sample leaf
    newSampleProbVect = probVectTerminalNode(newSampleDiffs, tree, None, ref_seq)

    # Ensure we have valid up/down partials at all nodes
    # (caller typically keeps this up-to-date, but this makes the function safer).
    if tree.probVectTotUp[0] is None:
        updatePartials(tree, force=True)

    prior_strength, prior_scale, sum_dev_strength = _prior_params()

    # 1) Fast pre-screen: score every edge with a single cheap configuration,
    # then keep only the top-K edges for full optimization.
    probe_leaf_len = _sanitize_blen(oneMutBLen, fallback=3e-5)
    edge_infos = []

    for childIdx in range(len(tree.up)):
        parentIdx = tree.up[childIdx]
        if parentIdx is None:
            continue
        if tree.probVect[childIdx] is None:
            continue

        parent_children = tree.children[parentIdx]
        if len(parent_children) == 1:
            vectUpExclChild = tree.probVectTotUp[parentIdx]
        elif len(parent_children) == 2:
            if childIdx == parent_children[0]:
                vectUpExclChild = tree.probVectUpRight[parentIdx]
            else:
                vectUpExclChild = tree.probVectUpLeft[parentIdx]
        else:
            continue

        if vectUpExclChild is None:
            continue

        edgeLen0 = _sanitize_blen(tree.dist[childIdx], fallback=2.0 * _min_blen())
        childFromTip = _is_tip(tree, childIdx)

        # probe score: split in half, attach leaf with a fixed length
        top_probe = _sanitize_blen(edgeLen0 * 0.5)
        bot_probe = _sanitize_blen(edgeLen0 - top_probe)
        ins = mergeVectors(
            probVect1=vectUpExclChild,
            bLen1=top_probe,
            fromTip1=False,
            probVect2=tree.probVect[childIdx],
            bLen2=bot_probe,
            fromTip2=childFromTip,
            returnLK=True,
            isUpDown=True,
        )
        if ins is None:
            continue
        insertionBaseVect, lk_ins = ins
        if insertionBaseVect is None:
            continue
        merged = mergeVectors(
            probVect1=insertionBaseVect,
            bLen1=0.0,
            fromTip1=False,
            probVect2=newSampleProbVect,
            bLen2=probe_leaf_len,
            fromTip2=True,
            returnLK=True,
            isUpDown=False,
        )
        if merged is None:
            continue
        _, lk_leaf = merged
        lk_total = lk_ins + lk_leaf
        edge_infos.append((lk_total, parentIdx, childIdx, vectUpExclChild, edgeLen0, childFromTip))

    if not edge_infos:
        return bestParent, bestChild, bestLenP, bestLenC, bestNewLen, bestLikelihood

    edge_infos.sort(key=lambda t: t[0], reverse=True)
    topK = 40
    edge_infos = edge_infos[: min(topK, len(edge_infos))]

    # 2) Full (but still small) optimization on the shortlisted edges.
    try:
        base = float(oneMutBLen) if oneMutBLen else 3e-5
    except Exception:
        base = 3e-5

    split_fracs = [0.0, 0.25, 0.5, 0.75, 1.0]
    leaf_cands = [
        _min_blen(),
        1e-6,
        3e-6,
        base,
        base * 3.0,
        base * 10.0,
        3e-4,
        1e-3,
        3e-3,
        1e-2,
    ]
    leaf_cands = sorted({round(_sanitize_blen(v), 16) for v in leaf_cands})
    leaf_cands = [float(v) for v in leaf_cands]

    for (probeScore, parentIdx, childIdx, vectUpExclChild, edgeLen0, childFromTip) in edge_infos:
        edge_len_cands = [edgeLen0, edgeLen0 * 0.5, edgeLen0 * 2.0, _min_blen()]
        edge_len_cands = sorted({round(_sanitize_blen(v), 16) for v in edge_len_cands})
        edge_len_cands = [float(v) for v in edge_len_cands]

        bestLocalScore = float('-inf')
        bestLocalLK = float('-inf')
        bestLocalTop = None
        bestLocalBot = None
        bestLocalLeaf = None

        for edgeLenTry in edge_len_cands:
            dev_edge = abs(edgeLenTry - edgeLen0)
            for f in split_fracs:
                topLen = _sanitize_blen(edgeLenTry * f)
                botLen = _sanitize_blen(edgeLenTry - topLen)
                ins = mergeVectors(
                    probVect1=vectUpExclChild,
                    bLen1=topLen,
                    fromTip1=False,
                    probVect2=tree.probVect[childIdx],
                    bLen2=botLen,
                    fromTip2=childFromTip,
                    returnLK=True,
                    isUpDown=True,
                )
                if ins is None:
                    continue
                insertionBaseVect, lk_ins = ins
                if insertionBaseVect is None:
                    continue

                for leafLen in leaf_cands:
                    merged = mergeVectors(
                        probVect1=insertionBaseVect,
                        bLen1=0.0,
                        fromTip1=False,
                        probVect2=newSampleProbVect,
                        bLen2=leafLen,
                        fromTip2=True,
                        returnLK=True,
                        isUpDown=False,
                    )
                    if merged is None:
                        continue
                    _, lk_leaf = merged
                    lk_total = lk_ins + lk_leaf

                    score = lk_total
                    score -= prior_strength * (topLen / prior_scale)
                    score -= prior_strength * (botLen / prior_scale)
                    score -= prior_strength * (leafLen / prior_scale)
                    score -= sum_dev_strength * (dev_edge / max(edgeLen0, _min_blen()))

                    if score > bestLocalScore:
                        bestLocalScore = score
                        bestLocalLK = lk_total
                        bestLocalTop = topLen
                        bestLocalBot = botLen
                        bestLocalLeaf = leafLen

        if bestLocalLeaf is None:
            continue

        if bestLocalScore > bestScoreGlobal:
            bestScoreGlobal = bestLocalScore
            bestLikelihood = bestLocalLK
            bestParent = parentIdx
            bestChild = childIdx
            bestLenP = bestLocalTop
            bestLenC = bestLocalBot
            bestNewLen = bestLocalLeaf

    return bestParent, bestChild, bestLenP, bestLenC, bestNewLen, bestLikelihood

def placeSampleOnTree(tree, newSampleName, newSampleDiffs, parentNode, childNode, lenParentToInternal, lenInternalToChild, newLeafLen, ref_seq):
    """Edge-based insertion by splitting the edge (parentNode -> childNode).

    This keeps the tree binary by creating a new internal node between the
    existing parent/child and attaching the new sample leaf to that internal.

    Returns:
        newLeafIdx
    """
    # Create new leaf node for the sample
    tree.addNode(dirtiness=True)
    newLeafIdx = len(tree.up) - 1
    tree.name[newLeafIdx] = newSampleName
    tree.dist[newLeafIdx] = newLeafLen
    tree.probVect[newLeafIdx] = probVectTerminalNode(
        diffs=newSampleDiffs,
        tree=tree,
        node=newLeafIdx,
        ref_seq=ref_seq,
    )

    # Create internal node that splits the existing edge
    tree.addNode(dirtiness=True)
    internalIdx = len(tree.up) - 1
    tree.name[internalIdx] = ""
    tree.up[internalIdx] = parentNode
    tree.dist[internalIdx] = lenParentToInternal
    tree.children[internalIdx] = [childNode, newLeafIdx]

    # Rewire child and new leaf under internal
    tree.up[childNode] = internalIdx
    tree.dist[childNode] = lenInternalToChild
    tree.up[newLeafIdx] = internalIdx

    # Replace childNode with internalIdx in parentNode's children list
    try:
        pos = tree.children[parentNode].index(childNode)
        tree.children[parentNode][pos] = internalIdx
    except ValueError:
        # If parent-child relationship is unexpected, fall back to appending.
        tree.children[parentNode].append(internalIdx)

    # Mark local neighborhood as dirty so partials are recomputed
    tree.dirty[parentNode] = True
    tree.dirty[internalIdx] = True
    tree.dirty[childNode] = True
    tree.dirty[newLeafIdx] = True

    # Incremental partial-likelihood update (MAPLE-style)
    nodeList = [
        (newLeafIdx, 2, True, False),
        (internalIdx, 2, True, False),
        (childNode,   2, True, False),
        (parentNode,  2, True, False),
    ]

    updatePartials(tree, nodeList=nodeList)

    return newLeafIdx

def updateProbVectAtNode(tree, nodeIdx, childIdx, mutMatrix):
    """
    Update the probability vector at a given node after branch length changes.
    
    Inputs:
        tree: Tree object
        nodeIdx: Index of node to update
        childIdx: Index of child that triggered update
        mutMatrix: Substitution matrix
    Returns:
        None
    """
    # If node has children, merge their vectors
    if len(tree.children[nodeIdx]) == 2:
        child1 = tree.children[nodeIdx][0]
        child2 = tree.children[nodeIdx][1]
        
        # Get partial vectors from children
        vect1 = getPartialVec(
            i12=6,  # Code for probability vector
            totLen=tree.dist[child1],
            mutMatrix=mutMatrix,
            errorRate=0.0,
            vect=tree.probVect[child1],
            upNode=False,
            flag=False
        )
        
        vect2 = getPartialVec(
            i12=6,
            totLen=tree.dist[child2],
            mutMatrix=mutMatrix,
            errorRate=0.0,
            vect=tree.probVect[child2],
            upNode=False,
            flag=False
        )
        
        # Merge at this node
        mergedVect, _ = mergeVectors(
            probVect1=vect1,
            bLen1=tree.dist[child1],
            fromTip1=(len(tree.children[child1]) == 0),
            probVect2=vect2,
            bLen2=tree.dist[child2],
            fromTip2=(len(tree.children[child2]) == 0),
            returnLK=False,
            isUpDown=False
        )
        
        tree.probVect[nodeIdx] = mergedVect
    
    # Mark as clean
    tree.dirty[nodeIdx] = False

def optimizeBranchLengths(tree, mutMatrix, maxIterations=5):
    """
    Optimize all branch lengths in the tree using likelihood derivatives.
    
    Inputs:
        tree: Tree object
        mutMatrix: Substitution matrix
        maxIterations: Max number of optimization passes
    
    Returns:
        converged: True if optimization converged
    """
    print(f"\nOptimizing branch lengths...")
    
    def _min_blen():
        try:
            return float(minBLenSensitivity) if minBLenSensitivity else 1e-8
        except Exception:
            return 1e-8

    for iteration in range(maxIterations):
        print(f"  Iteration {iteration + 1}/{maxIterations}")
        
        # Track if any branch length changed significantly
        anyChange = False
        nodeList = []
        
        # Iterate through all nodes (except root which has no parent branch)
        for nodeIdx in range(len(tree.up)):
            if tree.up[nodeIdx] is None:  # Root node
                continue
            
            # Only optimize if node is marked dirty or first iteration
            if not tree.dirty[nodeIdx] and iteration > 0:
                continue
            
            oldBranchLen = float(tree.dist[nodeIdx]) if tree.dist[nodeIdx] not in (None, False) else 0.0

            # updateBLen() uses the correct up-excluding-child vector; it also sanitizes
            # False/None outputs from derivative estimation.
            updateBLen(tree=tree, cNode=nodeIdx, addToList=True, nodeList=nodeList)
            newBranchLen = float(tree.dist[nodeIdx]) if tree.dist[nodeIdx] not in (None, False) else 0.0

            if abs(newBranchLen - oldBranchLen) > 0.0001 and newBranchLen >= _min_blen():
                anyChange = True
                print(f"    Node {nodeIdx}: {oldBranchLen:.6f} -> {newBranchLen:.6f}")
        
        # Update partial likelihood vectors affected by branch-length changes
        if nodeList:
            updatePartials(tree, nodeList=nodeList)
        
        # Check convergence
        if not anyChange:
            print(f"  Converged after {iteration + 1} iterations")
            return True
    
    print(f"  Did not fully converge after {maxIterations} iterations")
    return False
 
def main():
    """
    Simplified MAPLE main pipeline:
    1. Read reference and input alignment
    2. Initialize substitution model (JC69)
    3. Build initial tree by sequential sample placement
    4. Output final tree in Newick format
    """
    # Initialize inputs
    global lref, oneMutBLen, minBLenSensitivity, globalTotRate, refIndeces, cumulativeRate
    diffs = [('t', 313), ('g', 1832), ('c', 10029), ('c', 21618), ('t', 22917), ('c', 22995), ('a', 23063),
             ('c', 23604), ('a', 28271), ('g', 29742)] # from the sample from maple test
    node = 0
    tree = None
    
    # ============================================================================
    # 1. READ INPUT
    # ============================================================================
    print("Step 1: Reading input files...")
    
    # Read reference genome
    # refFile = "./maple_alignment_sample/aligned_europe.fasta"
    # inputFile = "./maple_alignment_sample/maple_europe.txt"
    inputFile = "maple_alignment_sample/maple_aligned_oceania.txt"
    # ref = collectReference(refFile)
    # lref = len(ref)
    # print(f"Reference genome length: {len(ref)}")
    
    # Read alignment in MAPLE diff format w/ extractReference=True for now
    ref, data = readConciseAlignment(inputFile, ref=None, extractReference=True, onlyRef=None)
    lref = len(ref)
    print(f"Reference extracted; length: {lref}")
    if not isinstance(data, dict):
        raise Exception("Alignment data should be a dictionary of sample_name: diffs")
    sampleNames = list(data.keys())
    numSamples = len(sampleNames)
    print(f"Number of samples: {numSamples}")
    
    # Reset globals in case main() is run multiple times in one session.
    refIndeces.clear()
    cumulativeRate.clear()
    cumulativeRate.append(0.0)

    oneMutBLen = 1.0 / lref

    # Fraction of a mutation to be considered as a precision for branch length estimation (default 0.001, which means branch lengths estimated up to a 1000th of a mutation precision).
    minBLenSensitivity= 0.001 * oneMutBLen
    
    # fill in refIndeces
    for i in range(lref):
        refNuc=ref[i].upper()
        refIndeces.append(allelesDict[refNuc]) if refNuc in allelesDict else refIndeces.append(0)  # default to A if unknown
    
    
    
    # ============================================================================
    # 2. INITIALIZE SUBSTITUTION MODEL
    # ============================================================================
    print("\nStep 2: Initializing JC69 substitution model...")
    
    # Initialize mutation matrix (4x4 for A, C, G, T) as a *rate* matrix Q.
    mutMatrix = [[0.0 for _ in range(4)] for _ in range(4)]
    updateSubMatrix("JC", mutMatrix)
    
    mutMatrix = np.array(mutMatrix)
    # Many core routines default to the global matrix.
    global mutMatrixGlobal
    mutMatrixGlobal = mutMatrix

    # Cumulative rates and globalTotRate must be consistent with Q.
    # MAPLE uses cumulativeRate[pos] based on the (reference) diagonal rates.
    nonMutRates = [-float(mutMatrixGlobal[i][i]) for i in range(4)]
    for i in range(lref):
        ind = refIndeces[i]
        cumulativeRate.append(cumulativeRate[-1] + nonMutRates[ind])
    # globalTotRate is expected to be negative in the derivative code.
    globalTotRate = -float(cumulativeRate[-1])

    print("Substitution matrix initialized:")
    print(mutMatrix)
    
    # ============================================================================
    # 3. BUILD INITIAL TREE
    # ============================================================================
    print("\nStep 3: Building initial tree...")
    import time 
    start = time.time()
    
    # Create tree structure
    tree = Tree()
    
    def _min_blen():
        try:
            return float(minBLenSensitivity) if minBLenSensitivity else 1e-8
        except Exception:
            return 1e-8

    # Create an unlabeled internal root, and attach the first sample as a leaf.
    tree.addNode()
    rootNode = 0
    tree.name[rootNode] = ""
    tree.dist[rootNode] = 0.0

    firstSample = sampleNames[0]
    tree.addNode()
    firstLeaf = 1
    tree.name[firstLeaf] = firstSample
    tree.dist[firstLeaf] = _min_blen()
    tree.up[firstLeaf] = rootNode
    tree.children[rootNode] = [firstLeaf]

    # Create probability vector for the first sample leaf
    tree.probVect[firstLeaf] = probVectTerminalNode(
        diffs=data[firstSample],
        tree=tree,
        node=firstLeaf,
        ref_seq=ref
    )

    # Initialize down+up partials so later branch-length logic has valid up vectors.
    updatePartials(tree, force=True)
    
    print(f"Root node created with sample: {firstSample}")
    
    # import random
    # MAX_SAMPLES = 100
    # SUBSEED = 1

    # root = sampleNames[0]
    # rest = sampleNames[1:]
    # random.Random(SUBSEED).shuffle(rest)

    # if len(sampleNames) > MAX_SAMPLES:
    #     sampleNames = [root] + rest[:MAX_SAMPLES - 1]
    # numSamples = len(sampleNames)

    # Place remaining samples one by one
    for i, sampleName in enumerate(sampleNames[1:], start=1):
        print(f"\nPlacing sample {i}/{numSamples-1}: {sampleName}")
        
        # Find best edge-splitting placement
        bestParent, bestChild, bestLenP, bestLenC, bestNewLen, bestLK = findBestParentForNewSample(
            tree=tree,
            newSampleDiffs=data[sampleName],
            ref_seq=ref,
            mutMatrix=mutMatrix
        )
        
        print(f"  Best edge: parent={bestParent}, child={bestChild}")
        print(f"  Split lengths: parent->internal={bestLenP:.6f}, internal->child={bestLenC:.6f}")
        print(f"  New leaf length: {bestNewLen:.6f}")
        print(f"  Log-likelihood: {bestLK:.2f}")
        
        # Place sample on tree
        newNodeIdx = placeSampleOnTree(
            tree=tree,
            newSampleName=sampleName,
            newSampleDiffs=data[sampleName],
            parentNode=bestParent,
            childNode=bestChild,
            lenParentToInternal=bestLenP,
            lenInternalToChild=bestLenC,
            newLeafLen=bestNewLen,
            ref_seq=ref
        )

        
        
        # Optimize branch lengths every 200 samples (or adjust frequency)
        # 10->35
        if (i % 200 == 0):
            print(f"\n  Optimizing branch lengths after {i} placements...")
            optimizeBranchLengths(tree, mutMatrix, maxIterations=3)
            
    # ============================================================================
    # 4. FINAL OPTIMIZATION
    # ============================================================================
    print("\nStep 4: Final branch length optimization...")
    optimizeBranchLengths(tree, mutMatrix, maxIterations=10)
    
    numSamples = len(data)
    placedSamples = sum(1 for n in tree.name if n)
    print(f"Tree built with {placedSamples}/{numSamples} samples")
    
    # ============================================================================
    # 5. OUTPUT TREE
    # ============================================================================
    print("\nStep 4: Writing output tree...")

    # Quick sanity check: are we actually carrying nonzero branch lengths?
    try:
        dists = [0.0 if (d is None or d is False) else float(d) for d in tree.dist]
        n_zero = sum(1 for d in dists if d == 0.0)
        pos = [d for d in dists if d > 0.0]
        print(
            "Branch length stats: "
            f"n={len(dists)} zeros={n_zero} "
            f"min_pos={min(pos) if pos else None} max={max(dists) if dists else None}"
        )
    except Exception as e:
        print(f"Branch length stats: unavailable ({e})")
    
    outputFile = "output_oceania_biohpc_tree.newick"
    
    # Generate Newick string
    newickString = createNewick(
        tree=tree,
        root_node_id=rootNode,
        namesInTree=[firstSample],
        includeMutationList=False,
        includeSupports=False
    )
    
    # Write to file
    with open(outputFile, "w") as f:
        f.write(newickString)
    
    print(f"Tree written to: {outputFile}")
    print(f"Newick string: {newickString}")
    end = time.time()
    
    # ============================================================================
    # 6. SUMMARY STATISTICS
    # ============================================================================
    print("\n" + "="*60)
    print("MAPLE Pipeline Complete!")
    print("="*60)
    print(f"Reference length: {len(ref)} bp")
    print(f"Total samples: {numSamples}")
    print(f"Samples placed: {placedSamples}")
    print(f"Output: {outputFile}")
    print("="*60)
    print(f"Total time: {end - start:.2f} seconds")


if __name__ == "__main__":
    mem = memory_usage((main, (), {}), interval=0.1)

    # Save to PDF
    plt.figure(figsize=(10, 6))
    plt.plot(mem, linewidth=2)
    plt.ylabel('Memory (MB)')
    plt.xlabel('Time (0.1s intervals)')
    plt.title('Memory Usage')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('memory_usage.pdf')  # Saves as PDF
    print(f"Peak memory: {max(mem):.2f} MB")
    
    main()