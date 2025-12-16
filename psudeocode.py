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
import sympy
import sys

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

oneMutBLen=1.0/lref

# Fraction of a mutation to be considered as a precision for branch length estimation (default 0.001, which means branch lengths estimated up to a 1000th of a mutation precision).
minBLenSensitivity= 0.001 * oneMutBLen

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
globalTotRate=float('inf') # remark: orignial -float(lref)

# beginning of merge vectors code
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

        ctx["state"] = sympy.simplify(ctx["newVec"], ctx["refNucToPass"])
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


def mergeVectors(probVect1, bLen1, fromTip1, probVect2, bLen2, fromTip2,
                 returnLK=False, isUpDown=False, numMinor1=0, numMinor2=0,
                 errorRateGlobalPassed=None, mutMatrixGlobalPassed=None,
                 errorRatesGlobal=None, mutMatricesGlobal=None,
                 cumulativeRateGlobal=None, cumulativeErrorRateGlobal=None):
    
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
    
    # remark: fix innitzas
    if useRateVariation:
        ctx["mutMatricesUsed"] = ctx["mutMatricesGlobal"] if ctx["mutMatricesGlobal"] is not None else mutMatricesGlobal
    else:
        ctx["mutMatrix"] = ctx["mutMatrixGlobalPassed"] if ctx["mutMatrixGlobalPassed"] is not None else mutMatrixGlobalPassed

    if usingErrorRate and errorRateSiteSpecific:
        ctx["errorRatesUsed"] = ctx["errorRatesGlobal"] if ctx["errorRatesGlobal"] is not None else errorRatesGlobal
    else:
        ctx["errorRate"] = ctx["errorRateGlobalPassed"] if ctx["errorRateGlobalPassed"] is not None else errorRateGlobalPassed

    if returnLK:
        ctx["cumulativeRateUsed"] = ctx["cumulativeRateGlobal"] if ctx["cumulativeRateGlobal"] is not None else cumulativeRateGlobal
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

        if ctx["entry1"][0] < 4 or ctx["entry1"][0] == 6:
            ctx["indexEntry1"] += 1
            ctx["entry1"] = probVect1[ctx["indexEntry1"]]
        elif ctx["pos"] == ctx["entry1"][1]:
            ctx["indexEntry1"] += 1
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


# def estimateBranchLengthWithDerivative(probVectP, probVectC, fromTipP, fromTipC, mutMatrix, minBL, maxBL, precision, errorRate, pseudoCountsGlobal, mutMatricesGlobal, cumulativeRateGlobal):

    
# NOTE: this is a helper function from MAPLE_original.py
# Flags have been fixed to False, None or [0.0]
def estimateBranchLengthWithDerivative(probVectP,probVectC,fromTipC=False,errorRateGlobalPassed=None,mutMatrixGlobalPassed=None,errorRatesGlobal=None,mutMatricesGlobal=None,cumulativeRateGlobal=None):
    """
    Estimate branch length maximizing likelihood; optionally use derivatives.

    Inputs: ['probVectP', 'probVectC', 'fromTipP', 'fromTipC', 'mutMatrix', 'minBL', 'maxBL', 'precision', 'errorRate', 'pseudoCountsGlobal', 'mutMatricesGlobal', 'cumulativeRateGlobal']
    Outputs: see return docs in MAPLE source.
    """
    mutMatricesUsed = None  # because rateLimiter is false, this is not used
    if mutMatrixGlobalPassed!=None:
        mutMatrix=mutMatrixGlobalPassed
    else:
        mutMatrix=mutMatrixGlobal

    # Note: usingErrorRate and errorRateSiteSpecific always set to False
    if usingErrorRate and errorRateSiteSpecific:
        if errorRatesGlobal!=None:
            errorRatesUsed=errorRatesGlobal
        else:
            errorRatesUsed=errorRates
    else:
        if errorRateGlobalPassed==None:
             errorRate=errorRateGlobal
        else:
            errorRate=errorRateGlobalPassed
    
    if cumulativeRateGlobal!=None:
        cumulativeRateUsed=cumulativeRateGlobal
    else:
        cumulativeRateUsed=cumulativeRate
        
    if not (usingErrorRate and errorRateSiteSpecific):
        errorRate=errorRateGlobal
    if not useRateVariation:
        mutMatrix=mutMatrixGlobal
  
    c1=globalTotRate
    ais=[]
    indexEntry1, indexEntry2, pos, contribLength, nZeros = 0, 0, 0, 0.0, 0
    entry1=probVectP[indexEntry1]
    entry2=probVectC[indexEntry2]
    while True:
        if entry2[0]==5:
            if entry1[0]==4 or entry1[0]==5:
                end=min(entry1[1],entry2[1])
            else:
                end=pos+1
            c1+=(cumulativeRateUsed[pos]-cumulativeRateUsed[end])
            pos=end
        elif entry1[0]==5: # case entry1 or entry2 is N
			#if parent node is type "N", in theory we might have to calculate the contribution of root nucleotides; 
			# however, if this node is "N" then every other node in the current tree is "N", so we can ignore this since this contribution cancels out in relative terms.
            if entry2[0]==4:
                end=min(entry1[1],entry2[1])
            else:
                end=pos+1
            c1+=(cumulativeRateUsed[pos]-cumulativeRateUsed[end])
            pos=end
        else:
			#below, when necessary, we represent the likelihood as coeff0*l +coeff1, where l is the branch length to be optimized.
            if entry1[0]==4 and entry2[0]==4: # case entry1 and entry2 are R
                pos=min(entry1[1],entry2[1])
            else:
                if useRateVariation and mutMatricesUsed is not None:
                    mutMatrix=mutMatricesUsed[pos]
                
                if entry1[0]==4:
                    c1-=mutMatrix[entry2[1]][entry2[1]]
                else:
                    c1-=mutMatrix[entry1[1]][entry1[1]]
                flag1 = (usingErrorRate and (entry1[0]!=6) and len(entry1)>2 and entry1[-1]) # flag1 true if error rate applies to entry1
                flag2 = (usingErrorRate and (entry2[0]!=6) and (fromTipC or (len(entry2)>2 and entry2[-1])))
                if usingErrorRate and errorRateSiteSpecific and errorRatesUsed is not None: errorRate = errorRatesUsed[pos]

				#contribLength will be here the total length from the root or from the upper node, down to the down node.
                contribLength=False
                if entry1[0] < 5:
                    if len(entry1)==3+usingErrorRate:
                        contribLength=entry1[2]
                    elif len(entry1)==4+usingErrorRate:
                        contribLength=entry1[3]
                else:
                    if len(entry1)>3:
                        contribLength=entry1[2]
                if entry2[0]<5:
                    if len(entry2)>2+usingErrorRate:
                        contribLength+=entry2[2]
                else:
                    if len(entry2)>3:
                        contribLength+=entry2[2]

                if entry1[0] == 4:
					# entry1 is reference and entry2 is of type "O"
                    if entry2[0] == 6:
                        i1=entry2[1]
                        if len(entry1)==(4+usingErrorRate):
                            coeff0=rootFreqs[i1]*entry2[-1][i1] 
                            coeff1=0.0
                            for i in range(4):
                                coeff0+=rootFreqs[i]*mutMatrix[i][i1]*entry1[2]*entry2[-1][i]
                                coeff1+=mutMatrix[i1][i]*entry2[-1][i]
                            coeff1*=rootFreqs[i1]
                            if contribLength:
                                coeff0+=coeff1*contribLength
                            if flag1 and errorRate is not None:
                                coeff0-=1.33333*errorRate*rootFreqs[i1]*entry2[-1][i1]
                                for i in range(4):
                                    coeff0+=rootFreqs[i]*entry2[-1][i]*0.33333*errorRate
                        else:
                            coeff0=entry2[-1][i1]
                            coeff1=0.0
                            for j in range(4):
                                coeff1+=mutMatrix[i1][j]*entry2[-1][j]
                            if contribLength:
                                coeff0+=coeff1*contribLength
                        if coeff1<0.0:
                            c1+=coeff1/coeff0
                        elif coeff1:
                            coeff0=coeff0/coeff1
                            ais.append(coeff0)
                        pos+=1

                    else: #entry1 is R and entry2 is a different but single nucleotide
                        if len(entry1)==4+usingErrorRate:
                            i1=entry2[1]
                            i2=entry2[0]
                            coeff0=rootFreqs[i2]*mutMatrix[i2][i1]*entry1[2]
                            if contribLength:
                                coeff0+=rootFreqs[i1]*mutMatrix[i1][i2]*contribLength
                            if flag2:
                                coeff0+=rootFreqs[i1]*0.33333*errorRate
                            if flag1:
                                coeff0+=rootFreqs[i2]*0.33333*errorRate
                            coeff1=rootFreqs[i1]*mutMatrix[i1][i2]
                            if coeff1:
                                coeff0=coeff0/coeff1
                            else:
                                coeff0=None
                        else:
                            coeff0=contribLength
                            if flag2 and errorRate is not None:
                                if mutMatrix[entry2[1]][entry2[0]]:
                                    coeff0+=errorRate*0.33333/mutMatrix[entry2[1]][entry2[0]]
                                else:
                                    coeff0=None
                        if coeff0!=None:	
                            if coeff0:
                                ais.append(coeff0)
                            else:
                                nZeros+=1
                        pos+=1

				# entry1 is of type "O"
                elif entry1[0]==6:
                    if entry2[0]==6:
                        coeff0=entry1[-1][0]*entry2[-1][0]+entry1[-1][1]*entry2[-1][1]+entry1[-1][2]*entry2[-1][2]+entry1[-1][3]*entry2[-1][3]
                        coeff1=0.0
                        for i in range(4):
                            for j in range(4):
                                coeff1+=entry1[-1][i]*entry2[-1][j]*mutMatrix[i][j]
                        if contribLength:
                            coeff0+=coeff1*contribLength
                    else: #entry1 is "O" and entry2 is a nucleotide
                        if entry2[0]==4:
                            i2=entry1[1]
                        else:
                            i2=entry2[0]
                        coeff0=entry1[-1][i2]
                        coeff1=0.0
                        for i in range(4):
                            coeff1+=entry1[-1][i]*mutMatrix[i][i2]
                        if contribLength:
                            coeff0+=coeff1*contribLength
                        if flag2 and errorRate is not None:
                            coeff0+=errorRate*0.33333
                    if coeff1<0.0:
                        c1+=coeff1/coeff0
                    elif coeff1:
                        coeff0=coeff0/coeff1
                        ais.append(coeff0)
                    pos+=1
                else: #entry1 is a non-ref nuc
                    if entry2[0]==entry1[0]:
                        c1+=mutMatrix[entry1[0]][entry1[0]]
                    else: #entry1 is a nucleotide and entry2 is not the same as entry1
                        i1=entry1[0]
                        if entry2[0]<5: #entry2 is a nucleotide
                            if entry2[0]==4:
                                i2=entry1[1]
                            else:
                                i2=entry2[0]

                            if len(entry1)==4+usingErrorRate:
                                coeff0=rootFreqs[i2]*mutMatrix[i2][i1]*entry1[2]
                                if contribLength:
                                    coeff0+=rootFreqs[i1]*mutMatrix[i1][i2]*contribLength
                                if flag2:
                                    coeff0+=rootFreqs[i1]*0.33333*errorRate
                                if flag1:
                                    coeff0+=rootFreqs[i2]*0.33333*errorRate
                                coeff1=rootFreqs[i1]*mutMatrix[i1][i2]
                                if coeff1:
                                    coeff0=coeff0/coeff1
                                else:
                                    coeff0=None
                            else:
                                coeff0=contribLength
                                if flag2 and errorRate is not None:
                                    coeff0+=errorRate*0.33333/mutMatrix[i1][i2]
                            if coeff0!=None:
                                if coeff0:
                                    ais.append(coeff0)
                                else:
                                    nZeros+=1

                        else: #entry1 is a nucleotide and entry2 is of type "O"
                            if len(entry1)==4+usingErrorRate:
                                coeff0=rootFreqs[i1]*entry2[-1][i1] 
                                coeff1=0.0
                                for i in range(4):
                                    coeff0+=rootFreqs[i]*mutMatrix[i][i1]*entry1[2]*entry2[-1][i]
                                    coeff1+=mutMatrix[i1][i]*entry2[-1][i]
                                coeff1*=rootFreqs[i1]
                                if contribLength:
                                    coeff0+=coeff1*contribLength
                                if flag1 and errorRate is not None:
                                    coeff0-=1.33333*errorRate*rootFreqs[i1]*entry2[-1][i1]
                                    for i in range(4):
                                        coeff0+=rootFreqs[i]*entry2[-1][i]*0.33333*errorRate
                            else:
                                coeff0=entry2[-1][i1]
                                coeff1=0.0
                                for j in range(4):
                                    coeff1+=mutMatrix[i1][j]*entry2[-1][j]
                                if contribLength:
                                    coeff0+=coeff1*contribLength
                            if coeff1<0.0:
                                c1+=coeff1/coeff0
                            elif coeff1:
                                coeff0=coeff0/coeff1
                                ais.append(coeff0)	
                    pos+=1

        if pos== lref:
            break
        if entry1[0]<4 or entry1[0]==6:
            indexEntry1+=1
            entry1=probVectP[indexEntry1]
        elif pos==entry1[1]:
            indexEntry1+=1
            entry1=probVectP[indexEntry1]
        if entry2[0]<4 or entry2[0]==6:
            indexEntry2+=1
            entry2=probVectC[indexEntry2]
        elif pos==entry2[1]:
            indexEntry2+=1
            entry2=probVectC[indexEntry2]

	#now optimized branch length based on coefficients
    c1=-c1
    n=len(ais)+nZeros
    if n==0:
        return False
    else:
        if len(ais):
            minAis=min(ais)
        else:
            minAis=0.0
        if nZeros:
            minAis=min(0.0,minAis)
        if minAis<0.0:
            return 0.1
        tDown=min(0.1,n/c1-minAis)
        if tDown<=0.0:
            return False
        if nZeros:
            vDown=nZeros/tDown
        else:
            vDown=0.0
        for ai in ais:
            vDown+=1.0/(ai+tDown)
        if len(ais):
            maxAis=max(ais)
        else:
            maxAis=0.0
        tUp=min(0.1,n/c1-maxAis)
        if tUp>=0.1:
            return 0.1
        if tUp<=minBLenSensitivity:
            if minAis:
                tUp=0.0
            else:
                tUp=minBLenSensitivity
        if nZeros:
            vUp=nZeros/tUp
        else:
            vUp=0.0
        for ai in ais:
            vUp+=1.0/(ai+tUp)
    if vDown>c1+minBLenSensitivity or vUp<c1-minBLenSensitivity:
        if vUp<c1-minBLenSensitivity and (not tUp):
            return False
        if (vDown>c1+minBLenSensitivity) and tDown>=0.1:
            return 0.1
        print("Initial border parameters don't fit expectations")

    while tDown-tUp>minBLenSensitivity:
        tMiddle=(tUp+tDown)/2
        if nZeros:
            vMiddle=nZeros/tMiddle
        else:
            vMiddle=0.0
        for ai in ais:
            vMiddle+=1.0/(ai+tMiddle)
        if vMiddle>c1:
            tUp=tMiddle
        else:
            tDown=tMiddle

    return tUp


def updateBLen(tree, cNode, addToList, nodeList):
    """ Commit a branch-length change and update impacted node lists/vectors.

Inputs: ['tree', 'cNode', 'addToList', 'nodeList']
Outputs: see return docs in MAPLE source.
"""
    # store local variables
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

    if cNode == children[parent][0] : #node is left child
         cIdx = 0
         vectUp = probUpRight[parent]
    else : #node is right child
         cIdx = 1
         vectUp = probUpLeft[parent]
    
    vectDown = probDown[cNode]

    if vectUp is None or vectDown is None:
        return
    
    bestLength = estimateBranchLengthWithDerivative(vectUp, vectDown, fromTipC = (len(children[cNode]) == 0))
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

def compare_ACGTR_entry(entry1 ,entry2) :
    if len(entry1) > 2 and len(entry2) > 2: 
        if (abs(entry1[2] - entry2[2]) > THRESHOLD) : #compare branch lengths
            return True
    return False

def compare_O_entry(entry1, entry2) :
    if len(entry1) > 2 and len(entry2) > 2:
        if abs(entry1[2] - entry2[2]) > THRESHOLD : #compare branch lengths
            return True
    for i in range(4) : #compare probabilities of each nucleotide
        diffVal = abs(entry1[-1][i] - entry2[-1][i])
        if (diffVal > thresholdDiffForUpdate) :
            return True
        if (diffVal>THRESHOLD and ((diffVal/entry1[-1][i]>thresholdFoldChangeUpdate)  or  (diffVal/entry2[-1][i]>thresholdFoldChangeUpdate))):
            return True
    return False

def update_singular_pos(pos) :
    return pos + 1

def update_contiguous_pos(entry1, entry2) :
    return min(entry1[1], entry2[1])


        

def areVectorsDifferent(probVect1, probVect2):
    """ Return True if two probability vectors differ beyond thresholds.

Inputs: ['probVect1', 'probVect2']
Outputs: see return docs in MAPLE source.
"""
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
            if (compare_ACGTR_entry(entry1, entry2)):
                return True
            if entry1[0] <= 3: #ACGT
                pos = update_singular_pos(pos)
            else : #type R
                pos = update_contiguous_pos(entry1, entry2)

        elif entry1[0] == 5: #type N
             pos = update_contiguous_pos(entry1, entry2)

        elif entry1[0] == 6: # type 6
            if (compare_O_entry(entry1, entry2)) :
                return True
            pos = update_contiguous_pos(entry1, entry2)

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
            # Hannah's NOTE: label does not use tree in the args, so I removed it for now
            if is_leaf:
                label_str = _label(nextNode, includeSupports, minSupport, includeMutationList, performLineageAssignmentByRefPlacement)
                stringList.append(label_str)

            # If internal, append ')' followed by the label (if any)
            if not is_leaf:
                stringList.append(")")
                label_str = _label(nextNode, includeSupports, minSupport, includeMutationList, performLineageAssignmentByRefPlacement)
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
    
 
def main():
    """
    Simplified MAPLE main pipeline:
    1. Read reference and input alignment
    2. Initialize substitution model (JC69)
    3. Build initial tree by sequential sample placement
    4. Output final tree in Newick format
    """
    # Initialize inputs
    diffs = [('t', 313), ('g', 1832), ('c', 10029), ('c', 21618), ('t', 22917), ('c', 22995), ('a', 23063),
             ('c', 23604), ('a', 28271), ('g', 29742)] # from the sample from maple test
    node = 0
    tree = None
    
    # Cumulative Rates 
    nonMutRates=[0,0,0,0]
    
    # ============================================================================
    # 1. READ INPUT
    # ============================================================================
    print("Step 1: Reading input files...")
    
    # Read reference genome
    refFile = "reference.fasta"  # TODO: find name
    inputFile = "alignment.txt" # TODO: find name
    
    ref = collectReference(refFile)
    print(f"Reference genome length: {len(ref)}")
    
    # fill in refIndeces    
    for i in range(lref):
        refNuc=ref[i]
        refIndeces.append(allelesDict[refNuc]) if refNuc in allelesDict else refIndeces.append(0)  # default to A if unknown
      
    # fill in cumulative rates  
    for i in range(lref):
        ind=refIndeces[i]
        cumulativeRate.append(cumulativeRate[-1]+nonMutRates[ind])
    
    
    # Read alignment in MAPLE diff format WITHOUT EXTRACT REFERENCE for now
    data = readConciseAlignment(inputFile, extractReference=False, ref=ref, onlyRef=False)
    if not isinstance(data, dict):
        raise Exception("Alignment data should be a dictionary of sample_name: diffs")
    print(f"Number of samples in alignment: {len(data)}")
    
    # ============================================================================
    # 2. INITIALIZE SUBSTITUTION MODEL
    # ============================================================================
    print("\nStep 2: Initializing JC69 substitution model...")
    
    # Initialize mutation matrix (4x4 for A, C, G, T)
    mutMatrix = [[0.25, 0.25, 0.25, 0.25] for _ in range(4)]
    
    # Update to normalized JC69 model
    updateSubMatrix("JC", mutMatrix)
    print("Substitution matrix initialized:")
    print(mutMatrix)
    
    # ============================================================================
    # 3. BUILD INITIAL TREE
    # ============================================================================
    print("\nStep 3: Building initial tree...")
    
    # Create tree structure
    tree = Tree()
    
    # Add root node with first sample
    tree.addNode()
    rootNode = 0
    
    # must check data is a dict
    firstSample = list(data.keys())[0]
    tree.name[rootNode] = firstSample
    
    # Create probability vector for root sample 
    tree.probVect[rootNode] = probVectTerminalNode(
        diffs=data[firstSample],
        tree=tree,
        node=rootNode,
        ref_seq=ref
    )
    
    print(f"Root node created with sample: {firstSample}")

    # TODO: Idea:
    # - Loop through remaining samples
    # - For each sample, find best placement using findBestParentForNewSample()
        # findBestParentForNewSample(tree, newSampleDiffs, ref_seq, mutMatrix) where are these arguments from??
    # - Place sample on tree using placeSampleOnTree()
        #copy and paste placeSampleOnTree() from codebase?
    
    #***below TEMPORARILY copied and pasted from original code
    # bestNode , bestScore, bestBranchLengths, bestPassedVect = findBestParentForNewSample(tree,t1,newPartials,numSamples,computePlacementSupportOnly=False, diffsTime=newPartialsTime)
    # if bestBranchLengths!=None:
    #     start=time()
    #     newRoot=placeSampleOnTree(tree,bestNode,bestPassedVect,numSamples,bestScore, bestBranchLengths[0], bestBranchLengths[1], bestBranchLengths[2],pseudoMutCounts,newPartialsTime=newPartialsTime)
    #     if newRoot!=None:
    #         t1=newRoot

    # - Optimize branch lengths using estimateBranchLengthWithDerivative()
    # - Update genome lists using updatePartials()
    
    numSamples = len(data)
    print(f"Tree initialized with 1/{numSamples} samples")
    print("(Sequential placement of remaining samples not yet implemented)")
    
    # ============================================================================
    # 4. OUTPUT TREE
    # ============================================================================
    print("\nStep 4: Writing output tree...")
    
    outputFile = "output_tree.newick"
    
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
    
    # ============================================================================
    # 5. SUMMARY STATISTICS
    # ============================================================================
    print("\n" + "="*60)
    print("MAPLE Pipeline Complete!")
    print("="*60)
    print(f"Reference length: {len(ref)} bp")
    print(f"Total samples: {numSamples}")
    print(f"Samples placed: 1 (root only)")
    print(f"Output: {outputFile}")
    print("="*60)


if __name__ == "__main__":
    main()
   