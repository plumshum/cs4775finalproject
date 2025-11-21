# def createNewick(tree, node, binary, namesInTree, includeMutationList, estimateMAT, networkOutput, sprtaOn, minSupport, count0BLenNodesOnce, includeSupports, keepInputIQtreeSupports, aBayesPlusOn, performLineageAssignmentByRefPlacement):
#     """ Serialize tree to Newick/Nexus; optionally include supports, MAT, and metadata.

# Inputs: ['tree', 'node', 'binary', 'namesInTree', 'includeMutationList', 'estimateMAT', 'networkOutput', 'sprtaOn', 'minSupport', 'count0BLenNodesOnce', 'includeSupports', 'keepInputIQtreeSupports', 'aBayesPlusOn', 'performLineageAssignmentByRefPlacement']
# Outputs: see return docs in MAPLE source.
# """
#     pass



def createNewick(tree, node, binary=False, namesInTree=None):
    """Serialize tree to Newick format (simplified version).
    
    Inputs: ['tree', 'node', 'binary', 'namesInTree']
    Returns: Newick string
    """
    pass
    if node.is_leaf():
        name = namesInTree.get(node.id, node.id) if namesInTree else node.id
        return name
    else:
        children_str = []
        for child in node.children:
            children_str.append(createNewick(tree, child, binary, namesInTree))
        return "(" + ",".join(children_str) + ")" + node.id
    

def createNewick(tree, node, binary, namesInTree, includeMutationList, estimateMAT, networkOutput, sprtaOn, minSupport, count0BLenNodesOnce, includeSupports, keepInputIQtreeSupports, aBayesPlusOn, performLineageAssignmentByRefPlacement):
    """ Serialize tree to Newick/Nexus; optionally include supports, MAT, and metadata.

Inputs: ['tree', 'node', 'binary', 'namesInTree', 'includeMutationList', 'estimateMAT', 'networkOutput', 'sprtaOn', 'minSupport', 'count0BLenNodesOnce', 'includeSupports', 'keepInputIQtreeSupports', 'aBayesPlusOn', 'performLineageAssignmentByRefPlacement']
Outputs: see return docs in MAPLE source.

binary — currently used only to control how multifurcations are handled (keeps them as-is; see note).

namesInTree — optional mapping from node id to display name.

includeMutationList — appends mutation annotations to node labels.

includeSupports — includes node support values in internal node labels.

minSupport — suppresses supports below threshold.

count0BLenNodesOnce — if True, any zero-length branch annotation printed only once (basic support implemented).

performLineageAssignmentByRefPlacement — if present, will add lineage labels if available on node as node.lineage or node['lineage'].
"""
    pass

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
    def _bl_str(n):
        bl = _get(n, "branch_length", _get(n, "bl", None))
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
