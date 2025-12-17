from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Set, FrozenSet, Tuple


@dataclass
class Node:
    label: str
    children: List["Node"]


def parse_newick(newick: str) -> Node:
    s = "".join(ch for ch in newick.strip() if not ch.isspace())
    if not s.endswith(";"):
        raise ValueError("Newick must end with ';'")
    s = s[:-1]
    i = 0

    def parse_subtree() -> Node:
        nonlocal i
        children: List[Node] = []

        if i < len(s) and s[i] == "(":
            i += 1
            while True:
                children.append(parse_subtree())
                if i >= len(s):
                    raise ValueError("Unexpected end while parsing children")
                if s[i] == ",":
                    i += 1
                    continue
                if s[i] == ")":
                    i += 1
                    break
                raise ValueError(f"Unexpected character in children list: {s[i]!r}")

        # optional label (can be empty)
        label_chars = []
        while i < len(s) and s[i] not in ":,()":
            label_chars.append(s[i])
            i += 1
        label = "".join(label_chars)

        # optional branch length
        if i < len(s) and s[i] == ":":
            i += 1
            # consume float/scientific until delimiter
            while i < len(s) and s[i] not in ",()":
                i += 1

        return Node(label=label, children=children)

    root = parse_subtree()
    if i != len(s):
        raise ValueError(f"Trailing content after Newick parse at pos {i}/{len(s)}")
    return root


def leaf_labels(root: Node) -> List[str]:
    out: List[str] = []

    def dfs(n: Node) -> None:
        if not n.children:
            if n.label:
                out.append(n.label)
            return
        for c in n.children:
            dfs(c)

    dfs(root)
    return out


def rooted_splits(root: Node) -> Set[FrozenSet[str]]:
    leaves = set(leaf_labels(root))
    if not leaves:
        return set()

    splits: Set[FrozenSet[str]] = set()

    def dfs(n: Node) -> Set[str]:
        if not n.children:
            return {n.label} if n.label else set()
        acc: Set[str] = set()
        for c in n.children:
            acc |= dfs(c)
        # clade under this node
        if 1 < len(acc) < len(leaves):
            splits.add(frozenset(acc))
        return acc

    dfs(root)
    # drop the root clade (it equals all leaves) if it slipped in
    splits.discard(frozenset(leaves))
    return splits


def compare(a_path: Path, b_path: Path) -> None:
    a = parse_newick(a_path.read_text(encoding="utf-8", errors="replace"))
    b = parse_newick(b_path.read_text(encoding="utf-8", errors="replace"))

    a_leaves = set(leaf_labels(a))
    b_leaves = set(leaf_labels(b))

    print(f"A: {a_path}\n  leaves={len(a_leaves)}")
    print(f"B: {b_path}\n  leaves={len(b_leaves)}")

    missing_in_a = sorted(b_leaves - a_leaves)
    missing_in_b = sorted(a_leaves - b_leaves)
    print(f"Leaf set mismatch: missing_in_A={len(missing_in_a)} missing_in_B={len(missing_in_b)}")
    if missing_in_a:
        print("  missing_in_A (first 20):", ", ".join(missing_in_a[:20]))
    if missing_in_b:
        print("  missing_in_B (first 20):", ", ".join(missing_in_b[:20]))

    # Only compare splits on shared taxa
    shared = a_leaves & b_leaves
    if len(shared) < 4:
        print("Not enough shared leaves to compare topology.")
        return

    def restrict(root: Node, keep: Set[str]) -> Node:
        # Prune leaves not in keep; suppress unary nodes.
        def prune(n: Node) -> Node | None:
            if not n.children:
                return n if n.label in keep else None
            kept = [prune(c) for c in n.children]
            kept = [c for c in kept if c is not None]
            if not kept:
                return None
            if len(kept) == 1:
                return kept[0]
            return Node(label=n.label, children=kept)

        pruned = prune(root)
        if pruned is None:
            return Node(label="", children=[])
        return pruned

    a_r = restrict(a, shared)
    b_r = restrict(b, shared)

    a_s = rooted_splits(a_r)
    b_s = rooted_splits(b_r)

    inter = a_s & b_s
    rf = (len(a_s - b_s) + len(b_s - a_s))

    print(f"Shared leaves used for topology: {len(shared)}")
    print(f"Rooted splits: A={len(a_s)} B={len(b_s)} shared={len(inter)}")
    print(f"Rooted RF-like distance (split symmetric difference): {rf}")


def main() -> None:
    mine = Path(r"c:\Users\hanna\Documents\cs4775\output_tree.newick")
    maple = Path(r"c:\Users\hanna\Documents\cs4775\FinaProject\MAPLE_outputs_original\MAPLE_outputFilePrefix_tree.tree_tree.tree")
    compare(mine, maple)


if __name__ == "__main__":
    main()
