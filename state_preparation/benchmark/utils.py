from typing import List, Sequence, Tuple


def graph_to_bit_string(n: int, graph: Sequence[Tuple[int, int]]) -> Sequence[int]:
    graph_size = n * (n - 1) / 2
    assert graph_size.is_integer()
    graph_size = int(graph_size)
    ret = [
        1 if (i, j) in graph or (j, i) in graph else 0
        for i in range(graph_size)
        for j in range(i + 1, graph_size)
    ]
    assert len(ret) == n, "Graph does not match expected size"
    return ret


def apply_perm_to_edges(
    perm: Sequence[int], edges: Sequence[Tuple[int, int]]
) -> List[Tuple[int, int]]:
    ret = list()
    for i, j in edges:
        a, b = perm[i], perm[j]
        ret.append((a, b) if a < b else (b, a))
    return ret
