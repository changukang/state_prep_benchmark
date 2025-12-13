from state_preparation.benchmark.utils import apply_perm_to_edges, graph_to_bit_string


def test_graph_to_bit_string():
    n = 3  # then the graph size is 3
    graph = [(0, 1), (1, 2)]

    assert graph_to_bit_string(n, graph) == [1, 0, 1]


def test_graph_permutation():
    graph = [(0, 1), (1, 2), (2, 3)]
    assert set(apply_perm_to_edges([3, 1, 2, 0], graph)) == {(1, 3), (1, 2), (0, 2)}
