from anti_clustering._union_find import UnionFind


def test_union_self():
    """
    Tests that unioning a node with itself does not connect it to other components.
    """
    uf = UnionFind(5)
    uf.union(0, 0)
    uf.union(1, 2)
    uf.union(2, 3)
    uf.union(3, 4)
    uf.union(4, 4)
    assert uf.components_count == 2
    assert uf.find(0) == 0
    assert uf.find(1) == uf.find(2) == uf.find(3) == uf.find(4)
    assert uf.find(0) != uf.find(1)
    assert uf.connected(0, 0)
    assert not uf.connected(0, 1)
    assert uf.connected(1, 4)


def test_construction():
    """
    Tests that the initial state is completely disconnected.
    """
    uf = UnionFind(3)
    assert uf.components_count == 3
    assert uf.find(0) == 0
    assert uf.find(1) == 1
    assert uf.find(2) == 2
    assert uf.connected(0, 0)
    assert not uf.connected(0, 1)
    assert not uf.connected(1, 2)
    assert not uf.connected(0, 2)


def test_imbalanced_union():
    """
    Tests that performing a chain of unioned elements connects all in the same components.
    This is interesting due to the weighted union.
    """
    uf = UnionFind(10)

    for i in range(9):
        uf.union(i, i+1)

    assert uf.components_count == 1
    assert uf.find(0) == uf.find(1) == uf.find(2) == uf.find(3) ==\
           uf.find(4) == uf.find(5) == uf.find(6) == uf.find(7) == \
           uf.find(8) == uf.find(9) == 0

    assert uf.connected(0, 9)
