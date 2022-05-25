class UnionFind:
    parent_node = {}

    def initialize(self, u):
        for idx, i in enumerate(u):
            self.parent_node[idx] = i

    def find(self, k):
        if self.parent_node[k] == k:
            return k
        return self.find(self.parent_node[k])

    def union(self, a, b):
        x = self.find(a)
        y = self.find(b)
        self.parent_node[x] = y

    def connected(self, a, b):
        return self.find(a) == self.find(b)
