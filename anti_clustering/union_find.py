# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
A union find data structure for collecting results of the anti-clustering algorithm.
"""

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
