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

from typing import Dict, TypeVar, Generic

T = TypeVar('T')  # pylint: disable=C0103


class UnionFind(Generic[T]):
    """
    A union find data structure for collecting results of the anti-clustering algorithm.
    """
    # A mapping from an element to its parent. If a parent maps to itself, it is the root of the component.
    parent = {}

    def __init__(self, parent: Dict[T, T]):
        """
        Initialize UnionFind with components.
        :param parent: The initial components.
            In most use cases all components will point to themselves (example: {0: 0, 1: 1, ...}).
        """
        self.parent = parent

    def find(self, a: T) -> T:
        """
        Find the root of component of element a.
        :param a: Element to find root of.
        :return: The root of the component.
        """
        if self.parent[a] == a:
            return a
        return self.find(self.parent[a])

    def union(self, a: T, b: T) -> None:
        """
        Unify components of two elements.
        :param a: Element to unify.
        :param b: Other element to unify.
        :return:
        """
        x = self.find(a)
        y = self.find(b)
        self.parent[x] = y

    def connected(self, a: T, b: T) -> bool:
        """
        Check if element a and b are in the same component.
        :param a: Element to check.
        :param b: Other element to check.
        :return: Whether a and b are in the same component.
        """
        return self.find(a) == self.find(b)
