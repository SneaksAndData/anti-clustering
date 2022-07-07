# Copyright 2022 ECCO Sneaks & Data
#
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

Based on:
Sedgewick, R. & Wayne, K. (2011), Algorithms, 4th Edition. , Addison-Wesley .
"""

from typing import Dict, TypeVar, Generic

T = TypeVar('T')  # pylint: disable=C0103


class UnionFind(Generic[T]):
    """
    A union find data structure for collecting results of the anti-clustering algorithm.
    This implementation uses the weighted quick union with path compression.
    """
    # A mapping from an element to its parent. If a parent maps to itself, it is the root of the component.
    _parent = {}
    _size = {}
    components_count = 0

    def __init__(self, initial_components_count: int):
        """
        Initialize UnionFind with components.
        :param initial_components_count: The initial number of components.
        """
        self.components_count = initial_components_count
        self._parent = {i: i for i in range(initial_components_count)}
        self._size = {i: 1 for i in range(initial_components_count)}

    def _find(self, a: T) -> T:
        """
        Find the root of component of element a.
        :param a: Element to find root of.
        :return: The root of the component.
        """
        while a != self._parent[a]:
            b = self._parent[a]
            self._parent[a] = self._parent[b]
            a = b

        return a

    def find(self, a: T) -> T:
        """
        Find the root of component of element a.
        :param a: Element to find root of.
        :return: The root of the component.
        """
        return self._find(a)

    def union(self, a: T, b: T) -> None:
        """
        Unify components of two elements.
        :param a: Element to unify.
        :param b: Other element to unify.
        :return:
        """
        if a == b:
            return

        x = self._find(a)
        y = self._find(b)

        if x == y:
            return
        if self._size[x] < self._size[y]:
            self._parent[x] = y
            self._size[y] += self._size[x]
        else:
            self._parent[y] = x
            self._size[x] += self._size[y]
        self.components_count -= 1

    def connected(self, a: T, b: T) -> bool:
        """
        Check if element a and b are in the same component.
        :param a: Element to check.
        :param b: Other element to check.
        :return: Whether a and b are in the same component.
        """
        return self._find(a) == self._find(b)
