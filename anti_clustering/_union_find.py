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

from typing import TypeVar, Generic

T = TypeVar("T")  # pylint: disable=C0103


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

    def _find(self, element: T) -> T:
        """
        Find the root of component of element a.
        :param element: Element to find root of.
        :return: The root of the component.
        """
        # Compresses path while iterating up the tree.
        while element != self._parent[element]:
            parent = self._parent[element]
            self._parent[element] = self._parent[parent]
            element = parent

        return element

    def find(self, element: T) -> T:
        """
        Find the root of component of element a.
        :param element: Element to find root of.
        :return: The root of the component.
        """
        return self._find(element)

    def union(self, element_1: T, element_2: T) -> None:
        """
        Unify components of two elements.
        :param element_1: Element to unify.
        :param element_2: Other element to unify.
        :return:
        """
        if element_1 == element_2:
            return

        root_1 = self._find(element_1)
        root_2 = self._find(element_2)

        if root_1 == root_2:
            return

        # Weighted union - the smaller component becomes the child of the root of the larger component.
        if self._size[root_1] < self._size[root_2]:
            self._parent[root_1] = root_2
            self._size[root_2] += self._size[root_1]
        else:
            self._parent[root_2] = root_1
            self._size[root_1] += self._size[root_2]
        self.components_count -= 1

    def connected(self, element_1: T, element_2: T) -> bool:
        """
        Check if element a and b are in the same component.
        :param element_1: Element to check.
        :param element_2: Other element to check.
        :return: Whether a and b are in the same component.
        """
        return self._find(element_1) == self._find(element_2)
