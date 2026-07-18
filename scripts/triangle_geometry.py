#!/usr/bin/env python3
"""Dependency-free triangle intersection, distance, and deterministic BVH helpers."""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass
from typing import Any, Iterable


EPSILON = 1e-12
Vector = list[float]
Triangle = tuple[Vector, Vector, Vector]


class TriangleError(ValueError):
    """Invalid or empty triangle surface input."""


def add(a: Vector, b: Vector) -> Vector:
    return [a[index] + b[index] for index in range(3)]


def subtract(a: Vector, b: Vector) -> Vector:
    return [a[index] - b[index] for index in range(3)]


def scale(vector: Vector, factor: float) -> Vector:
    return [value * factor for value in vector]


def dot(a: Vector, b: Vector) -> float:
    return sum(a[index] * b[index] for index in range(3))


def cross(a: Vector, b: Vector) -> Vector:
    return [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]


def norm_squared(vector: Vector) -> float:
    return dot(vector, vector)


def clamp(value: float, minimum: float = 0.0, maximum: float = 1.0) -> float:
    return max(minimum, min(maximum, value))


def interpolate(start: Vector, end: Vector, parameter: float) -> Vector:
    return add(start, scale(subtract(end, start), parameter))


def closest_point_on_segment(point: Vector, start: Vector, end: Vector) -> Vector:
    direction = subtract(end, start)
    denominator = norm_squared(direction)
    if denominator <= EPSILON * EPSILON:
        return list(start)
    parameter = clamp(dot(subtract(point, start), direction) / denominator)
    return interpolate(start, end, parameter)


def closest_point_on_triangle(point: Vector, triangle: Triangle) -> Vector:
    a, b, c = triangle
    ab, ac, ap = subtract(b, a), subtract(c, a), subtract(point, a)
    normal_squared = norm_squared(cross(ab, ac))
    if normal_squared <= EPSILON * EPSILON:
        candidates = [
            closest_point_on_segment(point, a, b),
            closest_point_on_segment(point, b, c),
            closest_point_on_segment(point, c, a),
        ]
        return min(candidates, key=lambda candidate: norm_squared(subtract(point, candidate)))
    d1, d2 = dot(ab, ap), dot(ac, ap)
    if d1 <= 0.0 and d2 <= 0.0:
        return list(a)
    bp = subtract(point, b)
    d3, d4 = dot(ab, bp), dot(ac, bp)
    if d3 >= 0.0 and d4 <= d3:
        return list(b)
    vc = d1 * d4 - d3 * d2
    if vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0:
        parameter = d1 / (d1 - d3)
        return add(a, scale(ab, parameter))
    cp = subtract(point, c)
    d5, d6 = dot(ab, cp), dot(ac, cp)
    if d6 >= 0.0 and d5 <= d6:
        return list(c)
    vb = d5 * d2 - d1 * d6
    if vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0:
        parameter = d2 / (d2 - d6)
        return add(a, scale(ac, parameter))
    va = d3 * d6 - d5 * d4
    if va <= 0.0 and d4 - d3 >= 0.0 and d5 - d6 >= 0.0:
        parameter = (d4 - d3) / ((d4 - d3) + (d5 - d6))
        return add(b, scale(subtract(c, b), parameter))
    denominator = 1.0 / (va + vb + vc)
    v, w = vb * denominator, vc * denominator
    return add(a, add(scale(ab, v), scale(ac, w)))


def closest_points_on_segments(p1: Vector, q1: Vector, p2: Vector, q2: Vector) -> tuple[Vector, Vector]:
    d1, d2, r = subtract(q1, p1), subtract(q2, p2), subtract(p1, p2)
    a, e = norm_squared(d1), norm_squared(d2)
    if a <= EPSILON * EPSILON and e <= EPSILON * EPSILON:
        return list(p1), list(p2)
    if a <= EPSILON * EPSILON:
        s, t = 0.0, clamp(dot(d2, r) / e)
    else:
        c = dot(d1, r)
        if e <= EPSILON * EPSILON:
            s, t = clamp(-c / a), 0.0
        else:
            b, f = dot(d1, d2), dot(d2, r)
            denominator = a * e - b * b
            s = clamp((b * f - c * e) / denominator) if abs(denominator) > EPSILON * EPSILON else 0.0
            t = (b * s + f) / e
            if t < 0.0:
                t, s = 0.0, clamp(-c / a)
            elif t > 1.0:
                t, s = 1.0, clamp((b - c) / a)
    return interpolate(p1, q1, s), interpolate(p2, q2, t)


def segment_triangle_intersection(start: Vector, end: Vector, triangle: Triangle) -> Vector | None:
    a, b, c = triangle
    direction = subtract(end, start)
    edge1, edge2 = subtract(b, a), subtract(c, a)
    pvec = cross(direction, edge2)
    determinant = dot(edge1, pvec)
    scale_hint = max(1.0, math.sqrt(norm_squared(direction) * norm_squared(edge1) * norm_squared(edge2)))
    tolerance = EPSILON * scale_hint
    if abs(determinant) <= tolerance:
        return None
    inverse = 1.0 / determinant
    tvec = subtract(start, a)
    u = dot(tvec, pvec) * inverse
    if u < -EPSILON or u > 1.0 + EPSILON:
        return None
    qvec = cross(tvec, edge1)
    v = dot(direction, qvec) * inverse
    if v < -EPSILON or u + v > 1.0 + EPSILON:
        return None
    parameter = dot(edge2, qvec) * inverse
    if parameter < -EPSILON or parameter > 1.0 + EPSILON:
        return None
    return interpolate(start, end, clamp(parameter))


def triangle_triangle_distance(left: Triangle, right: Triangle) -> tuple[float, Vector, Vector]:
    left_edges = ((left[0], left[1]), (left[1], left[2]), (left[2], left[0]))
    right_edges = ((right[0], right[1]), (right[1], right[2]), (right[2], right[0]))
    for start, end in left_edges:
        point = segment_triangle_intersection(start, end, right)
        if point is not None:
            return 0.0, point, list(point)
    for start, end in right_edges:
        point = segment_triangle_intersection(start, end, left)
        if point is not None:
            return 0.0, list(point), point
    best_squared = math.inf
    best_left, best_right = list(left[0]), list(right[0])
    for point in left:
        other = closest_point_on_triangle(point, right)
        distance_squared = norm_squared(subtract(point, other))
        if distance_squared < best_squared:
            best_squared, best_left, best_right = distance_squared, list(point), other
    for point in right:
        other = closest_point_on_triangle(point, left)
        distance_squared = norm_squared(subtract(other, point))
        if distance_squared < best_squared:
            best_squared, best_left, best_right = distance_squared, other, list(point)
    for left_start, left_end in left_edges:
        for right_start, right_end in right_edges:
            left_point, right_point = closest_points_on_segments(left_start, left_end, right_start, right_end)
            distance_squared = norm_squared(subtract(left_point, right_point))
            if distance_squared < best_squared:
                best_squared, best_left, best_right = distance_squared, left_point, right_point
    return math.sqrt(max(0.0, best_squared)), best_left, best_right


def triangle_bounds(triangle: Triangle) -> tuple[Vector, Vector]:
    return (
        [min(vertex[axis] for vertex in triangle) for axis in range(3)],
        [max(vertex[axis] for vertex in triangle) for axis in range(3)],
    )


def aabb_distance_squared(left_minimum: Vector, left_maximum: Vector, right_minimum: Vector, right_maximum: Vector) -> float:
    squared = 0.0
    for axis in range(3):
        gap = max(0.0, right_minimum[axis] - left_maximum[axis], left_minimum[axis] - right_maximum[axis])
        squared += gap * gap
    return squared


@dataclass
class BVHNode:
    minimum: Vector
    maximum: Vector
    triangle_indices: tuple[int, ...]
    left: "BVHNode | None" = None
    right: "BVHNode | None" = None

    @property
    def is_leaf(self) -> bool:
        return self.left is None and self.right is None


def build_bvh(triangles: list[Triangle], leaf_size: int = 8) -> BVHNode:
    if not triangles:
        raise TriangleError("triangle surface contains no triangles")
    if leaf_size < 1:
        raise TriangleError("BVH leaf_size must be positive")
    bounds = [triangle_bounds(triangle) for triangle in triangles]
    centroids = [[sum(vertex[axis] for vertex in triangle) / 3.0 for axis in range(3)] for triangle in triangles]

    def build(indices: list[int]) -> BVHNode:
        minimum = [min(bounds[index][0][axis] for index in indices) for axis in range(3)]
        maximum = [max(bounds[index][1][axis] for index in indices) for axis in range(3)]
        if len(indices) <= leaf_size:
            return BVHNode(minimum, maximum, tuple(indices))
        centroid_minimum = [min(centroids[index][axis] for index in indices) for axis in range(3)]
        centroid_maximum = [max(centroids[index][axis] for index in indices) for axis in range(3)]
        axis = max(range(3), key=lambda value: centroid_maximum[value] - centroid_minimum[value])
        ordered = sorted(indices, key=lambda index: (centroids[index][axis], index))
        midpoint = len(ordered) // 2
        left, right = build(ordered[:midpoint]), build(ordered[midpoint:])
        return BVHNode(minimum, maximum, tuple(), left, right)

    return build(list(range(len(triangles))))


def bvh_surface_distance(
    left_triangles: list[Triangle],
    right_triangles: list[Triangle],
    left_bvh: BVHNode | None = None,
    right_bvh: BVHNode | None = None,
    leaf_size: int = 8,
) -> dict[str, Any]:
    left_root = left_bvh or build_bvh(left_triangles, leaf_size)
    right_root = right_bvh or build_bvh(right_triangles, leaf_size)
    queue: list[tuple[float, int, BVHNode, BVHNode]] = []
    sequence = 0

    def enqueue(left: BVHNode, right: BVHNode) -> None:
        nonlocal sequence
        lower_bound = aabb_distance_squared(left.minimum, left.maximum, right.minimum, right.maximum)
        heapq.heappush(queue, (lower_bound, sequence, left, right))
        sequence += 1

    enqueue(left_root, right_root)
    best_distance_squared = math.inf
    best_left, best_right = list(left_triangles[0][0]), list(right_triangles[0][0])
    best_left_index, best_right_index = 0, 0
    node_pairs_visited = 0
    triangle_pairs_tested = 0
    while queue:
        lower_bound, _, left_node, right_node = heapq.heappop(queue)
        if lower_bound > best_distance_squared:
            continue
        node_pairs_visited += 1
        if left_node.is_leaf and right_node.is_leaf:
            for left_index in left_node.triangle_indices:
                for right_index in right_node.triangle_indices:
                    triangle_pairs_tested += 1
                    distance, left_point, right_point = triangle_triangle_distance(left_triangles[left_index], right_triangles[right_index])
                    distance_squared = distance * distance
                    if distance_squared < best_distance_squared:
                        best_distance_squared = distance_squared
                        best_left, best_right = left_point, right_point
                        best_left_index, best_right_index = left_index, right_index
            continue
        if left_node.is_leaf:
            assert right_node.left is not None and right_node.right is not None
            enqueue(left_node, right_node.left)
            enqueue(left_node, right_node.right)
        elif right_node.is_leaf:
            assert left_node.left is not None and left_node.right is not None
            enqueue(left_node.left, right_node)
            enqueue(left_node.right, right_node)
        else:
            assert left_node.left is not None and left_node.right is not None
            assert right_node.left is not None and right_node.right is not None
            left_extent = math.prod(left_node.maximum[axis] - left_node.minimum[axis] for axis in range(3))
            right_extent = math.prod(right_node.maximum[axis] - right_node.minimum[axis] for axis in range(3))
            if left_extent >= right_extent:
                enqueue(left_node.left, right_node)
                enqueue(left_node.right, right_node)
            else:
                enqueue(left_node, right_node.left)
                enqueue(left_node, right_node.right)
    return {
        "distance_m": math.sqrt(max(0.0, best_distance_squared)),
        "witness_point_left": best_left,
        "witness_point_right": best_right,
        "left_triangle_index": best_left_index,
        "right_triangle_index": best_right_index,
        "node_pairs_visited": node_pairs_visited,
        "triangle_pairs_tested": triangle_pairs_tested,
    }


def point_inside_closed_surface(point: Vector, triangles: list[Triangle]) -> bool:
    """Classify a point by the generalized winding solid angle of a closed surface."""
    if not triangles:
        raise TriangleError("closed surface contains no triangles")
    solid_angle = 0.0
    for triangle in triangles:
        a, b, c = (subtract(vertex, point) for vertex in triangle)
        length_a = math.sqrt(norm_squared(a))
        length_b = math.sqrt(norm_squared(b))
        length_c = math.sqrt(norm_squared(c))
        if min(length_a, length_b, length_c) <= EPSILON:
            return True
        numerator = dot(a, cross(b, c))
        denominator = (
            length_a * length_b * length_c
            + dot(a, b) * length_c
            + dot(b, c) * length_a
            + dot(c, a) * length_b
        )
        solid_angle += 2.0 * math.atan2(numerator, denominator)
    return abs(solid_angle) > 2.0 * math.pi


def transform_triangles(vertices: list[Vector], faces: Iterable[tuple[int, int, int]], transform: list[list[float]]) -> list[Triangle]:
    transformed = [
        [sum(transform[row][column] * vertex[column] for column in range(3)) + transform[row][3] for row in range(3)]
        for vertex in vertices
    ]
    return [(transformed[a], transformed[b], transformed[c]) for a, b, c in faces]


def box_surface(size_xyz: Vector, transform: list[list[float]]) -> list[Triangle]:
    half = [value / 2.0 for value in size_xyz]
    vertices = [[x, y, z] for x in (-half[0], half[0]) for y in (-half[1], half[1]) for z in (-half[2], half[2])]
    faces = [
        (0, 2, 3), (0, 3, 1),
        (4, 5, 7), (4, 7, 6),
        (0, 1, 5), (0, 5, 4),
        (2, 6, 7), (2, 7, 3),
        (0, 4, 6), (0, 6, 2),
        (1, 3, 7), (1, 7, 5),
    ]
    return transform_triangles(vertices, faces, transform)
