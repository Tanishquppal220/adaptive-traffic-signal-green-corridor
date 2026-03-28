from __future__ import annotations

from typing import Iterable

from config import DIRECTION_TO_LANE, DIRECTIONS, LANE_KEYS, LANE_TO_DIRECTION


def empty_lane_counts() -> dict[str, int]:
    return {lane: 0 for lane in LANE_KEYS}


def lane_counts_to_direction_counts(lane_counts: dict[str, int]) -> dict[str, int]:
    return {
        direction: int(lane_counts.get(DIRECTION_TO_LANE[direction], 0))
        for direction in DIRECTIONS
    }


def direction_counts_to_lane_counts(direction_counts: dict[str, int]) -> dict[str, int]:
    return {
        DIRECTION_TO_LANE[direction]: int(direction_counts.get(direction, 0))
        for direction in DIRECTIONS
    }


def normalize_lane_counts(input_counts: dict[str, int] | None) -> dict[str, int]:
    if not input_counts:
        return empty_lane_counts()

    normalized = empty_lane_counts()
    for lane in LANE_KEYS:
        normalized[lane] = max(0, int(input_counts.get(lane, 0)))
    for direction in DIRECTIONS:
        if direction in input_counts:
            lane = DIRECTION_TO_LANE[direction]
            normalized[lane] = max(0, int(input_counts[direction]))
    return normalized


def resolve_direction_from_point(
    x: float,
    y: float,
    frame_width: int,
    frame_height: int,
) -> str:
    center_x = frame_width / 2.0
    center_y = frame_height / 2.0
    dx = x - center_x
    dy = y - center_y

    if abs(dy) >= abs(dx):
        return "N" if dy < 0 else "S"
    return "W" if dx < 0 else "E"


def top_direction(counts: dict[str, int], preference: Iterable[str] = DIRECTIONS) -> str:
    best = None
    best_value = -1
    for direction in preference:
        value = int(counts.get(direction, 0))
        if value > best_value:
            best = direction
            best_value = value
    return best or "N"


def lane_key_for_direction(direction: str) -> str:
    return DIRECTION_TO_LANE[direction]


def direction_for_lane_key(lane_key: str) -> str:
    return LANE_TO_DIRECTION[lane_key]
