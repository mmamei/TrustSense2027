"""Dataset schema and modality definitions for ExtraSensory."""

from __future__ import annotations

from collections import OrderedDict


MODALITY_PREFIXES: OrderedDict[str, tuple[str, ...]] = OrderedDict(
    [
        ("phone_acc", ("raw_acc:",)),
        ("phone_gyro", ("proc_gyro:",)),
        ("phone_magnet", ("raw_magnet:",)),
        ("watch_acc", ("watch_acceleration:",)),
        ("watch_heading", ("watch_heading:",)),
        ("location", ("location:", "location_quick_features:")),
        ("audio", ("audio_naive:", "audio_properties:")),
        ("phone_state", ("discrete:", "lf_measurements:")),
    ]
)


def label_columns(columns: list[str]) -> list[str]:
    """Return cleaned ExtraSensory label columns."""
    return [column for column in columns if column.startswith("label:")]


def modality_columns(columns: list[str]) -> dict[str, list[str]]:
    """Map each modality to the feature columns assigned to it."""
    mapping: dict[str, list[str]] = {}
    for modality, prefixes in MODALITY_PREFIXES.items():
        mapping[modality] = [
            column for column in columns if any(column.startswith(prefix) for prefix in prefixes)
        ]
    return mapping

