"""Aggregate caption length statistics."""

from typing import Dict, Set


def calculate_average_tag_count(tags_dict: Dict[str, Set[str]]) -> Dict[str, float]:
    """
    Вычисляет среднее, минимальное и максимальное количество тегов в словаре тегов.

    Args:
        tags_dict: Словарь, где ключи - имена изображений, а значения - множества тегов

    Returns:
        Словарь со статистикой по количеству тегов
    """
    if not tags_dict:
        return {"avg": 0, "min": 0, "max": 0, "median": 0}
    tag_counts = [len(tags) for tags in tags_dict.values()]
    avg_count = sum(tag_counts) / len(tag_counts)
    min_count = min(tag_counts)
    max_count = max(tag_counts)
    median_count = sorted(tag_counts)[len(tag_counts) // 2]
    return {"avg": avg_count, "min": min_count, "max": max_count, "median": median_count}
