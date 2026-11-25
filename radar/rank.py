import heapq


def top_k(counts, k):
    """Largest k counts using a size-k min-heap: O(m log k) for m themes.

    Heap items are (count, -position, id). When the heap grows past k the
    smallest item is dropped, and on equal counts the later theme goes first,
    so ties keep the earlier theme.
    """
    heap = []
    for pos, (theme_id, n) in enumerate(counts.items()):
        if n == 0:
            continue
        heapq.heappush(heap, (n, -pos, theme_id))
        if len(heap) > k:
            heapq.heappop(heap)
    return [(theme_id, n) for n, _, theme_id in sorted(heap, reverse=True)]
