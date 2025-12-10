import random
import heapq

num_comps = 0
num_comp_batches = 0

def partition(docs, indexes: list[int], low: int, high: int, K: int, pivot_value = None) -> int:
    global num_comps, num_comp_batches
    i = low - 1

    # With embedding optimization
    if pivot_value == None:
        if K <= high - low:
            pivot_value = heapq.nsmallest(K, indexes[low : high + 1])[-1]
        else:
            pivot_value = heapq.nsmallest(int((high - low + 1) / 2), indexes[low : high + 1])[-1]
    pivot_index = indexes.index(pivot_value)

    pivot = docs[pivot_value]
    indexes[pivot_index], indexes[high] = indexes[high], indexes[pivot_index]

    pairs = [(docs[indexes[j]], pivot) for j in range(low, high)]
    comparisons = [(a < b) for a,b in pairs]
    num_comps += len(comparisons)
    num_comp_batches += 1

    for j, doc1_is_better in enumerate(comparisons, start=low):
        if doc1_is_better:
            i += 1
            indexes[i], indexes[j] = indexes[j], indexes[i]

    indexes[i + 1], indexes[high] = indexes[high], indexes[i + 1]
    return i + 1

def quicksort_recursive(docs, indexes: list[int], low: int, high: int, K: int, opt=True) -> None:
    if high <= low:
        return

    if opt and (high - low <= K):
        return

    pi = partition(docs, indexes, low, high, K)
    left_size = pi - low
    if left_size + 1 >= K:
        quicksort_recursive(docs, indexes, low, pi - 1, K)
    else:
        if not opt:
            quicksort_recursive(docs, indexes, low, pi - 1, left_size)
        quicksort_recursive(docs, indexes, pi + 1, high, K - left_size - 1)

def quick_topk(docs, wanted_k, opt=True):
    indexes = list(range(len(docs)))
    quicksort_recursive(docs, indexes, 0, len(indexes) - 1, wanted_k, opt=opt)

    return [docs[i] for i in indexes[:wanted_k]]

def incr_quick_topk(existing_docs, new_docs, wanted_k):
    global num_comps, num_comp_batches

    # 1. compare all of new to existing[-1]
    pivot = existing_docs[-1]
    pairs = [(new_docs[j], pivot) for j in range(0, len(new_docs))]
    comparisons = [(a < b) for a,b in pairs]

    num_comp_batches += 1
    num_comps += len(comparisons)

    # 2. if any were less, remove existing[-1] (guaranteed not to be in topk) and append the less ones
    # existing[-2] likely to be near the top, so use as pivot
    pivot_idx = len(existing_docs) - 2
    new_cands = [new_docs[idx] for idx, lt in enumerate(comparisons) if lt]
    if len(new_cands) == 0:
        return existing_docs

    existing_docs = existing_docs[:-1] + new_cands
    partition(existing_docs, list(range(len(existing_docs))), 0, len(existing_docs) - 1, wanted_k, pivot_value=pivot_idx)

    # 3. now need to quick_topk existing_docs to get it back down to size.
    return quick_topk(existing_docs, wanted_k)

if __name__ == '__main__':
    x = list(range(100))
    random.seed(42)
    random.shuffle(x)

    wanted_k = 15

    # way -1: lotus way (full sort)
    num_comp_batches = 0
    num_comps = 0
    r1 = quick_topk(x, wanted_k, opt=False)
    print(r1)
    print('lotus quick_topk in ', num_comp_batches, 'batches, ', num_comps, ' comparisons')

    # way 0: quick_topk once
    num_comp_batches = 0
    num_comps = 0
    r1 = quick_topk(x, wanted_k)
    print(r1)
    print('1round quick_topk in ', num_comp_batches, 'batches, ', num_comps, ' comparisons')

    # way 1: quick_topk twice
    num_comp_batches = 0
    num_comps = 0
    x1 = x[:50]
    x2 = x[50:]
    r1 = quick_topk(x1, wanted_k)
    r1.extend(x2)
    print(quick_topk(r1, wanted_k))

    print('2round quick_topk in ', num_comp_batches, 'batches, ', num_comps, ' comparisons')

    # way 2: incr_quick_topk
    num_comp_batches = 0
    num_comps = 0
    r1 = quick_topk(x1, wanted_k)
    r1 = incr_quick_topk(r1, x2, wanted_k)
    print(r1)
    print('incr quick_topk in ', num_comp_batches, 'batches, ', num_comps, ' comparisons')
