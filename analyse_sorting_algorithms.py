import numpy as np
import timeit
import matplotlib.pyplot as plt

# Quick Sort Algorithm
def quicksort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quicksort(left) + middle + quicksort(right)

# Heap Sort Algorithm
def heapify(arr, n, i):
    largest = i
    left = 2 * i + 1
    right = 2 * i + 2

    if left < n and arr[left] > arr[largest]:
        largest = left

    if right < n and arr[right] > arr[largest]:
        largest = right

    if largest != i:
        arr[i], arr[largest] = arr[largest], arr[i]
        heapify(arr, n, largest)

def heap_sort(arr):
    n = len(arr)

    for i in range(n // 2 - 1, -1, -1):
        heapify(arr, n, i)

    for i in range(n - 1, 0, -1):
        arr[i], arr[0] = arr[0], arr[i]
        heapify(arr, i, 0)

    return arr

# Merge Sort Algorithm
def merge_sort(arr):
    if len(arr) <= 1:
        return arr
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])
    return custom_merge(left, right)

def custom_merge(left, right):
    merged = []
    i, j = 0, 0
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            merged.append(left[i])
            i += 1
        else:
            merged.append(right[j])
            j += 1
    merged.extend(left[i:])
    merged.extend(right[j:])
    return merged

# Radix Sort Algorithm
def counting_sort(arr, exp):
    n = len(arr)
    output = [0] * n
    count = [0] * 10

    for i in range(n):
        index = arr[i] // exp
        count[index % 10] += 1

    for i in range(1, 10):
        count[i] += count[i - 1]

    i = n - 1
    while i >= 0:
        index = arr[i] // exp
        output[count[index % 10] - 1] = arr[i]
        count[index % 10] -= 1
        i -= 1
    i=0
    for i in range(n):
        arr[i] = output[i]

def radix_sort(arr):
    if len(arr) == 0:
        return arr

    max_num = max(arr)
    exp = 1
    while max_num // exp > 0:
        counting_sort(arr, exp)
        exp *= 10
    return arr

# Bucket Sort Algorithm
def insertion_sort_bucket(bucket):
    for i in range(1, len(bucket)):
        key = bucket[i]
        j = i - 1
        while j >= 0 and key < bucket[j]:
            bucket[j + 1] = bucket[j]
            j -= 1
        bucket[j + 1] = key
    return bucket

def bucket_sort(arr):
    # Determine the range of values in the input array
    max_value = max(arr)
    min_value = min(arr)
    range_of_values = max_value - min_value

    # Define the number of buckets
    num_buckets = len(arr) 

    # Create buckets
    buckets = [[] for _ in range(num_buckets)]

    # Distribute elements into buckets
    for num in arr:
        # Calculate the index for each element
        if range_of_values != 0:
            index = int((num - min_value) / (range_of_values) * (num_buckets - 1))
        else:
            index = 0
        buckets[index].append(num)

    # Sort each bucket and concatenate them
    sorted_arr = []
    for bucket in buckets:
        insertion_sort_bucket(bucket)
        sorted_arr.extend(bucket)

    return sorted_arr

# Timsort Algorithm
def merge(arr, l, m, r):
    left_half = arr[l:m+1]
    right_half = arr[m+1:r+1]

    i = j = 0
    k = l

    while i < len(left_half) and j < len(right_half):
        if left_half[i] <= right_half[j]:
            arr[k] = left_half[i]
            i += 1
        else:
            arr[k] = right_half[j]
            j += 1
        k += 1

    while i < len(left_half):
        arr[k] = left_half[i]
        i += 1
        k += 1

    while j < len(right_half):
        arr[k] = right_half[j]
        j += 1
        k += 1

def insertion_sort_tim(arr, l, r):
    for i in range(l+1, r+1):
        key = arr[i]
        j = i - 1
        while j >= l and arr[j] > key:
            arr[j + 1] = arr[j]
            j -= 1
        arr[j + 1] = key

def timsort(arr):
    min_run = 32

    n = len(arr)
    for i in range(0, n, min_run):
        insertion_sort_tim(arr, i, min((i + min_run - 1), (n - 1)))

    size = min_run
    while size < n:
        for start in range(0, n, size * 2):
            mid = min((start + size - 1), (n - 1))
            end = min((start + size * 2 - 1), (n - 1))
            merge(arr, start, mid, end)
        size *= 2
    return arr

# Generate input data for different scenarios
def generate_random_integers(n):
    return np.random.randint(0, n, n)

def generate_random_integers_range_k(n, k):
    return np.random.randint(0, k, n)

def generate_random_integers_cube(n):
    return np.random.randint(0, n ** 3, n)

def generate_random_integers_log(n):
    return np.random.randint(0, int(np.log(n)), n)

def generate_random_integers_multiple_of_1000(n):
    return np.random.randint(0, n, n) * 1000

def generate_in_order_with_swaps(n):
    arr = np.arange(n)
    num_swaps = int(np.log(n) / 2)
    indices = np.random.choice(n, num_swaps, replace=False)
    for i in range(num_swaps):
        j = np.random.randint(n)
        arr[indices[i]], arr[j] = arr[j], arr[indices[i]]
    return arr

# Test the sorting algorithms on different input scenarios
algorithms = {
    "Quick Sort": quicksort,
    "Heap Sort": heap_sort,
    "Merge Sort": merge_sort,
    "Radix Sort": radix_sort,
    "Bucket Sort": bucket_sort,
    "Timsort": timsort
}

input_generators = {
    "Random Integers": generate_random_integers,
    "Random Integers Range K": lambda n: generate_random_integers_range_k(n, 1000),
    "Random Integers Cube": generate_random_integers_cube,
    "Random Integers Log": generate_random_integers_log,
    "Random Integers Multiple of 1000": generate_random_integers_multiple_of_1000,
    "In Order with Swaps": generate_in_order_with_swaps
}

input_sizes = list(range(1000, 6000, 1000))

for input_name, generate_input in input_generators.items():
    plt.figure(figsize=(10, 6))
    plt.title(f"Performance of Sorting Algorithms for {input_name} Input")
    plt.xlabel("Input Size")
    plt.ylabel("Time (seconds)")
    plt.xticks(input_sizes[::1])  
    for algorithm_name, sorting_algorithm in algorithms.items():
        timings = []
        for input_size in input_sizes:
            arr = generate_input(input_size)
            time_taken = timeit.timeit(lambda: sorting_algorithm(arr.copy()), number=1)
            timings.append(time_taken)
        plt.plot(input_sizes, timings, label=algorithm_name)
    plt.legend()
    plt.grid(True)
    plt.show()
