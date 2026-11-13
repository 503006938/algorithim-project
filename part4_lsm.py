"""
PART 4 - THE LSM-TREE + BLOOM FILTER  (Stage 5)

Idea: do NOT rewrite the disk for every new item.
  1. New items go into a small sorted box in memory (MemTable).
  2. When the box is full, write it to "disk" as ONE sorted batch (a run / SSTable).
  3. When too many runs pile up, merge them into one (compaction).
  4. Every run has a tiny Bloom filter: "definitely NOT here" or "maybe here".
Run:  python3 part4_lsm.py
"""
import bisect                                  # binary search + sorted insert
import random                                  # random test data
import time                                    # stopwatch


class BloomFilter:
    def __init__(self, num_bits, num_hashes=4):
        self.num_bits = max(8, num_bits)       # size of the bit row
        self.num_hashes = num_hashes           # how many bits each key switches on
        self.bits = bytearray(self.num_bits)   # the bit row, all 0 at the start

    def _positions(self, key):                 # which bits belong to this key?
        spots = []
        for i in range(self.num_hashes):       # one position per hash function
            h = hash((i, key))                 # a number made from (i, key)
            h = h ^ (h >> 17)                  # mix the bits so keys spread out
            spots.append(h % self.num_bits)    # squeeze into the bit row
        return spots

    def add(self, key):
        for p in self._positions(key):         # for each of the key's bits ...
            self.bits[p] = 1                   # ... switch it on

    def might_contain(self, key):
        # any bit is 0 -> DEFINITELY not here; all bits are 1 -> MAYBE here
        return all(self.bits[p] for p in self._positions(key))


class SSTable:                                 # one sorted run + its Bloom filter
    def __init__(self, sorted_items, bits_per_key):
        self.keys = [k for k, _ in sorted_items]       # sorted keys
        self.values = [v for _, v in sorted_items]     # matching values
        self.bloom = None                      # no filter unless asked for
        if bits_per_key > 0:
            self.bloom = BloomFilter(bits_per_key * len(self.keys))   # size ~ 10 bits per key
            for k in self.keys:
                self.bloom.add(k)              # register every key in the filter

    def lookup(self, key):                     # binary search inside the run
        i = bisect.bisect_left(self.keys, key)         # first key >= ours
        if i < len(self.keys) and self.keys[i] == key:
            return True, self.values[i]        # found
        return False, None                     # not in this run


class V5LSM:
    def __init__(self, memtable_limit=1000, bloom_bits=10, max_runs=4):
        self.memtable_limit = memtable_limit   # flush when the MemTable has this many keys
        self.bloom_bits = bloom_bits           # bits per key (0 = no Bloom filters)
        self.max_runs = max_runs               # compact when there are more runs than this
        self.mem_keys = []                     # sorted keys in memory
        self.mem_values = {}                   # key -> value in memory
        self.runs = []                         # disk runs, newest first
        self.disk_reads = 0                    # how many runs we really read
        self.wasted_reads = 0                  # reads where the key was NOT there

    def put(self, key, value):
        if key not in self.mem_values:         # new key?
            bisect.insort(self.mem_keys, key)  # keep the MemTable sorted
        self.mem_values[key] = value           # store (or overwrite) the value
        if len(self.mem_keys) >= self.memtable_limit:  # box is full
            self.flush()

    def flush(self):                           # write the MemTable as ONE sorted run
        if not self.mem_keys:
            return
        items = [(k, self.mem_values[k]) for k in self.mem_keys]
        self.runs.insert(0, SSTable(items, self.bloom_bits))   # newest run goes first
        self.mem_keys, self.mem_values = [], {}                # empty the MemTable
        if len(self.runs) > self.max_runs:     # too many runs ...
            self._compact()                     # ... merge them

    def _compact(self):                         # merge ALL runs into one
        merged = {}
        for run in reversed(self.runs):        # oldest run first, newest last
            for k, v in zip(run.keys, run.values):
                merged[k] = v                  # newer value overwrites older one
        self.runs = [SSTable(sorted(merged.items()), self.bloom_bits)]

    def get(self, key):
        if key in self.mem_values:             # 1) look in memory first
            return self.mem_values[key]
        for run in self.runs:                  # 2) then runs, newest to oldest
            if run.bloom is not None and not run.bloom.might_contain(key):
                continue                       # Bloom says "not here": skip this run
            self.disk_reads += 1               # we have to read this run
            found, value = run.lookup(key)
            if found:
                return value
            self.wasted_reads += 1             # read for nothing (false alarm)
        return None                            # key does not exist


def missing_key_experiment(keys, missing, bloom_bits):
    """Store `keys`, then look up keys that are NOT stored.
    Returns (disk reads per lookup, microseconds per lookup)."""
    lsm = V5LSM(memtable_limit=1000, bloom_bits=bloom_bits, max_runs=4)
    for k in keys:
        lsm.put(k, k)
    lsm.flush()
    lsm.disk_reads = 0                         # start counting fresh
    start = time.perf_counter()
    for k in missing:
        lsm.get(k)
    microseconds = (time.perf_counter() - start) / len(missing) * 1e6
    return lsm.disk_reads / len(missing), microseconds


def tests():
    rng = random.Random(5)
    lsm, oracle = V5LSM(memtable_limit=100), {}        # LSM and a plain dictionary
    for _ in range(20000):                     # 20,000 random operations
        key = rng.randint(0, 3000)
        if rng.random() < 0.6:
            value = rng.randint(0, 10**6)
            lsm.put(key, value)
            oracle[key] = value
        else:
            assert lsm.get(key) == oracle.get(key)     # must match the dictionary
    print("LSM test passed: 20,000 random operations agree with a dictionary.")

    stored = rng.sample(range(2_000_000), 100_000)     # keys inside the filter
    stored_set = set(stored)
    run = SSTable([(k, k) for k in sorted(stored)], bits_per_key=10)
    other = [k for k in rng.sample(range(2_000_000), 150_000) if k not in stored_set][:100_000]
    false_alarms = sum(run.bloom.might_contain(k) for k in other)
    assert all(run.bloom.might_contain(k) for k in stored)     # never a false "no"
    print(f"Bloom filter (10 bits/key): false-positive rate = "
          f"{false_alarms / len(other):.2%}, false negatives = 0")

    keys = rng.sample(range(2_000_000), 100_000)
    key_set = set(keys)
    missing = [x for x in rng.sample(range(2_000_000), 30_000) if x not in key_set][:20_000]
    for bits in (0, 10):
        reads, us = missing_key_experiment(keys, missing, bits)
        print(f"missing key, bloom_bits={bits:>2}: {reads:.3f} disk reads per lookup")


v5_lsm_engine = V5LSM                          # name used in the assignment sheet

if __name__ == "__main__":
    tests()