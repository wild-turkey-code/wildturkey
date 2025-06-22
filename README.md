# Wild Turkey: Artifact for ICDE 2026 Submission

This repository provides the code, scripts, and configurations necessary to reproduce the experimental results of our ICDE 2026 submission titled *Wild Turkey*. The artifact has been anonymized to preserve the integrity of the double-blind review process.

## Introduction

**Wild Turkey** is a key-value store framework that rethinks the integration of learned indexes into LSM-tree systems, particularly under write-intensive workloads. While learned indexes are effective for accelerating point lookups, prior approaches overlook the compaction-heavy and write-sensitive nature of LSM-trees.

Wild Turkey introduces two key mechanisms to address this:

- **Level-Aware Compaction (LAC):** Dynamically sizes SSTables per level based on read/write patterns, reducing write amplification and compaction overhead.
- **Wild-Learning:** An RL-based mechanism that adaptively tunes the error bounds of learned indexes according to the data distribution of each SSTable.

Integrated into LevelDB and RocksDB, Wild Turkey improves write throughput by up to **1.84×**, reduces write stall time by **36%**, cuts compactions by **78.4%**, and improves read throughput by up to **1.52×**. These gains are consistent across various workloads (YCSB, SOSD) and storage types (SATA, NVMe, Optane).

Wild Turkey demonstrates that level-aware design and dynamic model tuning are essential for efficiently combining learned indexes with LSM-trees.


## Artifact for Paper #345

This repository includes:
- Source code implementing **Wild Turkey**, including Level-Aware Compaction (LAC) and Wild-Learning.
- Benchmark scripts for comparison with **LevelDB**, **WiscKey**, and **Bourbon**.
- Configuration and usage instructions to reproduce all key results.

---

## 🔧 Build Instructions

To compile the benchmark binary:

```bash
mkdir -p build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
cmake --build . -j
```

---

## ▶️ Running Benchmarks

### Option 1: Run All Tests via Script

```bash
cd scripts
sudo bash WT_test.sh
```

### Option 2: Manual Benchmark Commands

```bash
cd build

# Wild Turkey
./db_bench --benchmarks="fillrandom,readrandom,stats" --mod=10 > WildTurkey_test.log

# WiscKey
./db_bench --benchmarks="fillrandom,readrandom,stats" --mod=8 > WiscKey_test.log

# Bourbon
./db_bench --benchmarks="fillrandom,readrandom,stats" --mod=7 > Bourbon_test.log

# LevelDB (vanilla)
./db_bench --benchmarks="fillrandom,readrandom,stats" --mod=5 > LevelDB_test.log
```

> 🔸 The `--mod` flag selects the storage engine mode:
> - `10`: Wild Turkey
> - `8`: WiscKey
> - `7`: Bourbon
> - `5`: LevelDB

---

## ⏱️ Profiling Timers (Custom Stats)

The benchmark output includes internal timers for detailed profiling. Below are key timer IDs:

| Timer ID | Description                          |
|----------|--------------------------------------|
| 0        | Level read time                      |
| 1        | File open (load IndexBlock + Filter) |
| 2        | Search Index Block                   |
| 3        | Search Data Block                    |
| 5        | Load Data Block                      |
| 7        | Compaction time                      |
| 8        | Load learned model                   |
| 11       | Model training time                  |
| 12       | Value reading time                   |
| 14       | Value read from MemTable/ImmTable    |
| 15       | FilteredLookup time                  |
| 17       | Model prediction time                |
| 18       | Load chunk after prediction          |
| 19       | Locate key in predicted region       |

---

## 📎 Notes

- Benchmarks match configurations in Section V of the paper.
- Keys are 16 bytes and values are 100 bytes unless otherwise noted.
- All experiments are intended to be run on SSD-backed storage; default configurations assume SATA, NVMe, or Optane drives.
- This artifact includes no author-identifiable metadata and is fully anonymized.

---

This artifact is provided solely for the purpose of anonymous review. A public release will follow upon paper acceptance.
