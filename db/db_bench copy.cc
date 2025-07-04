// Copyright (c) 2011 The LevelDB Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file. See the AUTHORS file for names of contributors.
#include <stdlib.h>
#include <sys/types.h>
#include <mod/util.h>

#include <fstream>
#include <string>

#include "leveldb/cache.h"
#include "leveldb/db.h"
#include "leveldb/env.h"
#include "leveldb/filter_policy.h"
#include "leveldb/write_batch.h"
#include "port/port.h"
#include "util/crc32c.h"
#include "util/histogram.h"
#include "util/mutexlock.h"
#include "util/random.h"
#include "util/testutil.h"
#include <iostream>


// Comma-separated list of operations to run in the specified order
//   Actual benchmarks:
//      fillseq       -- write N values in sequential key order in async mode
//      fillrandom    -- write N values in random key order in async mode
//      overwrite     -- overwrite N values in random key order in async mode
//      fillsync      -- write N/100 values in random key order in sync mode
//      fill100K      -- write N/1000 100K values in random order in async mode
//      deleteseq     -- delete N keys in sequential order
//      deleterandom  -- delete N keys in random order
//      readseq       -- read N times sequentially
//      readreverse   -- read N times in reverse order
//      readrandom    -- read N times in random order
//      readmissing   -- read N missing keys in random order
//      readhot       -- read N times in random order from 1% section of DB
//      seekrandom    -- N random seeks
//      open          -- cost of opening a DB
//      crc32c        -- repeated crc32c of 4K of data
//   Meta operations:
//      compact     -- Compact the entire DB
//      stats       -- Print DB stats
//      sstables    -- Print sstable info
//      heapprofile -- Dump a heap profile (if supported by this port)
static const char* FLAGS_benchmarks =
    "fillseq,"
    "fillsync,"
    "fillrandom,"
    "overwrite,"
    "readrandom,"
    "readrandom,"  // Extra run to allow previous compactions to quiesce
    "readseq,"
    "99p,"
    "readreverse,"
    "compact,"
    "readrandom,"
    "readseq,"
    "readreverse,"
    "fill100K,"
    "crc32c,"
    "snappycomp,"
    "snappyuncomp,";

// Number of key/values to place in database
static int FLAGS_num = 1000000;

// Number of read operations to do.  If negative, do FLAGS_num reads.
static int FLAGS_reads = -1;

// Number of concurrent threads to run.
static int FLAGS_threads = 1;

// Size of each value
static int FLAGS_value_size = 100;

std::vector<long long> latencies; // for 99 precent latency 
std::vector<long long> latencies_w; // for 99 precent latency 
// Arrange to generate values that shrink to this fraction of
// their original size after compression
static double FLAGS_compression_ratio = 0.5;

// Print histogram of operation timings
static bool FLAGS_histogram = false;

// Number of bytes to buffer in memtable before compacting
// (initialized to default value by "main")
static int FLAGS_write_buffer_size = 0;

// Number of bytes written to each file.
// (initialized to default value by "main")
static int FLAGS_max_file_size = 0;

// Approximate size of user data packed per block (before compression.
// (initialized to default value by "main")
static int FLAGS_block_size = 0;

// Number of bytes to use as a cache of uncompressed data.
// Negative means use default settings.
static int FLAGS_cache_size = -1;

// Maximum number of files to keep open at the same time (use default if == 0)
static int FLAGS_open_files = 0;

// Bloom filter bits per key.
// Negative means use default settings.
static int FLAGS_bloom_bits = -1;

// If true, do not destroy the existing database.  If you set this
// flag and also specify a benchmark that wants a fresh database, that
// benchmark will fail.
static bool FLAGS_use_existing_db = false;

// If true, reuse existing log/MANIFEST files when re-opening a database.
static bool FLAGS_reuse_logs = false;

static int FLAGS_ycsb_uniform = 1;
// Use the db with the following name.
static const char* FLAGS_db = nullptr;

static const char* input_file = nullptr;

// Whether to use real data or not
static bool FLAGS_use_real_data = false;

// file path of real data
static const char* FLAGS_path_real_data = "invalid";

//#define __OPTIMIZE__
//#define NDEBUG


namespace leveldb {

namespace {
leveldb::Env* g_env = nullptr;

// Helper for quickly generating random data.
class RandomGenerator {
 private:
  std::string data_;
  int pos_;

 public:
  RandomGenerator() {
    // We use a limited amount of data over and over again and ensure
    // that it is larger than the compression window (32KB), and also
    // large enough to serve all typical value sizes we want to write.
    Random rnd(301);
    std::string piece;
    while (data_.size() < 1048576) {
      // Add a short fragment that is as compressible as specified
      // by FLAGS_compression_ratio.
      test::CompressibleString(&rnd, FLAGS_compression_ratio, 100, &piece);
      data_.append(piece);
    }
    pos_ = 0;
  }

  Slice Generate(size_t len) {
    if (pos_ + len > data_.size()) {
      pos_ = 0;
      assert(len < data_.size());
    }
    pos_ += len;
    return Slice(data_.data() + pos_ - len, len);
  }
};

#if defined(__linux)
static Slice TrimSpace(Slice s) {
  size_t start = 0;
  while (start < s.size() && isspace(s[start])) {
    start++;
  }
  size_t limit = s.size();
  while (limit > start && isspace(s[limit - 1])) {
    limit--;
  }
  return Slice(s.data() + start, limit - start);
}
#endif

static void AppendWithSpace(std::string* str, Slice msg) {
  if (msg.empty()) return;
  if (!str->empty()) {
    str->push_back(' ');
  }
  str->append(msg.data(), msg.size());
}

class Stats {
 private:
  double start_;
  double finish_;
  double seconds_;
  int done_;
  int next_report_;
  int64_t bytes_;
  double last_op_finish_;
  Histogram hist_;
  std::string message_;

 public:
  Stats() { Start(); }

  void Start() {
    next_report_ = 100;
    last_op_finish_ = start_;
    hist_.Clear();
    done_ = 0;
    bytes_ = 0;
    seconds_ = 0;
    start_ = g_env->NowMicros();
    finish_ = start_;
    message_.clear();
  }

  void Merge(const Stats& other) {
    hist_.Merge(other.hist_);
    done_ += other.done_;
    bytes_ += other.bytes_;
    seconds_ += other.seconds_;
    if (other.start_ < start_) start_ = other.start_;
    if (other.finish_ > finish_) finish_ = other.finish_;

    // Just keep the messages from one thread
    if (message_.empty()) message_ = other.message_;
  }

  void Stop() {
    finish_ = g_env->NowMicros();
    seconds_ = (finish_ - start_) * 1e-6;
    seconds_ = (finish_ - start_) * 1e-6;
  }

  void AddMessage(Slice msg) { AppendWithSpace(&message_, msg); }

  void FinishedSingleOp() {
    if (FLAGS_histogram) {
      double now = g_env->NowMicros();
      double micros = now - last_op_finish_;
      hist_.Add(micros);
      if (micros > 20000) {
        fprintf(stderr, "long op: %.1f micros%30s\r", micros, "");
        fflush(stderr);
      }
      last_op_finish_ = now;
    }

    done_++;
    if (done_ >= next_report_) {
      if (next_report_ < 1000)
        next_report_ += 100;
      else if (next_report_ < 5000)
        next_report_ += 500;
      else if (next_report_ < 10000)
        next_report_ += 1000;
      else if (next_report_ < 50000)
        next_report_ += 5000;
      else if (next_report_ < 100000)
        next_report_ += 10000;
      else if (next_report_ < 500000)
        next_report_ += 50000;
      else
        next_report_ += 100000;
      fprintf(stderr, "... finished %d ops%30s\r", done_, "");
      fflush(stderr);
    }
  }

  void AddBytes(int64_t n) { bytes_ += n; }

  void Report(const Slice& name) {
    // Pretend at least one op was done in case we are running a benchmark
    // that does not call FinishedSingleOp().
    if (done_ < 1) done_ = 1;

    std::string extra;
    if (bytes_ > 0) {
      // Rate is computed on actual elapsed time, not the sum of per-thread
      // elapsed times.
      double elapsed = (finish_ - start_) * 1e-6;
      char rate[100];
      snprintf(rate, sizeof(rate), "%6.1f MB/s",
               (bytes_ / 1048576.0) / elapsed);
      extra = rate;
    }
    AppendWithSpace(&extra, message_);

      fprintf(stdout, "%-12s : %11.3f micros/op;%s%s;%10.3f\n",
              name.ToString().c_str(),
              seconds_ * 1e6 / done_,
              (extra.empty() ? "" : " "),
              extra.c_str(),
              seconds_);
    if (FLAGS_histogram) {
      fprintf(stdout, "Microseconds per op:\n%s\n", hist_.ToString().c_str());
    }
    fflush(stdout);
  }
};

// State shared by all concurrent executions of the same benchmark.
struct SharedState {
  port::Mutex mu;
  port::CondVar cv GUARDED_BY(mu);
  int total GUARDED_BY(mu);

  // Each thread goes through the following states:
  //    (1) initializing
  //    (2) waiting for others to be initialized
  //    (3) running
  //    (4) done

  int num_initialized GUARDED_BY(mu);
  int num_done GUARDED_BY(mu);
  bool start GUARDED_BY(mu);

  SharedState(int total)
      : cv(&mu), total(total), num_initialized(0), num_done(0), start(false) {}
};

// Per-thread state for concurrent executions of the same benchmark.
struct ThreadState {
  int tid;      // 0..n-1 when running in n threads
  Random rand;  // Has different seeds for different threads
  Stats stats;
  SharedState* shared;

  ThreadState(int index) : tid(index), rand(1000 + index), shared(nullptr) {}
};

}  // namespace

class Benchmark {
 private:
  Cache* cache_;
  const FilterPolicy* filter_policy_;
  DB* db_;
  int num_;
  
  std::string input_file;
  std::ifstream input;
  using key_type = uint64_t;
  std::vector<key_type> data;
  std::vector<string> data_ycsb;

  int value_size_;
  int entries_per_batch_;
  WriteOptions write_options_;
  int reads_;
  int heap_counter_;

  void PrintHeader() {
    const int kKeySize = 16;
    PrintEnvironment();
    fprintf(stdout, "Keys:       %d bytes each\n", kKeySize);
    fprintf(stdout, "Values:     %d bytes each (%d bytes after compression)\n",
            FLAGS_value_size,
            static_cast<int>(FLAGS_value_size * FLAGS_compression_ratio + 0.5));
    fprintf(stdout, "Entries:    %d\n", num_);
    fprintf(stdout, "RawSize:    %.1f MB (estimated)\n",
            ((static_cast<int64_t>(kKeySize + FLAGS_value_size) * num_) /
             1048576.0));
    fprintf(stdout, "FileSize:   %.1f MB (estimated)\n",
            (((kKeySize + FLAGS_value_size * FLAGS_compression_ratio) * num_) /
             1048576.0));
    PrintWarnings();
    fprintf(stdout, "------------------------------------------------\n");
  }

  void PrintWarnings() {
#if defined(__GNUC__) && !defined(__OPTIMIZE__)
    fprintf(
        stdout,
        "WARNING: Optimization is disabled: benchmarks unnecessarily slow\n");
#endif
#ifndef NDEBUG
    fprintf(stdout,
            "WARNING: Assertions are enabled; benchmarks unnecessarily slow\n");
#endif

    // See if snappy is working by attempting to compress a compressible string
    const char text[] = "yyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyy";
    std::string compressed;
    if (!port::Snappy_Compress(text, sizeof(text), &compressed)) {
      fprintf(stdout, "WARNING: Snappy compression is not enabled\n");
    } else if (compressed.size() >= sizeof(text)) {
      fprintf(stdout, "WARNING: Snappy compression is not effective\n");
    }
  }

  void PrintEnvironment() {
    fprintf(stderr, "LevelDB:    version %d.%d\n", kMajorVersion,
            kMinorVersion);

#if defined(__linux)
    time_t now = time(nullptr);
    fprintf(stderr, "Date:       %s", ctime(&now));  // ctime() adds newline

    FILE* cpuinfo = fopen("/proc/cpuinfo", "r");
    if (cpuinfo != nullptr) {
      char line[1000];
      int num_cpus = 0;
      std::string cpu_type;
      std::string cache_size;
      while (fgets(line, sizeof(line), cpuinfo) != nullptr) {
        const char* sep = strchr(line, ':');
        if (sep == nullptr) {
          continue;
        }
        Slice key = TrimSpace(Slice(line, sep - 1 - line));
        Slice val = TrimSpace(Slice(sep + 1));
        if (key == "model name") {
          ++num_cpus;
          cpu_type = val.ToString();
        } else if (key == "cache size") {
          cache_size = val.ToString();
        }
      }
      fclose(cpuinfo);
      fprintf(stderr, "CPU:        %d * %s\n", num_cpus, cpu_type.c_str());
      fprintf(stderr, "CPUCache:   %s\n", cache_size.c_str());
    }
#endif
  }

 public:
  Benchmark()
      : cache_(FLAGS_cache_size >= 0 ? NewLRUCache(FLAGS_cache_size) : nullptr),
        filter_policy_(FLAGS_bloom_bits >= 0
                           ? NewBloomFilterPolicy(FLAGS_bloom_bits)
                           : nullptr),
        db_(nullptr),
        num_(FLAGS_num),
        value_size_(FLAGS_value_size),
        entries_per_batch_(1),
        reads_(FLAGS_reads < 0 ? FLAGS_num : FLAGS_reads),
        heap_counter_(0) {
    std::vector<std::string> files;
    g_env->GetChildren(FLAGS_db, &files);
    for (size_t i = 0; i < files.size(); i++) {
      if (Slice(files[i]).starts_with("heap-")) {
        g_env->DeleteFile(std::string(FLAGS_db) + "/" + files[i]);
      }
    }
    if (!FLAGS_use_existing_db) {
      DestroyDB(FLAGS_db, Options());
    }
  }

  ~Benchmark() {
    delete db_;
    delete cache_;
    delete filter_policy_;
  }

  void Run() {
    PrintHeader();
    Open();

    const char* benchmarks = FLAGS_benchmarks;
    while (benchmarks != nullptr) {
      const char* sep = strchr(benchmarks, ',');
      Slice name;
      if (sep == nullptr) {
        name = benchmarks;
        benchmarks = nullptr;
      } else {
        name = Slice(benchmarks, sep - benchmarks);
        benchmarks = sep + 1;
      }

      // Reset parameters that may be overridden below
      num_ = FLAGS_num;
      reads_ = (FLAGS_reads < 0 ? FLAGS_num : FLAGS_reads);
      value_size_ = FLAGS_value_size;
      entries_per_batch_ = 1;
      write_options_ = WriteOptions();

      void (Benchmark::*method)(ThreadState*) = nullptr;
      bool fresh_db = false;
      int num_threads = FLAGS_threads;

      if (name == Slice("open")) {
        method = &Benchmark::OpenBench;
        num_ /= 10000;
        if (num_ < 1) num_ = 1;
      } else if (name == Slice("fillseq")) {
        fresh_db = true;
        method = &Benchmark::WriteSeq;
      } else if (name == Slice("uni40")) {
      uint64_t key;
      fresh_db = true;
      input_file = "/mnt/datasets/uniform/keys_40m.txt";
      std::ifstream input; 
      input.open(input_file); 
      if (!input.is_open()) {
          std::cerr << "Error opening file" << std::endl;
          exit(1);
      }
      std::string line;
      while (std::getline(input, line)) {
          key = std::stoull(line);
          data.push_back(key);
      }

      input.close();
      method = &Benchmark::unirandom;
      } else if (name == Slice("uniread")) {
      
      method = &Benchmark::unirandom_read;

      }else if (name == Slice("uniwrite")) {
      
      method = &Benchmark::unirandom;

      } else if (name == Slice("osm_w")) {
      uint64_t key;
      fresh_db = true;
  
      input_file = "/mnt/datasets/data_set/data/osm_cellids_200M_uint64";
      std::ifstream input; 
      input.open(input_file, std::ios::binary); 
      if (!input.is_open()) {
          std::cerr << "Error opening file" << std::endl;
          exit(1);
      }

      while (input.read(reinterpret_cast<char*>(&key), sizeof(uint64_t))) {

        data.push_back(key);
      }

      input.close();
      std::random_shuffle(data.begin(), data.end());
      method = &Benchmark::real_workload_w;

      } else if (name == Slice("book_w")) {
      uint64_t key;
      fresh_db = true;
      input_file = "/mnt/datasets/data_set/data/books_200M_uint64";
      std::ifstream input; 
      input.open(input_file, std::ios::binary); 
      if (!input.is_open()) {
          std::cerr << "Error opening file" << std::endl;
          exit(1);
      }
      while (input.read(reinterpret_cast<char*>(&key), sizeof(uint64_t))) {
        data.push_back(key);
      }
      input.close();
      // std::random_shuffle(data.begin(), data.end());
      method = &Benchmark::real_workload_w;

      }else if (name == Slice("fb_w")) {
      uint64_t key;
      fresh_db = true;
      input_file = "/mnt/datasets/data_set/data/fb_200M_uint64";
      std::ifstream input; 
      input.open(input_file, std::ios::binary); 
      if (!input.is_open()) {
          std::cerr << "Error opening file" << std::endl;
          exit(1);
      }
      while (input.read(reinterpret_cast<char*>(&key), sizeof(uint64_t))) {
        data.push_back(key);
      }
      input.close();
      // std::random_shuffle(data.begin(), data.end());
      method = &Benchmark::real_workload_w;

      }
       else if (name == Slice("wiki_w")) {
      uint64_t key;
      fresh_db = true;
      input_file = "/mnt/datasets/data_set/data/wiki_ts_200M_uint64";
      std::ifstream input; 
      input.open(input_file, std::ios::binary); 
      if (!input.is_open()) {
          std::cerr << "Error opening file" << std::endl;
          exit(1);
      }
      while (input.read(reinterpret_cast<char*>(&key), sizeof(uint64_t))) {
        data.push_back(key);
      }
      input.close();
      // std::random_shuffle(data.begin(), data.end());
      method = &Benchmark::real_workload_w;

      } else if (name == Slice("osm_blanced")) {
       uint64_t key;
      fresh_db = true;
  
      input_file = "/mnt/datasets/data_set/data/osm_cellids_200M_uint64";
      std::ifstream input; 
      input.open(input_file, std::ios::binary); 
      if (!input.is_open()) {
          std::cerr << "Error opening file" << std::endl;
          exit(1);
      }

      while (input.read(reinterpret_cast<char*>(&key), sizeof(uint64_t))) {

        data.push_back(key);
      }

      input.close();
      std::random_shuffle(data.begin(), data.end());
      method = &Benchmark::real_blanced;
      } else if (name == Slice("book_blanced")) {
              uint64_t key;
      fresh_db = true;
      input_file = "/mnt/datasets/data_set/data/books_200M_uint64";
      std::ifstream input; 
      input.open(input_file, std::ios::binary); 
      if (!input.is_open()) {
          std::cerr << "Error opening file" << std::endl;
          exit(1);
      }
      while (input.read(reinterpret_cast<char*>(&key), sizeof(uint64_t))) {
        data.push_back(key);
      }
      input.close();
      std::random_shuffle(data.begin(), data.end());
      method = &Benchmark::real_blanced;
      }else if (name == Slice("fb_blanced")) {
      uint64_t key;
      fresh_db = true;
      input_file = "/mnt/datasets/data_set/data/fb_200M_uint64";
      std::ifstream input; 
      input.open(input_file, std::ios::binary); 
      if (!input.is_open()) {
          std::cerr << "Error opening file" << std::endl;
          exit(1);
      }
      while (input.read(reinterpret_cast<char*>(&key), sizeof(uint64_t))) {
        data.push_back(key);
      }
      input.close();
      std::random_shuffle(data.begin(), data.end());
      method = &Benchmark::real_blanced;
      }else if (name == Slice("wiki_blanced")) {
         uint64_t key;
      fresh_db = true;
      input_file = "/mnt/datasets/data_set/data/wiki_ts_200M_uint64";
      std::ifstream input; 
      input.open(input_file, std::ios::binary); 
      if (!input.is_open()) {
          std::cerr << "Error opening file" << std::endl;
          exit(1);
      }
      while (input.read(reinterpret_cast<char*>(&key), sizeof(uint64_t))) {
        data.push_back(key);
      }
      input.close();
      std::random_shuffle(data.begin(), data.end());
      method = &Benchmark::real_blanced;
      }else if (name == Slice("osm_readheavy")) {
      method = &Benchmark::real_readheavy;
      } else if (name == Slice("book_readheavy")) {
      method = &Benchmark::real_readheavy;
      }else if (name == Slice("fb_readheavy")) {
      method = &Benchmark::real_readheavy;
      }else if (name == Slice("wiki_readheavy")) {
      method = &Benchmark::real_readheavy;
      }
      else if (name == Slice("real_r")) {
   
      method = &Benchmark::real_workload_r;

      }else if (name == Slice("ycsba")) {
        uint64_t key;
        fresh_db = true;
    
        input_file = "/mnt/datasets/data_set/data/osm_cellids_200M_uint64";
        std::ifstream input; 
        input.open(input_file, std::ios::binary); 
        if (!input.is_open()) {
            std::cerr << "Error opening file" << std::endl;
            exit(1);
        }
  
        while (input.read(reinterpret_cast<char*>(&key), sizeof(uint64_t))) {
  
          data.push_back(key);
        }
  
        input.close();
        std::random_shuffle(data.begin(), data.end());
  
      method = &Benchmark::YCSBA;

      }else if (name == Slice("ycsbb")) {
      
      method = &Benchmark::YCSBB;

      }else if (name == Slice("ycsbc")) {
      
      method = &Benchmark::YCSBC;

      }else if (name == Slice("ycsbd")) {
      
      method = &Benchmark::YCSBD;

      }else if (name == Slice("ycsbe")) {
      
      method = &Benchmark::YCSBE;

      }else if (name == Slice("ycsbf")) {
      
      method = &Benchmark::YCSBF;

      }else if (name == Slice("fillbatch")) {
        fresh_db = true;
        entries_per_batch_ = 1000;
        method = &Benchmark::WriteSeq;
      } else if (name == Slice("fillrandom")) {
        fresh_db = true;
        method = &Benchmark::WriteRandom;
      }  else if (name == Slice("zipwrite")) {
        fresh_db = true;
        method = &Benchmark::zipfian;
      }else if (name == Slice("overwrite")) {
        fresh_db = false;
        method = &Benchmark::WriteRandom;
      } else if (name == Slice("fillsync")) {
        fresh_db = true;
        num_ /= 1000;
        write_options_.sync = true;
        method = &Benchmark::WriteRandom;
      } else if (name == Slice("fill100K")) {
        fresh_db = true;
        num_ /= 1000;
        value_size_ = 100 * 1000;
        method = &Benchmark::WriteRandom;
      } else if (name == Slice("readseq")) {
        method = &Benchmark::ReadSequential;
      } else if (name == Slice("readreverse")) {
        method = &Benchmark::ReadReverse;
      } else if (name == Slice("readrandom")) {
        method = &Benchmark::ReadRandom;
      }else if (name == Slice("99p")) {
        method = &Benchmark::Get99PercentileLatency;
      }  else if (name == Slice("zipread")) {
        method = &Benchmark::zipfianread;
      }else if (name == Slice("readmissing")) {
        method = &Benchmark::ReadMissing;
      } else if (name == Slice("seekrandom")) {
        method = &Benchmark::SeekRandom;
      } else if (name == Slice("readhot")) {
        method = &Benchmark::ReadHot;
      } else if (name == Slice("readrandomsmall")) {
        reads_ /= 1000;
        method = &Benchmark::ReadRandom;
      } else if (name == Slice("deleteseq")) {
        method = &Benchmark::DeleteSeq;
      } else if (name == Slice("deleterandom")) {
        method = &Benchmark::DeleteRandom;
      } else if (name == Slice("readwhilewriting")) {
        num_threads++;  // Add extra thread for writing
        method = &Benchmark::ReadWhileWriting;
      } else if (name == Slice("compact")) {
        method = &Benchmark::Compact;
      } else if (name == Slice("crc32c")) {
        method = &Benchmark::Crc32c;
      } else if (name == Slice("snappycomp")) {
        method = &Benchmark::SnappyCompress;
      } else if (name == Slice("snappyuncomp")) {
        method = &Benchmark::SnappyUncompress;
      } else if (name == Slice("heapprofile")) {
        HeapProfile();
      } else if (name == Slice("stats")) {
        PrintStats("leveldb.stats");
      } else if (name == Slice("sstables")) {
        PrintStats("leveldb.sstables");
      } else {
        if (!name.empty()) {  // No error message for empty name
          fprintf(stderr, "unknown benchmark '%s'\n", name.ToString().c_str());
        }
      }

      if (fresh_db) {
        if (FLAGS_use_existing_db) {
          fprintf(stdout, "%-12s : skipped (--use_existing_db is true)\n",
                  name.ToString().c_str());
          method = nullptr;
        } else {
          delete db_;
          db_ = nullptr;
          DestroyDB(FLAGS_db, Options());
          Open();
        }
      }

      if (method != nullptr) {
        RunBenchmark(num_threads, name, method);
      }
    }
  }

 private:
  struct ThreadArg {
    Benchmark* bm;
    SharedState* shared;
    ThreadState* thread;
    void (Benchmark::*method)(ThreadState*);
  };

  static void ThreadBody(void* v) {
    ThreadArg* arg = reinterpret_cast<ThreadArg*>(v);
    SharedState* shared = arg->shared;
    ThreadState* thread = arg->thread;
    {
      MutexLock l(&shared->mu);
      shared->num_initialized++;
      if (shared->num_initialized >= shared->total) {
        shared->cv.SignalAll();
      }
      while (!shared->start) {
        shared->cv.Wait();
      }
    }

    thread->stats.Start();
    (arg->bm->*(arg->method))(thread);
    thread->stats.Stop();

    {
      MutexLock l(&shared->mu);
      shared->num_done++;
      if (shared->num_done >= shared->total) {
        shared->cv.SignalAll();
      }
    }
  }

  void RunBenchmark(int n, Slice name,
                    void (Benchmark::*method)(ThreadState*)) {
    SharedState shared(n);

    ThreadArg* arg = new ThreadArg[n];
    for (int i = 0; i < n; i++) {
      arg[i].bm = this;
      arg[i].method = method;
      arg[i].shared = &shared;
      arg[i].thread = new ThreadState(i);
      arg[i].thread->shared = &shared;
      g_env->StartThread(ThreadBody, &arg[i]);
    }

    shared.mu.Lock();
    while (shared.num_initialized < n) {
      shared.cv.Wait();
    }

    shared.start = true;
    shared.cv.SignalAll();
    while (shared.num_done < n) {
      shared.cv.Wait();
    }
    shared.mu.Unlock();

    for (int i = 1; i < n; i++) {
      arg[0].thread->stats.Merge(arg[i].thread->stats);
    }
    arg[0].thread->stats.Report(name);

    for (int i = 0; i < n; i++) {
      delete arg[i].thread;
    }
    delete[] arg;
  }

  void Crc32c(ThreadState* thread) {
    // Checksum about 500MB of data total
    const int size = 4096;
    const char* label = "(4K per op)";
    std::string data(size, 'x');
    int64_t bytes = 0;
    uint32_t crc = 0;
    while (bytes < 500 * 1048576) {
      crc = crc32c::Value(data.data(), size);
      thread->stats.FinishedSingleOp();
      bytes += size;
    }
    // Print so result is not dead
    fprintf(stderr, "... crc=0x%x\r", static_cast<unsigned int>(crc));

    thread->stats.AddBytes(bytes);
    thread->stats.AddMessage(label);
  }

  void SnappyCompress(ThreadState* thread) {
    RandomGenerator gen;
    Slice input = gen.Generate(Options().block_size);
    int64_t bytes = 0;
    int64_t produced = 0;
    bool ok = true;
    std::string compressed;
    while (ok && bytes < 1024 * 1048576) {  // Compress 1G
      ok = port::Snappy_Compress(input.data(), input.size(), &compressed);
      produced += compressed.size();
      bytes += input.size();
      thread->stats.FinishedSingleOp();
    }

    if (!ok) {
      thread->stats.AddMessage("(snappy failure)");
    } else {
      char buf[100];
      snprintf(buf, sizeof(buf), "(output: %.1f%%)",
               (produced * 100.0) / bytes);
      thread->stats.AddMessage(buf);
      thread->stats.AddBytes(bytes);
    }
  }

  void SnappyUncompress(ThreadState* thread) {
    RandomGenerator gen;
    Slice input = gen.Generate(Options().block_size);
    std::string compressed;
    bool ok = port::Snappy_Compress(input.data(), input.size(), &compressed);
    int64_t bytes = 0;
    char* uncompressed = new char[input.size()];
    while (ok && bytes < 1024 * 1048576) {  // Compress 1G
      ok = port::Snappy_Uncompress(compressed.data(), compressed.size(),
                                   uncompressed);
      bytes += input.size();
      thread->stats.FinishedSingleOp();
    }
    delete[] uncompressed;

    if (!ok) {
      thread->stats.AddMessage("(snappy failure)");
    } else {
      thread->stats.AddBytes(bytes);
    }
  }

  void Open() {
    assert(db_ == nullptr);
    Options options;
    options.env = g_env;
    options.create_if_missing = !FLAGS_use_existing_db;
    options.block_cache = cache_;
    options.write_buffer_size = FLAGS_write_buffer_size;
    options.max_file_size = FLAGS_max_file_size;
    options.block_size = FLAGS_block_size;
    options.max_open_files = FLAGS_open_files;
    options.filter_policy = filter_policy_;
    options.reuse_logs = FLAGS_reuse_logs;
    Status s = DB::Open(options, FLAGS_db, &db_);
    if (!s.ok()) {
      fprintf(stderr, "open error: %s\n", s.ToString().c_str());
      exit(1);
    }
  }

  void OpenBench(ThreadState* thread) {
    for (int i = 0; i < num_; i++) {
      delete db_;
      Open();
      thread->stats.FinishedSingleOp();
    }
  }

  void WriteSeq(ThreadState* thread) { DoWrite(thread, true); }

  void WriteRandom(ThreadState* thread) { DoWrite(thread, false); }

  
  void zipfian(ThreadState* thread) { // here
    RandomGenerator gen;
    WriteBatch batch;
    Status s;
    int64_t bytes = 0;
    // std::ofstream output_file("/home/eros/workspace-lsm/wildturkey/db/k_values.txt");
    if (num_ != FLAGS_num) {
      char msg[100];
      snprintf(msg, sizeof(msg), "(%d ops)", num_);
      thread->stats.AddMessage(msg);
    }

      for (int i = 0; i < num_; i += entries_per_batch_) {
        batch.Clear();
        for (int j = 0; j < entries_per_batch_; j++) {
      
          const int k =  thread->rand.Zipfian(FLAGS_num, 1);
          // printf("key: %d\n", k);
          // output_file << k << std::endl;

          char key[100];
          snprintf(key, sizeof(key), "%016d", k);
          db_->Put(write_options_, key, gen.Generate(value_size_));
          bytes += value_size_ + strlen(key);
          thread->stats.FinishedSingleOp();
        }
      }
      thread->stats.AddBytes(bytes);
      
      // static_cast<leveldb::DBImpl*>(db_)->CompactOrderdRange(nullptr, nullptr, 0);
    }
    
  void DoWrite(ThreadState* thread, bool seq) { // here
    RandomGenerator gen;
    WriteBatch batch;
    Status s;
    int64_t bytes = 0;
    // std::ofstream output_file("/mnt/datasets/uniform/keys_64m.txt");
    if (num_ != FLAGS_num) {
      char msg[100];
      snprintf(msg, sizeof(msg), "(%d ops)", num_);
      thread->stats.AddMessage(msg);
    }
      for (int i = 0; i < num_; i += entries_per_batch_) {
        batch.Clear();
        for (int j = 0; j < entries_per_batch_; j++) {
          auto start = std::chrono::high_resolution_clock::now(); // start time

          // const int k = seq ? i + j : (thread->rand.Next() % FLAGS_num);
            int k = 0;
            if (FLAGS_ycsb_uniform==1){
            k = seq ? i + j : (thread->rand.Next() % FLAGS_num);
            //  k = thread->rand.Uniform(FLAGS_num);
            }else{
            k = seq ? i + j :  thread->rand.Zipfian(FLAGS_num, 1.0);
            }

          // output_file << k << std::endl;

          char key[100];
          snprintf(key, sizeof(key), "%016d", k);
          db_->Put(write_options_, key, gen.Generate(value_size_));
          bytes += value_size_ + strlen(key);

        auto end = std::chrono::high_resolution_clock::now(); // end time
        auto latency = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count(); // latency in nanoseconds
        latencies_w.push_back(latency); // store latency in vector

        thread->stats.FinishedSingleOp();
        }
      }
      
      thread->stats.AddBytes(bytes);

      // if (adgMod::MOD==10 or adgMod::sst_size>=1 ){
      // static_cast<leveldb::DBImpl*>(db_)->CompactOrderdRange(nullptr, nullptr, 0);
      // }
    //  output_file.close();
    static_cast<leveldb::DBImpl*>(db_)->WaitForBackground();

      
      
    }



  void real_blanced(ThreadState* thread) { // here
    RandomGenerator gen;
    //  string the_key;
    ReadOptions options;
    WriteBatch batch;
    Status s;
    string the_key;
    std::string value;
    int64_t bytes = 0;
    int64_t found = 0;
    int64_t reads_done = 0;
    int64_t not_found = 0;
    int64_t writes_done = 0;
    std::vector<long long> latenciesre; 


    char key[100];
    if (num_ != FLAGS_num) {
      char msg[100];
      snprintf(msg, sizeof(msg), "(%d ops)", num_);
      thread->stats.AddMessage(msg);
    }
    
    for (int i = 0; i < num_; i++) {
          
      the_key = adgMod::generate_key(std::to_string(data[i]));

      int next_op = thread->rand.Next() % 100;
      
      if (next_op < 50){

        auto start = std::chrono::high_resolution_clock::now();
         if (db_->Get(options, the_key, &value).ok()) {
          found++;
          bytes += value.size() + strlen(key);
         
        }else{
          not_found++;
        }
        
        auto end = std::chrono::high_resolution_clock::now(); // end time
        auto latency = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count(); // latency in nanoseconds
        latenciesre.push_back(latency); // store latency in vector
          thread->stats.FinishedSingleOp();
          reads_done++;
      }else{
        auto start = std::chrono::high_resolution_clock::now();
          db_->Put(write_options_,  the_key, gen.Generate(value_size_));
          bytes += value_size_ + strlen(key);
          writes_done++;

        auto end = std::chrono::high_resolution_clock::now(); // end time
        auto latency = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count(); // latency in nanoseconds
        latenciesre.push_back(latency); // store latency in vector
        thread->stats.FinishedSingleOp();
      }

    
    }

    thread->stats.AddBytes(bytes);
    char msg[100];
        snprintf(msg, sizeof(msg), "( reads:%d"  " writes:%d" \
             " total:%d"  " found:%d"  " not found:%d"  ")",
             reads_done, writes_done, num_, found , not_found);
    thread->stats.AddMessage(msg);   

  std::sort(latenciesre.begin(), latenciesre.end());
  int index = 0.99 * latenciesre.size();

  // thread->stats.AddMessage("\nRead 99 percentile latency: " + std::to_string(latencies[index]) + " ns\n");
  printf("\nblanced 99 percentile latency: %ld ns\n", latenciesre[index]);
 
  // thread->stats.AddMessage("Write 99 percentile latency: " + std::to_string(latencies_w[index_w]) + " ns\n");
  

    
  }

  void real_readheavy(ThreadState* thread) { // here
    RandomGenerator gen;
    //  string the_key;
    ReadOptions options;
    WriteBatch batch;
    Status s;
    string the_key;
    std::string value;
    int64_t bytes = 0;
    int64_t found = 0;
    int64_t reads_done = 0;
    int64_t not_found = 0;
    int64_t writes_done = 0;
    std::vector<long long> latencies1; 

    char key[100];
    if (num_ != FLAGS_num) {
      char msg[100];
      snprintf(msg, sizeof(msg), "(%d ops)", num_);
      thread->stats.AddMessage(msg);
    }
    


    for (int i = 0; i < num_; i++) {
          
      the_key = adgMod::generate_key(std::to_string(data[i]));

      int next_op = thread->rand.Next() % 100;
      
      if (next_op < 95){

        auto start = std::chrono::high_resolution_clock::now();
         if (db_->Get(options, the_key, &value).ok()) {
          found++;
          bytes += value.size() + strlen(key);
         
        }else{
          not_found++;
        }
        
        auto end = std::chrono::high_resolution_clock::now(); // end time
        auto latency = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count(); // latency in nanoseconds
        latencies1.push_back(latency); // store latency in vector
          thread->stats.FinishedSingleOp();
          reads_done++;
      }else{
        auto start = std::chrono::high_resolution_clock::now();
          db_->Put(write_options_,  the_key, gen.Generate(value_size_));
          bytes += value_size_ + strlen(key);
          writes_done++;

        auto end = std::chrono::high_resolution_clock::now(); // end time
        auto latency = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count(); // latency in nanoseconds
        latencies1.push_back(latency); // store latency in vector
        thread->stats.FinishedSingleOp();
      }

    
    }

    thread->stats.AddBytes(bytes);
    char msg[100];
        snprintf(msg, sizeof(msg), "( reads:%d"  " writes:%d" \
             " total:%d"  " found:%d"  " not found:%d"  ")",
             reads_done, writes_done, num_, found , not_found);
    thread->stats.AddMessage(msg);   

  std::sort(latencies1.begin(), latencies1.end());
  int index = 0.99 * latencies1.size();

  // thread->stats.AddMessage("\nRead 99 percentile latency: " + std::to_string(latencies[index]) + " ns\n");
  printf("\nblanced 99 percentile latency: %ld ns\n", latencies1[index]);
 
  // thread->stats.AddMessage("Write 99 percentile latency: " + std::to_string(latencies_w[index_w]) + " ns\n");
  

    
  }

  void YCSBA(ThreadState* thread) { // here


    
    RandomGenerator gen;
    ReadOptions options;
    WriteBatch batch;
    Status s;
    string the_key;
    std::string value;
    int64_t bytes = 0;
    int64_t found = 0;
    int64_t reads_done = 0;
    int64_t not_found = 0;
    int64_t writes_done = 0;
    int kkkk=0;
    batch.Clear();
    char key[100];

    int k2=0;
    for (int i = 0; i < num_; i++) {
      

      // data set
      // if (FLAGS_ycsb_uniform==1){
      // k2 = thread->rand.Next() % FLAGS_num;
      // }else{
      // k2 =  thread->rand.Zipfian(FLAGS_num, 1.0);
      // }
      



      // snprintf(key, sizeof(key), "%016d", k2);

      the_key = adgMod::generate_key(std::to_string(data[i]));

      int next_op = thread->rand.Next() % 100;
      
      if (next_op < 50){
       
         if (db_->Get(options, key, &value).ok()) {
          found++;
          bytes += value.size() + strlen(key);
         
        }else{
          not_found++;
        }
        
          thread->stats.FinishedSingleOp();
          reads_done++;
      }else{
     
          db_->Put(write_options_,  key, gen.Generate(value_size_));
          bytes += value_size_ + strlen(key);
          writes_done++;

        thread->stats.FinishedSingleOp();
      }

    
    }

    thread->stats.AddBytes(bytes);
    char msg[100];
        snprintf(msg, sizeof(msg), "( reads:%d"  " writes:%d" \
             " total:%d"  " found:%d"  " not found:%d"  ")",
             reads_done, writes_done, num_, found , not_found);
    thread->stats.AddMessage(msg);   

    
  }

 void YCSBB(ThreadState* thread) { // here
    RandomGenerator gen;
    ReadOptions options;
    WriteBatch batch;
    Status s;
    std::string value;
    int64_t bytes = 0;
    int64_t found = 0;
    int64_t reads_done = 0;
    int64_t not_found = 0;
    int64_t writes_done = 0;
    int k;
    batch.Clear();
    char key[100];
    // num_=10000000;
    // if (num_ != FLAGS_num) {
    //   char msg[100];
    //   snprintf(msg, sizeof(msg), "(%d ops)", num_);
    //   thread->stats.AddMessage(msg);
    // }



    // int write=0;
    // for (int i = 0; i < num_; i++) {
    //   if (FLAGS_ycsb_uniform==1){
    //    write= thread->rand.Next() % FLAGS_num;
    //   }else{
    //     write =  thread->rand.Zipfian(FLAGS_num, 1.0);
    //   }
    //   snprintf(key, sizeof(key), "%016d", write);
    //   db_->Put(write_options_, key, gen.Generate(value_size_));
    // }

    int k2=0;
    for (int i = 0; i < num_; i++) {
          
      if (FLAGS_ycsb_uniform==1){
      k2 = thread->rand.Next() % FLAGS_num;
      }else{
      k2 =  thread->rand.Zipfian(FLAGS_num, 1.0);
      }
      snprintf(key, sizeof(key), "%016d", k2);



      int next_op = thread->rand.Next() % 100;
      
      if (next_op < 95){

        
         if (db_->Get(options, key, &value).ok()) {
          found++;
          bytes += value.size() + strlen(key);
         
        }else{
          not_found++;
        }
      
          reads_done++;
      }else{
       
          db_->Get(options, key, &value);
          db_->Put(write_options_, key, gen.Generate(value_size_));
          bytes += value_size_ + strlen(key);
          writes_done++;
      
          thread->stats.FinishedSingleOp();

      }

    
    }
    thread->stats.AddBytes(bytes);
    char msg[100];
        snprintf(msg, sizeof(msg), "( reads:%d"  " writes:%d" \
             " total:%d"  " found:%d"  " not found:%d"  ")",
             reads_done, writes_done, num_, found , not_found);
    thread->stats.AddMessage(msg);   

    
  }

   void YCSBC(ThreadState* thread) { // here
    RandomGenerator gen;
    ReadOptions options;
    WriteBatch batch;
    Status s;
    std::string value;
    int64_t bytes = 0;
    int64_t found = 0;
    int64_t reads_done = 0;
    int64_t not_found = 0;
    int64_t writes_done = 0;
    int k;
    batch.Clear();
    char key[100];
    num_=10000000;
    // if (num_ != FLAGS_num) {
    //   char msg[100];
    //   snprintf(msg, sizeof(msg), "(%d ops)", num_);
    //   thread->stats.AddMessage(msg);
    // }


    // int write=0;
    // for (int i = 0; i < num_; i++) {
    //   if (FLAGS_ycsb_uniform==1){
    //   write = thread->rand.Next() % FLAGS_num;
    //   }else{
    //     write =  thread->rand.Zipfian(FLAGS_num, 1.0);
    //   }
    //   snprintf(key, sizeof(key), "%016d", write);
    //   db_->Put(write_options_, key, gen.Generate(value_size_));
    //   thread->stats.FinishedSingleOp();
    // }

    int k2=0;
    for (int i = 0; i < num_; i++) {
          
      if (FLAGS_ycsb_uniform==1){
       k2 = thread->rand.Next() % FLAGS_num;
      }else{
        k2 = thread->rand.Zipfian(FLAGS_num, 1.0);
      }
      snprintf(key, sizeof(key), "%016d", k2);

      if (db_->Get(options, key, &value).ok()) {
          found++;
          bytes += value.size() + strlen(key);
          
        }else{
          not_found++;
        }
        thread->stats.FinishedSingleOp();
        reads_done++;

    }
    thread->stats.AddBytes(bytes);
    char msg[100];
        snprintf(msg, sizeof(msg), "( reads:%d"  " writes:%d" \
             " total:%d"  " found:%d"  " not found:%d"  ")",
             reads_done, writes_done, num_, found , not_found);
    thread->stats.AddMessage(msg);   

    
  }
   void YCSBD(ThreadState* thread) { // here
    RandomGenerator gen;
    ReadOptions options;
    WriteBatch batch;
    Status s;
    std::string value;
    int64_t bytes = 0;
    int64_t found = 0;
    int64_t reads_done = 0;
    int64_t not_found = 0;
    int64_t writes_done = 0;
    int k;
    batch.Clear();
    char key[100];
    num_=10000000;
    // if (num_ != FLAGS_num) {
    //   char msg[100];
    //   snprintf(msg, sizeof(msg), "(%d ops)", num_);
    //   thread->stats.AddMessage(msg);
    // }




    // int write=0;
    // for (int i = 0; i < num_; i++) {
    //   if (FLAGS_ycsb_uniform==1){
    //    write= thread->rand.Next() % FLAGS_num;
    //   }else{
    //     write =  thread->rand.Zipfian(FLAGS_num, 1.0);
    //   }
    //   snprintf(key, sizeof(key), "%016d", write);
    //   db_->Put(write_options_, key, gen.Generate(value_size_));
    // }

    int k2=0;
    for (int i = 0; i < num_; i++) {
          
      if (FLAGS_ycsb_uniform==1){
      k2 = thread->rand.Next() % FLAGS_num;
      }else{
      k2 =  thread->rand.Zipfian(FLAGS_num, 1.0);
      }
      snprintf(key, sizeof(key), "%016d", k2);



      int next_op = thread->rand.Next() % 100;
      
      if (next_op < 95){

         if (db_->Get(options, key, &value).ok()) {
          found++;
          bytes += value.size() + strlen(key);
          thread->stats.FinishedSingleOp();
        }else{
          not_found++;
        }
      
          reads_done++;
      }else{
          db_->Put(write_options_, key, gen.Generate(value_size_));
          bytes += value_size_ + strlen(key);
          writes_done++;
          thread->stats.FinishedSingleOp();

      }

    
    }
    thread->stats.AddBytes(bytes);
    char msg[100];
        snprintf(msg, sizeof(msg), "( reads:%d"  " writes:%d" \
             " total:%d"  " found:%d"  " not found:%d"  ")",
             reads_done, writes_done, num_, found , not_found);
    thread->stats.AddMessage(msg);   

    
  }
   void YCSBE(ThreadState* thread) { // here
    RandomGenerator gen;
    ReadOptions options;
    WriteBatch batch;
    Status s;
    std::string value;
    int64_t bytes = 0;
    int64_t found = 0;
    int64_t reads_done = 0;
    int64_t not_found = 0;
    int64_t writes_done = 0;
    int k;
    batch.Clear();
    char key[100];
    num_=10000000;
    // if (num_ != FLAGS_num) {
    //   char msg[100];
    //   snprintf(msg, sizeof(msg), "(%d ops)", num_);
    //   thread->stats.AddMessage(msg);
    // }


    // int write=0;
    // for (int i = 0; i < num_; i++) {
    //   if (FLAGS_ycsb_uniform==1){
    //    write= thread->rand.Next() % FLAGS_num;
    //   }else{
    //     write =  thread->rand.Zipfian(FLAGS_num, 1.0);
    //   }
    //   snprintf(key, sizeof(key), "%016d", write);
    //   db_->Put(write_options_, key, gen.Generate(value_size_));
    // }

    int k2=0;
    for (int i = 0; i < num_; i++) {
          
      if (FLAGS_ycsb_uniform==1){
      k2 = thread->rand.Next() % FLAGS_num;
      }else{
      k2 =  thread->rand.Zipfian(FLAGS_num, 1.0);
      }
      snprintf(key, sizeof(key), "%016d", k2);


      int next_op = thread->rand.Next() % 100;
      
      if (next_op < 95){

        Iterator* iter = db_->NewIterator(ReadOptions());
        int i = 0;
        for (iter->SeekToFirst(); i < 100 && iter->Valid(); iter->Next()) {
          bytes += iter->key().size() + iter->value().size();
          
          ++i;
        }
        delete iter;
        thread->stats.FinishedSingleOp();
       
        reads_done++;
      }else{
          db_->Put(write_options_, key, gen.Generate(value_size_));
          bytes += value_size_ + strlen(key);
          writes_done++;
          thread->stats.FinishedSingleOp();
          

      }
    
    
    }
    thread->stats.AddBytes(bytes);
    
    char msg[100];
        snprintf(msg, sizeof(msg), "( reads:%d"  " writes:%d" \
             " total:%d"  " found:%d"  " not found:%d"  ")",
             reads_done, writes_done, num_, found , not_found);
    thread->stats.AddMessage(msg);   

    
  }
   void YCSBF(ThreadState* thread) { // here
    RandomGenerator gen;
    ReadOptions options;
    WriteBatch batch;
    Status s;
    std::string value;
    int64_t bytes = 0;
    int64_t found = 0;
    int64_t reads_done = 0;
    int64_t not_found = 0;
    int64_t writes_done = 0;
    int k;
    batch.Clear();
    char key[100];
    num_=10000000;
    // if (num_ != FLAGS_num) {
    //   char msg[100];
    //   snprintf(msg, sizeof(msg), "(%d ops)", num_);
    //   thread->stats.AddMessage(msg);
    // }




    // int write=0;
    // for (int i = 0; i < num_; i++) {
    //   if (FLAGS_ycsb_uniform==1){
    //    write= thread->rand.Next() % FLAGS_num;
    //   }else{
    //     write =  thread->rand.Zipfian(FLAGS_num, 1.0);
    //   }
    //   snprintf(key, sizeof(key), "%016d", write);
    //   db_->Put(write_options_, key, gen.Generate(value_size_));
    // }

    int k2=0;
    for (int i = 0; i < num_; i++) {
          
      if (FLAGS_ycsb_uniform==1){
      k2 = thread->rand.Next() % FLAGS_num;
      }else{
      k2 =  thread->rand.Zipfian(FLAGS_num, 1.0);
      }
      snprintf(key, sizeof(key), "%016d", k2);



      int next_op = thread->rand.Next() % 100;
      
      if (next_op < 50){

         if (db_->Get(options, key, &value).ok()) {
          found++;
          bytes += value.size() + strlen(key);
          thread->stats.FinishedSingleOp();
        }else{
          not_found++;
        }
      
          reads_done++;
      }else{
          db_->Get(options, key, &value);
          db_->Put(write_options_, key, gen.Generate(value_size_));
          bytes += value_size_ + strlen(key);
          writes_done++;
          thread->stats.FinishedSingleOp();

      }

    
    }
    thread->stats.AddBytes(bytes);
    char msg[100];
        snprintf(msg, sizeof(msg), "( reads:%d"  " writes:%d" \
             " total:%d"  " found:%d"  " not found:%d"  ")",
             reads_done, writes_done, num_, found , not_found);
    thread->stats.AddMessage(msg);   

    
  }



  void unirandom(ThreadState* thread) {

    RandomGenerator gen;

    Status s;
    int64_t bytes = 0;
    
    if (num_ != FLAGS_num) {
      char msg[100];
      snprintf(msg, sizeof(msg), "(%d ops)", num_);
      thread->stats.AddMessage(msg);
    }

      for (int i = 0; i < num_; i += entries_per_batch_) {
          const int k = thread->rand.Next() % FLAGS_num;
          // const int k = data[i];
          // printf("key: %d\n", k);
          char key[100];
          snprintf(key, sizeof(key), "%016d", k);
          // db_->Put(write_options_, key, Slice("11111111111111"));
          db_->Put(write_options_, key, gen.Generate(value_size_));
          bytes += value_size_ + strlen(key);
          thread->stats.FinishedSingleOp();
        
      }
      thread->stats.AddBytes(bytes);
      // only for read performance test   只有测试读性能时才需要手动compaction
      // if (adgMod::bwise==1 or adgMod::sst_size>=1){
      // // db_->CompactRange(nullptr, nullptr);
      // static_cast<leveldb::DBImpl*>(db_)->CompactOrderdRange(nullptr, nullptr, 0);
      // }
  }




  void unirandom_read(ThreadState* thread) {
    
    ReadOptions options;
    std::string value;
    int found = 0;
    char key[100];
    int64_t bytes = 0;
    int k;

      for (int i = 0; i < reads_ ; i++) {
        // printf("key: %d\n", data[i]);
          k = data[i];
         snprintf(key, sizeof(key), "%016d", k);
        //  printf("the_key: %s\n", the_key.c_str());
        //  printf("key22: %s\n", key.c_str());
        if (db_->Get(options, key, &value).ok()) {
                found++;
                bytes += value.size() + strlen(key);
          }
          thread->stats.FinishedSingleOp();
    }
    thread->stats.AddBytes(bytes);
    char msg[100];
    snprintf(msg, sizeof(msg), "(%d of %d found)", found, num_);
    thread->stats.AddMessage(msg);
  }




  void real_workload_w(ThreadState* thread) {
    RandomGenerator gen;
    std::string value;
    int64_t bytes = 0;
    uint64_t i = 0;
    string the_key;
    // std::random_shuffle(data.begin(), data.end());

  //  while(data.size()){
    // for (int i = 0; i < data.size(); i++) {
      for (int i = 0; i < num_ ; i++) {
         auto start = std::chrono::high_resolution_clock::now(); // start time
        // printf("key: %s\n",std::to_string(data[i]));
        the_key = adgMod::generate_key(std::to_string(data[i]));
        // the_key= std::to_string(data[i]);
        // printf("the_key: %s\n", the_key.c_str());
        db_->Put(write_options_, the_key, gen.Generate(value_size_));

        auto end = std::chrono::high_resolution_clock::now(); // end time
        auto latency = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count(); // latency in nanoseconds
        latencies_w.push_back(latency); // store latency in vector

        thread->stats.FinishedSingleOp();
        bytes += value_size_ + the_key.length();
   }
  // printf("num: %d\n", num_);
  thread->stats.AddBytes(bytes);
  printf("finish writing\n");
  // input.close();
  // static_cast<leveldb::DBImpl*>(db_)->CompactOrderdRange(nullptr, nullptr, 0);
}


  void real_workload_r(ThreadState* thread) {
    static_cast<leveldb::DBImpl*>(db_)->WaitForBackground();
    RandomGenerator gen;
    ReadOptions options;
    std::string value;  

    // using key_type = uint64_t;
    // std::vector<key_type> data;

    string the_key;
    
    int found = 0;
    int64_t bytes = 0;

      for (int i = 0; i < reads_ ; i++) {
        // printf("key: %d\n", data[i]);
         auto start = std::chrono::high_resolution_clock::now(); // start time
        the_key = adgMod::generate_key(std::to_string(data[i]));
         
        //  printf("key22: %s\n", the_key.c_str());
        if (db_->Get(options, the_key, &value).ok()) {
              found++;
              bytes += value.size() + the_key.length();
              // printf("the_key: %s\n", the_key.c_str());
          }
        auto end = std::chrono::high_resolution_clock::now(); // end time
        auto latency = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count(); // latency in nanoseconds
        latencies.push_back(latency); // store latency in vector

        thread->stats.FinishedSingleOp();
    }
    // printf("sssssssssssssthe_key: %s\n", the_key.c_str());
    thread->stats.AddBytes(bytes);
    char msg[100];
    snprintf(msg, sizeof(msg), "(%d of %d found)", found, num_);
    thread->stats.AddMessage(msg);
  }





  void ReadSequential(ThreadState* thread) {
    Iterator* iter = db_->NewIterator(ReadOptions());
    int i = 0;
    int64_t bytes = 0;
    for (iter->SeekToFirst(); i < reads_ && iter->Valid(); iter->Next()) {
      bytes += iter->key().size() + iter->value().size();
      thread->stats.FinishedSingleOp();
      ++i;
    }
    delete iter;
    thread->stats.AddBytes(bytes);
  }

  void ReadReverse(ThreadState* thread) {
    Iterator* iter = db_->NewIterator(ReadOptions());
    int i = 0;
    int64_t bytes = 0;
    for (iter->SeekToLast(); i < reads_ && iter->Valid(); iter->Prev()) {
      bytes += iter->key().size() + iter->value().size();
      thread->stats.FinishedSingleOp();
      ++i;
    }
    delete iter;
    thread->stats.AddBytes(bytes);
  }

 void zipfianread(ThreadState* thread) {
    ReadOptions options;
    std::string value;
    int found = 0;
    char key[100];
    int64_t bytes = 0;
    int k;
      for (int i = 0; i < reads_; i++) {
        int k =  thread->rand.Zipfian(FLAGS_num, 1);
        snprintf(key, sizeof(key), "%016d", k);
        if (db_->Get(options, key, &value).ok()) {
          found++;
          bytes += value.size() + strlen(key);
        }
        thread->stats.FinishedSingleOp();
      }

    thread->stats.AddBytes(bytes);
    char msg[100];
    snprintf(msg, sizeof(msg), "(%d of %d found)", found, num_);
    thread->stats.AddMessage(msg);
  }
  
 void ReadRandom(ThreadState* thread) {
    // static_cast<leveldb::DBImpl*>(db_)->WaitForBackground();
    ReadOptions options;
    std::string value;
    int found = 0;
    char key[100];
    int64_t bytes = 0;
    int k;
      for (int i = 0; i < reads_; i++) {
        auto start = std::chrono::high_resolution_clock::now(); // start time
        k = thread->rand.Next() % FLAGS_num;
        // const int k = thread->rand.Uniform(FLAGS_num);
        snprintf(key, sizeof(key), "%016d", k);
        
        if (db_->Get(options, key, &value).ok()) {
          found++;
          bytes += value.size() + strlen(key);
        }
        
        auto end = std::chrono::high_resolution_clock::now(); // end time
        auto latency = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count(); // latency in nanoseconds
        latencies.push_back(latency); // store latency in vector
        thread->stats.FinishedSingleOp();
      }

    thread->stats.AddBytes(bytes);
    char msg[100];
    snprintf(msg, sizeof(msg), "(%d of %d found)", found, num_);
    thread->stats.AddMessage(msg);
  }

void Get99PercentileLatency(ThreadState* thread) {

  std::sort(latencies.begin(), latencies.end());
  std::sort(latencies_w.begin(), latencies_w.end());
  int index = 0.99 * latencies.size();
  int index_w = 0.99 * latencies_w.size();

  // thread->stats.AddMessage("\nRead 99 percentile latency: " + std::to_string(latencies[index]) + " ns\n");
  printf("Read 99 percentile latency: %ld ns\n", latencies[index]);
  printf("Write 99 percentile latency: %ld ns\n", latencies_w[index_w]);
  // thread->stats.AddMessage("Write 99 percentile latency: " + std::to_string(latencies_w[index_w]) + " ns\n");
  
}


  void ReadMissing(ThreadState* thread) {
    ReadOptions options;
    std::string value;
    for (int i = 0; i < reads_; i++) {
      char key[100];
      const int k = thread->rand.Next() % FLAGS_num;
      snprintf(key, sizeof(key), "%016d.", k);
      db_->Get(options, key, &value);
      thread->stats.FinishedSingleOp();
    }
  }

  void ReadHot(ThreadState* thread) {
    ReadOptions options;
    std::string value;
    const int range = (FLAGS_num + 99) / 100;
    for (int i = 0; i < reads_; i++) {
      char key[100];
      const int k = thread->rand.Next() % range;
      snprintf(key, sizeof(key), "%016d", k);
      db_->Get(options, key, &value);
      thread->stats.FinishedSingleOp();
    }
  }

  void SeekRandom(ThreadState* thread) {
    ReadOptions options;
    int found = 0;
    for (int i = 0; i < reads_; i++) {
      Iterator* iter = db_->NewIterator(options);
      char key[100];
      const int k = thread->rand.Next() % FLAGS_num;
      snprintf(key, sizeof(key), "%016d", k);
      iter->Seek(key);
      if (iter->Valid() && iter->key() == key) found++;
      delete iter;
      thread->stats.FinishedSingleOp();
    }
    char msg[100];
    snprintf(msg, sizeof(msg), "(%d of %d found)", found, num_);
    thread->stats.AddMessage(msg);
  }

  void DoDelete(ThreadState* thread, bool seq) {
    RandomGenerator gen;
    WriteBatch batch;
    Status s;
    for (int i = 0; i < num_; i += entries_per_batch_) {
      batch.Clear();
      for (int j = 0; j < entries_per_batch_; j++) {
        const int k = seq ? i + j : (thread->rand.Next() % FLAGS_num);
        char key[100];
        snprintf(key, sizeof(key), "%016d", k);
        batch.Delete(key);
        thread->stats.FinishedSingleOp();
      }
      s = db_->Write(write_options_, &batch);
      if (!s.ok()) {
        fprintf(stderr, "del error: %s\n", s.ToString().c_str());
        exit(1);
      }
    }
  }

  void DeleteSeq(ThreadState* thread) { DoDelete(thread, true); }

  void DeleteRandom(ThreadState* thread) { DoDelete(thread, false); }

  void ReadWhileWriting(ThreadState* thread) {
    if (thread->tid > 0) {
      ReadRandom(thread);
    } else {
      // Special thread that keeps writing until other threads are done.
      RandomGenerator gen;
      while (true) {
        {
          MutexLock l(&thread->shared->mu);
          if (thread->shared->num_done + 1 >= thread->shared->num_initialized) {
            // Other threads have finished
            break;
          }
        }

        const int k = thread->rand.Next() % FLAGS_num;
        char key[100];
        snprintf(key, sizeof(key), "%016d", k);
        Status s = db_->Put(write_options_, key, gen.Generate(value_size_));
        if (!s.ok()) {
          fprintf(stderr, "put error: %s\n", s.ToString().c_str());
          exit(1);
        }
      }

      // Do not count any of the preceding work/delay in stats.
      thread->stats.Start();
    }
  }

  void Compact(ThreadState* thread) { db_->CompactRange(nullptr, nullptr); }

  void PrintStats(const char* key) {
    std::string stats;
    if (!db_->GetProperty(key, &stats)) {
      stats = "(failed)";
    }
    fprintf(stdout, "\n%s\n", stats.c_str());
  }

  static void WriteToFile(void* arg, const char* buf, int n) {
    reinterpret_cast<WritableFile*>(arg)->Append(Slice(buf, n));
  }

  void HeapProfile() {
    char fname[100];
    snprintf(fname, sizeof(fname), "%s/heap-%04d", FLAGS_db, ++heap_counter_);
    WritableFile* file;
    Status s = g_env->NewWritableFile(fname, &file);
    if (!s.ok()) {
      fprintf(stderr, "%s\n", s.ToString().c_str());
      return;
    }
    bool ok = port::GetHeapProfile(WriteToFile, file);
    delete file;
    if (!ok) {
      fprintf(stderr, "heap profiling not supported\n");
      g_env->DeleteFile(fname);
    }
  }
};

}  // namespace leveldb

int main(int argc, char** argv) {
  FLAGS_write_buffer_size = leveldb::Options().write_buffer_size;
  FLAGS_max_file_size = leveldb::Options().max_file_size;
  FLAGS_block_size = leveldb::Options().block_size;
  FLAGS_open_files = leveldb::Options().max_open_files;
  std::string default_db_path;

  for (int i = 1; i < argc; i++) {
    double d;
    int n;
    char junk;
    if (leveldb::Slice(argv[i]).starts_with("--benchmarks=")) {
      FLAGS_benchmarks = argv[i] + strlen("--benchmarks=");
    } else if (sscanf(argv[i], "--compression_ratio=%lf%c", &d, &junk) == 1) {
      FLAGS_compression_ratio = d;
    } else if (sscanf(argv[i], "--histogram=%d%c", &n, &junk) == 1 &&
               (n == 0 || n == 1)) {
      FLAGS_histogram = n;
    } else if (sscanf(argv[i], "--use_existing_db=%d%c", &n, &junk) == 1 &&
               (n == 0 || n == 1)) {
      FLAGS_use_existing_db = n;
    } else if (sscanf(argv[i], "--reuse_logs=%d%c", &n, &junk) == 1 &&
               (n == 0 || n == 1)) {
      FLAGS_reuse_logs = n;
    } else if (sscanf(argv[i], "--num=%d%c", &n, &junk) == 1) {
      FLAGS_num = n;
    } else if (sscanf(argv[i], "--lsize=%d%c", &n, &junk) == 1) {
      adgMod::level_size = n;
    }else if (sscanf(argv[i], "--reads=%d%c", &n, &junk) == 1) {
      FLAGS_reads = n;
    } else if (sscanf(argv[i], "--adeb=%d%c", &n, &junk) == 1) {
      adgMod::adeb = n;
    } else if (sscanf(argv[i], "--threads=%d%c", &n, &junk) == 1) {
      FLAGS_threads = n;
    } else if (sscanf(argv[i], "--uni=%d%c", &n, &junk) == 1) {
      FLAGS_ycsb_uniform = n;
    }else if (sscanf(argv[i], "--value_size=%d%c", &n, &junk) == 1) {
      FLAGS_value_size = n;
    } else if (sscanf(argv[i], "--write_buffer_size=%d%c", &n, &junk) == 1) {
      FLAGS_write_buffer_size = n*1024*1024;
    } else if (sscanf(argv[i], "--max_file_size=%d%c", &n, &junk) == 1) {
      FLAGS_max_file_size = n*1024*1024;
    } else if (sscanf(argv[i], "--block_size=%d%c", &n, &junk) == 1) {
      FLAGS_block_size = n;
    } else if (sscanf(argv[i], "--cache_size=%d%c", &n, &junk) == 1) {
      FLAGS_cache_size = n;
    } else if (sscanf(argv[i], "--bloom_bits=%d%c", &n, &junk) == 1) {
      FLAGS_bloom_bits = n;
    } else if (sscanf(argv[i], "--open_files=%d%c", &n, &junk) == 1) {
      FLAGS_open_files = n;
    } else if (sscanf(argv[i], "--level_multiple=%d%c", &n, &junk) == 1) {
      adgMod::level_multiple = n;
    } else if (sscanf(argv[i], "--mod=%d%c", &n, &junk) == 1) {
      if (n==10){
      adgMod::MOD = n;  
      adgMod::sst_size = 4;
      adgMod::adeb = 1;
      printf("mod: %d\n", n);
      }else{    
      adgMod::MOD = n;
      printf("mod: %d\n", n);
      }
      
    } else if (sscanf(argv[i], "--bwise=%d%c", &n, &junk) == 1) {
      adgMod::bwise = n;
      adgMod::MOD = 7;
      adgMod::adeb=1;
      adgMod::sst_size = 4;
      adgMod::file_model_error=16;
    }else if (sscanf(argv[i], "--file_error=%d%c", &n, &junk) == 1) {
      adgMod::file_model_error=n;
    } else if (sscanf(argv[i], "--lac=%d%c", &n, &junk) == 1) {
      adgMod::MOD = 10;
      // printf("lac: %d\n", n);
      adgMod::sst_size = n;
    } else if (strncmp(argv[i], "--db=", 5) == 0) {
        FLAGS_db = argv[i] + 5;
    } else if (sscanf(argv[i], "--use_real_data=%d%c", &n, &junk) == 1 && (n == 0 || n == 1)) {
      FLAGS_use_real_data = n;
    } else if (leveldb::Slice(argv[i]).starts_with("--path_real_data=")) {
      FLAGS_path_real_data = argv[i] + strlen("--path_real_data=");
    }else {
      fprintf(stderr, "Invalid flag '%s'\n", argv[i]);
      exit(1);
    }
  }

  leveldb::g_env = leveldb::Env::Default();

  // Choose a location for the test database if none given with --db=<path>
  if (FLAGS_db == nullptr) {
    leveldb::g_env->GetTestDirectory(&default_db_path);
    default_db_path += "/dbbench";
    FLAGS_db = default_db_path.c_str();
  }

  leveldb::Benchmark benchmark;
  benchmark.Run();
  // if (adgMod::MOD == 7) {
  // adgMod::Stats* instance = adgMod::Stats::GetInstance();
  // instance->ReportTime();
  // }

  return 0;
}
