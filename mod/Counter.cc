
#include <iostream>
#include <numeric>
#include "Counter.h"


void Counter::Increment(int level, uint64_t n) {
    counts[level] += n;
    nums[level] += 1;
    //Report();
}

void Counter::Reset() {
    for (uint64_t& count : counts) count = 0;
    for (uint64_t& num : nums) num = 0;
}

void Counter::Report() {
    //1 l0 slow stall
    //2 mem stall
    // 3 stop stall
    std::cout << "Counter " << name << " " << Sum();
    for (uint64_t count : counts) {
        std::cout << " " << count;
    }
    std::cout << "\n";
    std::cout << NumSum();
    for (uint64_t num : nums) {
        std::cout << " " << num;
    }
    std::cout << "\n";
}

int Counter::Sum() {
    return std::accumulate(counts.begin(), counts.end(), 0.0);
}

int Counter::NumSum() {
    return std::accumulate(nums.begin(), nums.end(), 0.0);
}