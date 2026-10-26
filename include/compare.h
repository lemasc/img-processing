#pragma once
// Stage 3 correctness check: pixel-by-pixel comparison of a test output against a reference.
// Every later optimization must pass this before its timings are used.
#include <cstddef>
#include <cstdint>
#include <string>

struct DiffStats {
  size_t pixels = 0;
  size_t mismatched = 0;     // pixels where any channel differs
  size_t over_tol = 0;       // pixels where any channel differs by more than the tolerance
  int max_abs = 0;           // largest per-channel difference, in 8-bit levels
  double mean_abs = 0;       // mean per-channel difference over all channels
  size_t hist[4] = {0, 0, 0, 0};  // per-channel differences of 0, 1, 2, 3 or more
  size_t first_bad = (size_t)-1;  // pixel index of the first pixel over tolerance
};

struct Tolerance {
  int max_abs = 1;  // allowed per-channel difference in 8-bit levels
};

// a and b hold `pixels` interleaved pixels of `channels` bytes each.
DiffStats compare_images(const uint8_t* a, const uint8_t* b, size_t pixels, int channels, int tol);

inline bool passes(const DiffStats& d) { return d.over_tol == 0; }

// One-line summary, for the terminal.
std::string describe(const DiffStats& d);

// Append a row to a CSV (header written if the file is new).
void log_diff(const std::string& csv, const std::string& kernel, const std::string& impl,
              const std::string& input, int tol, const DiffStats& d);
