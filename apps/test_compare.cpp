// Self-test for the comparison code: it must catch errors we inject on purpose.
#include <cstdio>
#include <vector>
#include "compare.h"

static int failures = 0;
#define CHECK(cond) do { if (!(cond)) { printf("FAIL line %d: %s\n", __LINE__, #cond); failures++; } } while (0)

int main() {
  const size_t n = 1000;
  std::vector<uint8_t> a(n * 3, 100), b = a;

  DiffStats d = compare_images(a.data(), b.data(), n, 3, 1);
  CHECK(d.mismatched == 0 && d.max_abs == 0 && d.mean_abs == 0 && passes(d) && d.first_bad == (size_t)-1);

  b[3 * 10 + 1] = 101;  // one channel off by 1: within tolerance 1, still counted as mismatched
  d = compare_images(a.data(), b.data(), n, 3, 1);
  CHECK(d.mismatched == 1 && d.over_tol == 0 && d.max_abs == 1 && passes(d) && d.hist[1] == 1);

  b[3 * 20 + 2] = 98;   // off by 2: over tolerance 1
  d = compare_images(a.data(), b.data(), n, 3, 1);
  CHECK(d.mismatched == 2 && d.over_tol == 1 && d.max_abs == 2 && !passes(d) && d.first_bad == 20);
  d = compare_images(a.data(), b.data(), n, 3, 2);  // same data, looser tolerance passes
  CHECK(d.over_tol == 0 && passes(d));

  b[3 * 999] = 0;       // last pixel, large error, and the diff histogram bucket for 3+
  d = compare_images(a.data(), b.data(), n, 3, 1);
  CHECK(d.max_abs == 100 && d.over_tol == 2 && d.hist[3] == 1 && d.first_bad == 20);
  CHECK(d.mean_abs > 0.03 && d.mean_abs < 0.04);  // (1 + 2 + 100) / 3000 = 0.03433

  d = compare_images(a.data(), b.data(), 0, 3, 1);  // empty input must not divide by zero
  CHECK(d.mean_abs == 0 && passes(d));

  printf(failures ? "compare self-test: %d FAILED\n" : "compare self-test: all passed\n", failures);
  return failures ? 1 : 0;
}
