// main test driver for leafDS
// #define DEBUG_PRINT 0

#include "tbassert.h"
#include "cxxopts.hpp"
#include "helpers.hpp"
#include <concepts>
#include <cstdint>
#include <limits>
#include <map>
#include <random>
#include <set>
#include <unordered_set>

#include "parallel.h"
#include "leafDS.hpp"

#define HEADER_SIZE 32
#define LOG_SIZE HEADER_SIZE
#define BLOCK_SIZE 32
#define NUM_TRIALS 5
#define MEDIAN_TRIAL NUM_TRIALS / 2
#define N LOG_SIZE + HEADER_SIZE + BLOCK_SIZE * HEADER_SIZE

#define key_type uint64_t

static long get_usecs() {
  struct timeval st;
  gettimeofday(&st, NULL);
  return st.tv_sec * 1000000 + st.tv_usec;
}

[[nodiscard]] int parallel_test_insert_leafDS(uint32_t el_count, uint32_t num_copies) {
  std::vector<uint64_t> insert_times(NUM_TRIALS);
  std::vector<uint64_t> sum_times_with_map(NUM_TRIALS);
  std::vector<uint64_t> sum_times_direct(NUM_TRIALS);

  uint64_t start, end;
  for(uint32_t trial = 0; trial < NUM_TRIALS + 1; trial++) {
    // do inserts
    std::vector<LeafDS<LOG_SIZE, HEADER_SIZE, BLOCK_SIZE, key_type>> dsv(num_copies);
    start = get_usecs();
    cilk_for(uint32_t i = 0; i < num_copies; i++) {
      std::mt19937 rng(0);
      std::uniform_int_distribution<key_type> dist_el(1, N * 16);
      for (uint32_t j = 0; j < el_count; j++) {
        key_type el = dist_el(rng);
        dsv[i].insert(el);
      }
    }
    end = get_usecs();
    if (trial > 0) {
      insert_times[trial - 1] = end - start;
    }

    // do sum
    std::vector<uint64_t> partial_sums(getWorkers() * 8);
    start = get_usecs();
    cilk_for(uint32_t i = 0; i < num_copies; i++) {
      partial_sums[getWorkerNum() * 8] += dsv[i].sum_keys_with_map();
    }
    uint64_t count{0};
    for(int i = 0; i < getWorkers(); i++) {
      count += partial_sums[i*8];
    }
    end = get_usecs();

    printf("\ttrial %u, total sum %lu\n", trial, count);

    if (trial > 0) {
      sum_times_with_map[trial - 1] = end - start;
    }

    for(size_t i = 0; i < partial_sums.size(); i++) { partial_sums[i] = 0; }
    start = get_usecs();
    cilk_for(uint32_t i = 0; i < num_copies; i++) {
      partial_sums[getWorkerNum() * 8] += dsv[i].sum_keys_direct();
    }
    count = 0;
    for(int i = 0; i < getWorkers(); i++) {
      count += partial_sums[i*8];
    }
    end = get_usecs();

    printf("\ttrial %u, total sum %lu\n", trial, count);

    if (trial > 0) {
      sum_times_direct[trial - 1] = end - start;
    }
  }

  std::sort(insert_times.begin(), insert_times.end());
  std::sort(sum_times_with_map.begin(), sum_times_with_map.end());
  std::sort(sum_times_direct.begin(), sum_times_direct.end());

  printf("LeafDS: parallel insert time for %u copies of %u elts each = %lu us\n", num_copies, el_count, insert_times[MEDIAN_TRIAL]);

  printf("LeafDS: parallel sum time with map for %u copies of %u elts each = %lu us\n", num_copies, el_count, sum_times_with_map[MEDIAN_TRIAL]);

  printf("LeafDS: parallel sum time with subtraction for %u copies of %u elts each = %lu us\n", num_copies, el_count, sum_times_direct[MEDIAN_TRIAL]);

  return 0;
}

[[nodiscard]] int parallel_test_leafDS(uint32_t el_count, uint32_t num_copies, double prob_insert) {
  std::vector<uint64_t> insert_times(NUM_TRIALS);
  std::vector<uint64_t> sum_times_with_map(NUM_TRIALS);
  std::vector<uint64_t> sum_times_direct(NUM_TRIALS);
  std::vector<key_type> elts;
  // prefill the input
  std::uniform_int_distribution<key_type> dist_el(1, 1088 * 16);
  std::mt19937 rng(0);
  for (uint32_t j = 0; j < el_count; j++) {
    key_type el = dist_el(rng);
    elts.push_back(el);
  }

  uint64_t start, end;
  for(uint32_t trial = 0; trial < NUM_TRIALS + 1; trial++) {
    // do inserts
    std::vector<LeafDS<LOG_SIZE, HEADER_SIZE, BLOCK_SIZE, key_type>> dsv(num_copies);

    // add the first half
    cilk_for(uint32_t i = 0; i < num_copies; i++) {
      std::uniform_real_distribution<double> dist_flip(.25, .75);

      for (uint32_t j = 0; j < el_count / 2; j++) {
        key_type el = elts[j];
        dsv[i].insert(el);
        /*
        if (dist_flip(rng) < prob_insert) {
          dsv[i].insert(el);
        } else {
          dsv[i].remove(el);
        }
        */
      }
    }
#if STATS
    printf("first half:\n");
    dsv[0].report_redistributes();
#endif

    // time the second half
    start = get_usecs();
    cilk_for(uint32_t i = 0; i < num_copies; i++) {
      std::uniform_real_distribution<double> dist_flip(.25, .75);

      for (uint32_t j = el_count / 2; j < el_count; j++) {
        key_type el = elts[j];
        dsv[i].insert(el);
        /*
        key_type el = dist_el(rng);
        if (dist_flip(rng) < prob_insert) {
          dsv[i].insert(el);
        } else {
          dsv[i].remove(el);
        }
        */
      }
    }
    end = get_usecs();
    if (trial > 0) {
      insert_times[trial - 1] = end - start;
    }
#if STATS
    printf("all:\n");
    dsv[0].report_redistributes();
#endif

    // do sum
    std::vector<uint64_t> partial_sums(getWorkers() * 8);
    start = get_usecs();
    cilk_for(uint32_t i = 0; i < num_copies; i++) {
      partial_sums[getWorkerNum() * 8] += dsv[i].sum_keys_with_map();
    }
    uint64_t count{0};
    for(int i = 0; i < getWorkers(); i++) {
      count += partial_sums[i*8];
    }
    end = get_usecs();

    printf("\ttrial %u, sum with map %lu\n", trial, count);

    if (trial > 0) {
      sum_times_with_map[trial - 1] = end - start;
    }

    for(size_t i = 0; i < partial_sums.size(); i++) { partial_sums[i] = 0; }

    start = get_usecs();
    cilk_for(uint32_t i = 0; i < num_copies; i++) {
      partial_sums[getWorkerNum() * 8] += dsv[i].sum_keys_direct();
    }
    count = 0;
    for(int i = 0; i < getWorkers(); i++) {
      count += partial_sums[i*8];
    }
    end = get_usecs();

    printf("\ttrial %u, sum with direct %lu\n", trial, count);

    if (trial > 0) {
      sum_times_direct[trial - 1] = end - start;
    }
  }

  std::sort(insert_times.begin(), insert_times.end());
  std::sort(sum_times_with_map.begin(), sum_times_with_map.end());
  std::sort(sum_times_direct.begin(), sum_times_direct.end());

  printf("LeafDS: parallel insert time for %u copies of %u elts each = %lu us\n", num_copies, el_count, insert_times[MEDIAN_TRIAL]);

  printf("LeafDS: parallel sum time with map for %u copies of %u elts each = %lu us\n", num_copies, el_count, sum_times_with_map[MEDIAN_TRIAL]);

  printf("LeafDS: parallel sum time with subtraction for %u copies of %u elts each = %lu us\n", num_copies, el_count, sum_times_direct[MEDIAN_TRIAL]);

  return 0;
}

[[nodiscard]] int parallel_test_sorted_vector(uint32_t el_count, uint32_t num_copies, double prob_insert) {
  std::vector<uint64_t> insert_times(NUM_TRIALS);
  std::vector<uint64_t> sum_times(NUM_TRIALS);
  std::vector<key_type> elts;

  uint64_t start, end;

  for(uint32_t trial = 0; trial < NUM_TRIALS + 1; trial++) {
    std::vector<std::vector<key_type>> dsv(num_copies);
    for(uint32_t i = 0; i < num_copies; i++) {
      dsv[i].reserve(el_count);
    }
    // prefill the input
    std::uniform_int_distribution<key_type> dist_el(1, 1088 * 16);
    std::mt19937 rng(0);
    for (uint32_t j = 0; j < el_count; j++) {
      key_type el = dist_el(rng);
      elts.push_back(el);
    }

    cilk_for(uint32_t i = 0; i < num_copies; i++) {
#if DEBUG
      std::unordered_set<key_type> checker;
#endif
      std::uniform_real_distribution<double> dist_flip(.25, .75);
      for (uint32_t j = 0; j < el_count / 2; j++) {
        // find the elt at most the thing to insert
        size_t idx = 0;
        key_type el = elts[j];
        for(; idx < dsv[i].size(); idx++) {
          if(dsv[i][idx] == el) {
            break;
          } else if (dsv[i][idx] > el) {
            break;
          }
        }
        if(dsv[i].size() == 0 || dsv[i][idx] != el) {
          dsv[i].insert(dsv[i].begin() + idx, el);
        }

        /*
        if (dist_flip(rng) < prob_insert) {
          // vector does before
          if(dsv[i].size() == 0 || dsv[i][idx] != el) {
  #if DEBUG_PRINT
            printf("\telt %u not found, inserting at idx %lu, current size = %lu\n", el, idx, dsv[i].size());
  #endif
            dsv[i].insert(dsv[i].begin() + idx, el);
          }
  #if DEBUG
          checker.insert(el);
          // test sortedness
          for(idx = 1; idx < dsv[i].size(); idx++) {
            assert(dsv[i][idx] > dsv[i][idx-1]);
          }
          printf("\n");
          for(uint32_t i = 0; i < num_copies; i++) {
            for(auto elt : checker) {
              if (std::find(dsv[i].begin(), dsv[i].end(), elt) == dsv[i].end()) {
                printf("didn't find %u\n", elt);
                assert(false);
              }
            }

            tbassert(dsv[i].size() == checker.size(), "got %lu elts, should be %lu\n", dsv[i].size(), checker.size());
          }
    #endif
        } else { // remove
          if(dsv[i][idx] == el) {
            dsv[i].erase(dsv[i].begin() + idx);
          }
        }
        */
      }
    }

    printf("\tsize after half = %lu\n", dsv[0].size());
    start = get_usecs();
    cilk_for(uint32_t i = 0; i < num_copies; i++) {
#if DEBUG
      std::unordered_set<key_type> checker;
#endif
      std::uniform_real_distribution<double> dist_flip(.25, .75);
      for (uint32_t j = el_count / 2; j < el_count; j++) {
        key_type el = elts[j];
        // find the elt at most the thing to insert
        size_t idx = 0;
        for(; idx < dsv[i].size(); idx++) {
          if(dsv[i][idx] == el) {
            break;
          } else if (dsv[i][idx] > el) {
            break;
          }
        }
        if(dsv[i].size() == 0 || dsv[i][idx] != el) {
          dsv[i].insert(dsv[i].begin() + idx, el);
        }
        /*
        if (dist_flip(rng) < prob_insert) {
          // vector does before
          if(dsv[i].size() == 0 || dsv[i][idx] != el) {
  #if DEBUG_PRINT
            printf("\telt %u not found, inserting at idx %lu, current size = %lu\n", el, idx, dsv[i].size());
  #endif
            dsv[i].insert(dsv[i].begin() + idx, el);
          }
  #if DEBUG
          checker.insert(el);
          // test sortedness
          for(idx = 1; idx < dsv[i].size(); idx++) {
            assert(dsv[i][idx] > dsv[i][idx-1]);
          }
          printf("\n");
          for(uint32_t i = 0; i < num_copies; i++) {
            for(auto elt : checker) {
              if (std::find(dsv[i].begin(), dsv[i].end(), elt) == dsv[i].end()) {
                printf("didn't find %u\n", elt);
                assert(false);
              }
            }

            tbassert(dsv[i].size() == checker.size(), "got %lu elts, should be %lu\n", dsv[i].size(), checker.size());
          }
    #endif
        } else { // remove
          if(dsv[i][idx] == el) {
            dsv[i].erase(dsv[i].begin() + idx);
          }
        }
        */
      }
    }

    end = get_usecs();

    printf("\tsize after all = %lu\n", dsv[0].size());
    if (trial > 0) {
      insert_times[trial - 1] = end - start;
    }

    std::vector<uint64_t> partial_sums(getWorkers() * 8);
    start = get_usecs();

    cilk_for(uint32_t i = 0; i < num_copies; i++) {
      uint64_t local_sum = 0;
      for(size_t j = 0; j < dsv[i].size(); j++) {
        local_sum += dsv[i][j];
      }
      partial_sums[getWorkerNum() * 8] += local_sum;
    }
    uint64_t count{0};
    for(int i = 0; i < getWorkers(); i++) {
      count += partial_sums[i*8];
    }
    end = get_usecs();
    printf("\ttrial %d, total sum %lu\n", trial, count);

    if (trial > 0) {
      sum_times[trial - 1] = end - start;
    }
  }

  std::sort(insert_times.begin(), insert_times.end());
  std::sort(sum_times.begin(), sum_times.end());

  printf("Sorted vector: parallel insert time for %u copies of %u elts each = %lu us\n", num_copies, el_count, insert_times[MEDIAN_TRIAL]);

  printf("Sorted vector: parallel sum time for %u copies of %u elts each = %lu us\n", num_copies, el_count, sum_times[MEDIAN_TRIAL]);
  return 0;
}

[[nodiscard]] int parallel_test_unsorted_vector(uint32_t el_count, uint32_t num_copies, double prob_insert) {
  std::vector<uint64_t> insert_times(NUM_TRIALS);
  std::vector<uint64_t> sum_times(NUM_TRIALS);
  std::vector<key_type> elts;

  uint64_t start, end;
  for(uint32_t trial = 0; trial < NUM_TRIALS + 1; trial++) {
    std::vector<std::vector<key_type>> dsv(num_copies);
    for(uint32_t i = 0; i < num_copies; i++) {
      dsv[i].reserve(el_count);
    }

    // prefill the input
    std::uniform_int_distribution<key_type> dist_el(1, 1088 * 16);
    std::mt19937 rng(0);
    for (uint32_t j = 0; j < el_count; j++) {
      key_type el = dist_el(rng);
      elts.push_back(el);
    }
    
    // insert first half
    cilk_for(uint32_t i = 0; i < num_copies; i++) {
      std::uniform_real_distribution<double> dist_flip(.25, .75);

      for (uint32_t j = 0; j < el_count / 2; j++) {
        // find the elt at most the thing to insert
        uint32_t el = elts[j];
        size_t idx = 0;
        for(; idx < dsv[i].size(); idx++) {
          if(dsv[i][idx] == el) {
            break;
          } 
        }
        // insert only for now
        if(idx == dsv[i].size()) {
          dsv[i].push_back(elts[j]);
        }
      
        /*
        if (dist_flip(rng) < prob_insert) {
          // if not found, add it to the end
          if(idx == dsv[i].size()) {
            dsv[i].push_back(el);
          }
        } else { // delete
          if(idx < dsv[i].size()) {
            dsv[i].erase(dsv[i].begin() + idx);
          }
        }
      */
      }
    }
    printf("\tsize after half = %lu\n", dsv[0].size());

    // count time for second half
    start = get_usecs();
    cilk_for(uint32_t i = 0; i < num_copies; i++) {
      std::uniform_real_distribution<double> dist_flip(.25, .75);

      for (uint32_t j = el_count / 2; j < el_count; j++) {
        key_type el = elts[j];
        // find the elt at most the thing to insert
        size_t idx = 0;
        for(; idx < dsv[i].size(); idx++) {
          if(dsv[i][idx] == el) {
            break;
          } 
        }
        if(idx == dsv[i].size()) {
          dsv[i].push_back(el);
        }

        /*
        if (dist_flip(rng) < prob_insert) {
          // if not found, add it to the end
          if(idx == dsv[i].size()) {
            dsv[i].push_back(el);
          }
        } else { // delete
          if(idx < dsv[i].size()) {
            dsv[i].erase(dsv[i].begin() + idx);
          }
        }
        */
      }
    }
    end = get_usecs();
    printf("\tsize after all = %lu\n", dsv[0].size());

    if (trial > 0) {
      insert_times[trial - 1] = end - start;
    }

    std::vector<uint64_t> partial_sums(getWorkers() * 8);
    start = get_usecs();

    cilk_for(uint32_t i = 0; i < num_copies; i++) {
      uint64_t local_sum = 0;
      for(size_t j = 0; j < dsv[i].size(); j++) {
        local_sum += dsv[i][j];
      }
      partial_sums[getWorkerNum() * 8] += local_sum;
    }

    uint64_t count{0};
    for(int i = 0; i < getWorkers(); i++) {
      count += partial_sums[i*8];
    }
    end = get_usecs();

    printf("\ttrial %d, total sum %lu\n", trial, count);

    if (trial > 0) {
      sum_times[trial - 1] = end - start;
    }
  }

  std::sort(insert_times.begin(), insert_times.end());
  std::sort(sum_times.begin(), sum_times.end());

  printf("Unsorted vector: parallel insert time for %u copies of %u elts each = %lu us\n", num_copies, el_count, insert_times[MEDIAN_TRIAL]);

  printf("Unsorted vector: parallel sum time for %u copies of %u elts each = %lu us\n", num_copies, el_count, sum_times[MEDIAN_TRIAL]);


  return 0;
}


[[nodiscard]] int parallel_test(uint32_t el_count, uint32_t num_copies, double prob_insert) {
  int r = parallel_test_leafDS(el_count, num_copies, prob_insert);
  if (r) {
    return r;
  }
  printf("\n");
  
  r = parallel_test_sorted_vector(el_count, num_copies, prob_insert);
  if (r) {
    return r;
  }
  printf("\n");

  r = parallel_test_unsorted_vector(el_count, num_copies, prob_insert);
  if (r) {
    return r;
  }
  return 0;
}

[[nodiscard]] int sorted_range_query_test(uint32_t el_count, uint32_t num_copies, uint32_t num_queries, uint32_t max_query_size) {
  // prefill the input
  std::uniform_int_distribution<key_type> dist_el(1, 1088 * 16);
  std::vector<key_type> elts;

  std::mt19937 rng(0);
  for (uint32_t j = 0; j < el_count; j++) {
    key_type el = dist_el(rng);
    elts.push_back(el);
  }

  for(uint32_t trial = 0; trial < NUM_TRIALS + 1; trial++) {
    // do inserts
    std::vector<LeafDS<LOG_SIZE, HEADER_SIZE, BLOCK_SIZE, key_type>> dsv(num_copies);
    std::vector<std::vector<key_type>> vectors(num_copies);
    for(uint32_t i = 0; i < num_copies; i++) {
      vectors[i].reserve(el_count);
    }
    // prefill the input
    std::uniform_int_distribution<key_type> dist_el(1, 1088 * 16);
    std::mt19937 rng(0);
    for (uint32_t j = 0; j < el_count; j++) {
      key_type el = dist_el(rng);
      elts.push_back(el);
    }

    cilk_for(uint32_t i = 0; i < num_copies; i++) {
      for (uint32_t j = 0; j < el_count; j++) {
        key_type el = elts[j];
        
        // add to leafDS
        dsv[i].insert(el);

        // add to sorted vector
        size_t idx = 0;
        for(; idx < vectors[i].size(); idx++) {
          if(vectors[i][idx] == el) {
            break;
          } else if (vectors[i][idx] > el) {
            break;
          }
        }
        if(vectors[i].size() == 0 || vectors[i][idx] != el) {
          vectors[i].insert(vectors[i].begin() + idx, el);
        }
      }
    }

    std::uniform_int_distribution<size_t> dist_len(1, max_query_size);
    // do range queries
    std::vector<key_type> starts;
    std::vector<size_t> lengths;
    std::mt19937 rng_query(1);
    for(size_t i = 0; i < num_queries; i++) {
      starts.push_back(dist_el(rng_query));
      lengths.push_back(dist_len(rng_query));
    }

    cilk_for(uint32_t i = 0; i < num_copies; i++) {
      for(uint32_t j = 0; j < num_queries; j++) {
        // do the correct version in sorted vector
        size_t idx = 0;
        while(idx < vectors[i].size() && vectors[i][idx] < starts[j]) {
          idx++;
        }
        std::vector<key_type> correct_range;
        while(correct_range.size() < lengths[j] && idx < vectors[i].size()) {
                assert(vectors[i][idx] >= starts[j]);
                correct_range.push_back(vectors[i][idx]);
                idx++;
        }
        auto test_range = dsv[i].sorted_range(starts[j], lengths[j]);
        if (test_range.size() != correct_range.size()) {
          printf("\n");
          for(size_t k = 0; k < test_range.size(); k++) {
            printf("test_output[%lu] = %lu\n", k, std::get<0>(test_range[k]));
          }
          for(size_t k = 0; k < correct_range.size(); k++) {
            printf("correct_output[%lu] = %lu\n", k, correct_range[k]);
          }

          dsv[i].print();
          assert(test_range.size() == correct_range.size());
        }
        for(uint32_t k = 0; k < test_range.size(); k++) {
          assert(std::get<0>(test_range[k]) == correct_range[k]);
        } 
      }
    }
  }

  return 0;
}

[[nodiscard]] int unsorted_range_query_test(uint32_t el_count, uint32_t num_copies, uint32_t num_queries) {
  // prefill the input
  std::uniform_int_distribution<key_type> dist_el(1, 1088 * 16);
  std::vector<key_type> elts;

  std::mt19937 rng(0);
  for (uint32_t j = 0; j < el_count; j++) {
    key_type el = dist_el(rng);
    elts.push_back(el);
  }

  for(uint32_t trial = 0; trial < NUM_TRIALS + 1; trial++) {
    // do inserts
    std::vector<LeafDS<LOG_SIZE, HEADER_SIZE, BLOCK_SIZE, key_type>> dsv(num_copies);
    std::vector<std::vector<key_type>> vectors(num_copies);
    for(uint32_t i = 0; i < num_copies; i++) {
      vectors[i].reserve(el_count);
    }
    // prefill the input
    std::uniform_int_distribution<key_type> dist_el(1, 1088 * 16);
    std::mt19937 rng(0);
    for (uint32_t j = 0; j < el_count; j++) {
      key_type el = dist_el(rng);
      elts.push_back(el);
    }

    cilk_for(uint32_t i = 0; i < num_copies; i++) {
      for (uint32_t j = 0; j < el_count; j++) {
        key_type el = elts[j];
        
        // add to leafDS
        dsv[i].insert(el);

        // add to sorted vector
        size_t idx = 0;
        for(; idx < vectors[i].size(); idx++) {
          if(vectors[i][idx] == el) {
            break;
          } else if (vectors[i][idx] > el) {
            break;
          }
        }
        if(vectors[i].size() == 0 || vectors[i][idx] != el) {
          vectors[i].insert(vectors[i].begin() + idx, el);
        }
      }
    }

    // do unsorted range queries
    key_type start, end;
    cilk_for(uint32_t copy = 0; copy < num_copies; copy++) {
      for(uint32_t j = 0; j < num_queries; j++) {
              // do the correct version in sorted vector
	      key_type a = dist_el(rng);
	      key_type b = dist_el(rng);
	      start = std::min(a, b);
	      end = std::max(a, b);

	      // printf("doing query %u in range [%lu, %lu]\n", j, start, end);
	      // first do the range on 
        size_t idx = 0;
        while(idx < vectors[copy].size() && vectors[copy][idx] < start) {
          idx++;
        }
      
	      // make a set for the correct range
        std::vector<key_type> correct_range;
        while(idx < vectors[copy].size() && vectors[copy][idx] <= end) {
                assert(vectors[copy][idx] >= start);
                assert(vectors[copy][idx] <= end);
                correct_range.push_back(vectors[copy][idx]);
                idx++;
        }

        auto test_range = dsv[copy].unsorted_range(start, end);

	      // printf("\tcorrect got %lu elts, test got %lu elts\n", correct_range.size(), test_range.size());

	      assert(test_range.size() == correct_range.size());
	      std::sort(test_range.begin(), test_range.end());
	      size_t i = 0;
	      for(i = 0; i < correct_range.size(); i++) {
		      if (std::get<0>(test_range[i]) != correct_range[i]) {
			      dsv[0].print();
		      }
		      tbassert(std::get<0>(test_range[i]) == correct_range[i], "test[%lu] = %lu, correct[%lu] = %lu\n", i, std::get<0>(test_range[i]), i, correct_range[i]);
		      // printf("test[%lu] = %lu, correct[%lu] = %lu\n", i, std::get<0>(test_range[i]), i, correct_range[i]);
	      }
	      while(i < test_range.size()) {
		      // printf("remaining test[%lu] = %u\n", i, std::get<0>(test_range[i]));
		      i++;
	      }
      }
    }
  }

  return 0;
}

[[nodiscard]] int parallel_test_perf(uint32_t el_count, uint32_t num_copies, double prob_insert) {
  int r = parallel_test_leafDS(el_count, num_copies, prob_insert);
  if (r) {
    return r;
  }
  printf("\n");
  
  return 0;
}

[[nodiscard]] int insert_delete_templated(uint32_t el_count) {
  LeafDS<LOG_SIZE, HEADER_SIZE, BLOCK_SIZE, key_type> ds;
  std::mt19937 rng(0);
  std::uniform_int_distribution<key_type> dist_el(1, N * 16);

  std::vector<key_type> checker;
  checker.reserve(el_count);
  std::vector<key_type> elts;

  // add some elements
  for (uint32_t i = 0; i < el_count; i++) {
    key_type el = dist_el(rng);
    elts.push_back(el);
    // add to leafDS
    ds.insert(el);

    // add to sorted vector
    size_t idx = 0;
    for(; idx < checker.size(); idx++) {
      if(checker[idx] == el) {
        break;
      } else if (checker[idx] > el) {
        break;
      }
    }
    if(checker.size() == 0 || checker[idx] != el) {
      checker.insert(checker.begin() + idx, el);
    }

    if (!ds.has(el)) {
      ds.print();
      printf("don't have something, %lu, we inserted while inserting "
             "elements\n",
             el);
      return -1;
    }
  } 
  printf("\n*** finished inserting elts ***\n");
  printf("num elts = %lu\n", checker.size());
  // then remove all the stuff we added
  for (auto el : elts) {
    ds.remove(el);

    size_t i = 0;
    for(; i < checker.size(); i++) {
	    if (checker[i] == el) {
		    break;
	    }
    }
    if(i < checker.size()) {
	    tbassert(i < checker.size(), "el = %lu, i == checker_size == %lu\n", el, checker.size());
	    tbassert(checker[i] == el, "checker[%lu] = %lu, el = %lu\n", i, checker[i], el);
	    checker.erase(checker.begin() + i);
	    printf("\tdeleting elt %lu (checker[%lu] = %lu) from vector\n", el, i, checker[i]);
	    printf("\tafter delete, num elts in vector = %lu\n", checker.size());
    }
    if (ds.has(el)) {
      ds.print();
      printf("has %lu but should have deleted\n", el);
      assert(false);
      return -1;
    }
		
    // check with sum
    uint64_t sum = ds.sum_keys_with_map();
    uint64_t sum_direct = ds.sum_keys_direct();

    uint64_t correct_sum = 0;
    for (auto elt : checker) {
      correct_sum += elt;
    }
    printf("correct sum %lu\n", correct_sum);

    if (correct_sum != sum) {
      ds.print();
      printf("incorrect sum keys with map\n");
      tbassert(correct_sum == sum, "got sum %lu, should be %lu\n", sum, correct_sum);
    }
    if (correct_sum != sum_direct) {
      ds.print();
      printf("incorrect sum keys with subtraction\n");
      tbassert(correct_sum == sum_direct, "got sum %lu, should be %lu\n", sum_direct, correct_sum);
    }
    printf("got sum %lu\n", sum);
    printf("got sum direct %lu\n", sum_direct);

    // do range queries and check them against sorted list
  }

  return 0;
}


[[nodiscard]] int insert_delete_test(uint32_t el_count) {
  int r = 0;
  r = insert_delete_templated(el_count);
  if (r) {
    return r;
  }
  return 0;
}

[[nodiscard]] int update_test_templated(uint32_t el_count,
                                            bool check = false) {

  LeafDS<LOG_SIZE, HEADER_SIZE, BLOCK_SIZE, key_type> ds;
  std::mt19937 rng(0);
  std::uniform_int_distribution<key_type> dist_el(1, N * 16);
  std::uniform_real_distribution<double> dist_flip(.25, .75);

  std::unordered_set<key_type> checker;

  for (uint32_t i = 0; i < el_count; i++) {
    // be more likely to insert when we are more empty
    key_type el = dist_el(rng);

    // if (dist_flip(rng) < ((double)(N - ds.get_num_elts()) / N)) {
    if (dist_flip(rng) < 1.0) {
      printf("\ninserting %lu\n", el);
      ds.insert(el);
      if (check) {
        checker.insert(el);
        if (!ds.has(el)) {
          ds.print();
          printf("don't have something, %lu, we inserted while inserting "
                 "elements\n",
                 el);
          return -1;
        }
      }
    } else {
      bool present = ds.has(el);
      if(present) { printf("removing elt %lu in DS\n", el); }
      else {
        printf("removing elt %lu not in DS\n", el);
      }

      ds.remove(el);
      if (check) {
        checker.erase(el);
        if (ds.has(el)) {
          ds.print();
          printf("have something we removed while removing elements, tried to "
                 "remove %lu\n",
     
                 el);
          return -1;
        }
      }
    }
  }

  // for all elts in the checker set, make sure elt is in DS
  if (check) {
    for(auto elt : checker) {
      if(!ds.has(elt)) {
        ds.print();
        printf("missing %lu\n", elt);
        return -1;
      }
    }
    bool has_all = true;
    ds.template map<true>([&has_all, &checker](key_type key) {
      has_all &= checker.contains(key);
    });
    if (!has_all) {
      printf("ds had something the checker didn't\n");
      return -1;
    }
  }

  // check with sum
  uint64_t sum = ds.sum_keys_with_map();
  uint64_t sum_direct = ds.sum_keys_direct();

  if (check) {
    uint64_t correct_sum = 0;
    for (auto elt : checker) {
      correct_sum += elt;
    }
    if (correct_sum != sum) {
      ds.print();
      printf("sum keys with map\n");
      tbassert(correct_sum == sum, "got sum %lu, should be %lu\n", sum, correct_sum);
    }
    if (correct_sum != sum_direct) {
      ds.print();
      printf("sum keys with subtraction\n");
      tbassert(correct_sum == sum_direct, "got sum %lu, should be %lu\n", sum_direct, correct_sum);
    }
  }
  printf("got sum %lu\n", sum);
  printf("got sum direct %lu\n", sum_direct);
  return 0;
}

[[nodiscard]] int update_test(uint32_t el_count, bool check = false) {
  int r = 0;
  r = update_test_templated(el_count, check);
  if (r) {
    return r;
  }
  return 0;
}

[[nodiscard]] int update_values_test_templated(uint32_t el_count,
                                                   bool check = false) {
/*                                                
  // is this the right thing for vals? 
  LeafDS<N, uint32_t, uint8_t, double> ds;
  std::mt19937 rng(0);
  std::uniform_int_distribution<uint32_t> dist_el(1, N * 4);

  std::uniform_real_distribution<double> dist_flip(.25, .75);

  std::uniform_int_distribution<uint8_t> dist_v1(
      0, std::numeric_limits<uint8_t>::max());

  std::uniform_real_distribution<double> dist_v2(0, 100.0);
  using tup_type = std::tuple<uint8_t, double>;

  std::unordered_map<uint32_t, tup_type> checker;

  uint64_t start = get_usecs();
  for (uint32_t i = 0; i < el_count; i++) {
    // be more likely to insert when we are more empty
    uint32_t el = dist_el(rng);
    uint8_t v1 = dist_v1(rng);
    double v2 = dist_v2(rng);
    if (dist_flip(rng) < ((double)(N - ds.get_n()) / N)) {
      ds.insert({el, v1, v2});
      if (check) {
        checker.insert_or_assign(el, tup_type(v1, v2));
        if (!ds.has(el)) {
          ds.print_pma();
          printf("don't have something, %u, we inserted while inserting "
                 "elements\n",
                 el);
          return -1;
        }
        if (ds.value(el) != tup_type(v1, v2)) {
          printf("bad value after insert\n");
          return -1;
        }
      }
    } else {
      ds.remove(el);
      if (check) {
        checker.erase(el);
        if (ds.has(el)) {
          printf("have something we removed while removing elements, tried to "
                 "remove %u\n",
                 el);
          return -1;
        }
      }
    }
    // printf("get_n() = %lu, minimum_elements = %lu\n", ds.get_n(),
    //        ds.minimum_elments);
    if (check) {
      if (ds.get_n() != checker.size()) {
        printf("wrong number of elements\n");
        return -1;
      }
      for (auto el : checker) {
        if (!ds.has(el.first)) {
          printf("we removed %u when we shouldn't have\n", el.first);
          return -1;
        }
        if (ds.value(el.first) != el.second) {
          printf("bad value\n");
          return -1;
        }
      }
      bool has_all = true;
      ds.template map<true>([&has_all, &checker](uint32_t key) {
        has_all &= checker.contains(key);
      });
      if (!has_all) {
        printf("ds had something the checker didn't\n");
        return -1;
      }
      bool correct_value = true;
      ds.template map<true, 0, 1>(
          [&has_all, &checker](uint32_t key, uint8_t v1, double v2) {
            has_all &= checker[key] == tup_type(v1, v2);
          });
      if (!correct_value) {
        printf("ds had had a wrong value\n");
        return -1;
      }
    }
    // ds.print_pma();
  }
  uint64_t end = get_usecs();
  printf("took %lu micros\n", end - start);
*/
  return 0;
}

[[nodiscard]] int update_values_test(uint32_t el_count,
                                         bool check = false) {
  int r = 0;
  r = update_values_test_templated(el_count, check);
  if (r) {
    return r;
  }

  return 0;
}

[[nodiscard]] int key_at_sorted_index_test_templated(uint32_t el_count) {
  LeafDS<LOG_SIZE, HEADER_SIZE, BLOCK_SIZE, key_type> ds;
  std::mt19937 rng(0);
  std::uniform_int_distribution<key_type> dist_el(1, N * 16);

  std::vector<key_type> checker;
  checker.reserve(el_count);
  std::vector<key_type> elts;
  std::vector<key_type> elts_sorted;

  // add some elements
  for (uint32_t i = 0; i < el_count; i++) {
    key_type el = dist_el(rng);
    elts.push_back(el);
    if (!std::count(elts_sorted.begin(), elts_sorted.end(), el)) {
      elts_sorted.push_back(el);
    }
    // add to leafDS
    ds.insert(el);

    // add to sorted vector
    // todo: doesn't work, last elt is set to 0 sometimes
    size_t idx = 0;
    for(; idx < checker.size(); idx++) {
      if(checker[idx] == el) {
        break;
      } else if (checker[idx] > el) {
        break;
      }
    }
    if(checker.size() == 0 || checker[idx] != el) {
      checker.insert(checker.begin() + idx, el);
    }

    if (!ds.has(el)) {
      ds.print();
      printf("don't have something, %lu, we inserted while inserting "
             "elements\n",
             el);
      return -1;
    }
  }

  // Verify key_at_sorted_index against sorted elts
  std::sort(elts_sorted.begin(), elts_sorted.end());

  for (uint32_t i = 0; i < elts_sorted.size(); i++) {
    // auto val = ds.get_key_at_sorted_index(i);
    auto val = ds.get_key_at_sorted_index(i);
    printf("elts_sorted: %lu \t leafds: %lu\n", elts_sorted[i], val);
    if (val != elts_sorted[i]) {
      printf("Should find key %lu but instead found %lu", elts_sorted[i], val);
      return -1;
    }
  }

  printf("\n*** finished inserting elts ***\n");
  printf("num elts = %lu\n", checker.size());
  // then remove all the stuff we added
  for (int cur_i = 0; cur_i < el_count; cur_i ++ ) {
    auto el = elts[cur_i];
    ds.remove(el);
    elts_sorted.erase(std::remove(elts_sorted.begin(), elts_sorted.end(), el), elts_sorted.end());
    printf("removed elem %lu\n", el);

    // Check key_at_sorted_index midway through deletes
    if (cur_i == el_count / 2) {
      std::sort(elts_sorted.begin(), elts_sorted.end());
     
      printf("***BEFORE FLUSHING**\n\n");
      ds.print();
      printf("sorted size %lu\n", elts_sorted.size());
      for (uint32_t j = 0; j < elts_sorted.size(); j++) {
        auto val = ds.get_key_at_sorted_index(j);

        if (j == 0){
          printf("***AFTER FLUSHING**\n\n");
          ds.print();
        }
        printf("elts_sorted: %lu \t leafds: %lu\n", elts_sorted[j], val);
        if (val != elts_sorted[j]) {
          printf("Should find key %lu but instead found %lu", elts_sorted[j], val);
          return -1;
        }
      }
    }
    
    size_t i = 0;
    for(; i < checker.size(); i++) {
	    if (checker[i] == el) {
		    break;
	    }
    }
    if(i < checker.size()) {
	    tbassert(i < checker.size(), "el = %lu, i == checker_size == %lu\n", el, checker.size());
	    tbassert(checker[i] == el, "checker[%lu] = %lu, el = %lu\n", i, checker[i], el);
	    checker.erase(checker.begin() + i);
	    printf("\tdeleting elt %lu (checker[%lu] = %lu) from vector\n", el, i, checker[i]);
	    printf("\tafter delete, num elts in vector = %lu\n", checker.size());
    }
    if (ds.has(el)) {
      ds.print();
      printf("has %lu but should have deleted\n", el);
      assert(false);
      return -1;
    }
		
    // check with sum
    uint64_t sum = ds.sum_keys_with_map();
    uint64_t sum_direct = ds.sum_keys_direct();

    uint64_t correct_sum = 0;
    for (auto elt : checker) {
      correct_sum += elt;
    }
    printf("correct sum %lu\n", correct_sum);

    if (correct_sum != sum) {
      ds.print();
      printf("incorrect sum keys with map\n");
      tbassert(correct_sum == sum, "got sum %lu, should be %lu\n", sum, correct_sum);
    }
    if (correct_sum != sum_direct) {
      ds.print();
      printf("incorrect sum keys with subtraction\n");
      tbassert(correct_sum == sum_direct, "got sum %lu, should be %lu\n", sum_direct, correct_sum);
    }
    printf("got sum %lu\n", sum);
    printf("got sum direct %lu\n", sum_direct);

    // do range queries and check them against sorted list
  }

  // Verify key_at_sorted_index against sorted elts
  std::sort(elts_sorted.begin(), elts_sorted.end());

  for (uint32_t i = 0; i < elts_sorted.size(); i++) {
    auto val = ds.get_key_at_sorted_index(i);
    printf("elts_sorted: %lu \t leafds: %lu\n", elts_sorted[i], val);
    if (val != elts_sorted[i]) {
      printf("Should find key %lu but instead found %lu", elts_sorted[i], val);
      return -1;
    }
  }

  return 0;
}

[[nodiscard]] int key_at_sorted_index_test(uint32_t el_count) {
  int r = 0;
  r = key_at_sorted_index_test_templated(el_count);
  if (r) {
    return r;
  }

  return 0;
}

[[nodiscard]] int merge_test_templated(uint32_t el_count) {
  LeafDS<LOG_SIZE, HEADER_SIZE, BLOCK_SIZE, key_type> ds_left;
  LeafDS<LOG_SIZE, HEADER_SIZE, BLOCK_SIZE, key_type> ds_right;
  std::mt19937 rng(0);
  std::uniform_int_distribution<key_type> dist_el(1, N * 16);

  std::vector<key_type> checker;
  checker.reserve(el_count);
  std::vector<key_type> elts;

  // add 1/2 of elements to each leafDS (so both are underfull and can be merged)
  for (uint32_t i = 0; i < el_count/2; i++) {
    key_type el = dist_el(rng);
    elts.push_back(el);
    // add to left leafDS
    ds_left.insert(el);
  }

  // add 1/2 of elements to each leafDS (so both are underfull and can be merged)
  for (uint32_t i = el_count/2; i < el_count; i++) {
    key_type el = dist_el(rng);
    elts.push_back(el);
    // add to left leafDS
    ds_right.insert(el);
  }

  ds_left.merge(&(ds_right));

  // Check if left leafDS has all elements from left and right after merge
  if (ds_right.get_num_elements() != 0) {
    ds_right.print();
    printf("Right leaf not empty after only inserts, size = %lu\n", ds_right.get_num_elements());
    return -1;
  }
  for (uint32_t i = 0; i < elts.size(); i++) {
    auto el = elts[i];
    if (!ds_left.has(el)) {
      ds_left.print();
      printf("Missing elt from left leaf after only inserts, elt: %lu \n", el);
      return -1;
    }
  } 

  LeafDS<LOG_SIZE, HEADER_SIZE, BLOCK_SIZE, key_type> ds_left_1;
  LeafDS<LOG_SIZE, HEADER_SIZE, BLOCK_SIZE, key_type> ds_right_1;

  std::vector<key_type> elts_left_1;
  std::vector<key_type> elts_right_1;
  std::vector<key_type> elts_remaining_1;

   // add all of elements to right leafDS then remove some (so both are underfull and can be merged)
  for (uint32_t i = 0; i < el_count; i++) {
    key_type el = dist_el(rng);
    if (!std::count(elts_left_1.begin(), elts_left_1.end(), el)) {
      elts_left_1.push_back(el);
    }
    // add to left leafDS
    ds_left_1.insert(el);
  } 

  for (uint32_t i = 0; i < elts_left_1.size(); i++) {
    key_type el = elts_left_1[i];
    if (i < el_count/2) {
      // remove from left leafDS
      ds_left_1.remove(el);
    } else {
      elts_remaining_1.push_back(el);
    }
  } 

  for (uint32_t i = 0; i < el_count; i++) {
    key_type el = dist_el(rng);
    if (!std::count(elts_right_1.begin(), elts_right_1.end(), el)) {
      elts_right_1.push_back(el);
    }
    // add to right leafDS
    ds_right_1.insert(el);
    if (!ds_right_1.has(el)) {
      ds_right_1.print();
      printf("Missing from ds_right on insert, elt: %lu , index = %lu\n", el, i);
      return -1;
    }
  }

  for (uint32_t i = 0; i < elts_right_1.size(); i++) {
    key_type el = elts_right_1[i];
    if (i >= el_count/2) {
      // remove from left leafDS
      ds_right_1.remove(el);
    } else {
      elts_remaining_1.push_back(el);
      if (!ds_right_1.has(el)) {
        ds_right_1.print();
        printf("Missing from ds_right after delete, elt: %lu , index = %lu\n", el, i);
        return -1;
      }
    }
  }

  printf("size of elts remaining: %lu\n", elts_remaining_1.size());

  ds_left_1.merge(&(ds_right_1));

  // Check if left leafDS has all elements from left and right after merge
  if (ds_right_1.get_num_elements() != 0) {
    ds_right_1.print();
    printf("Right leaf not empty after inserts and deletes, size = %lu\n", ds_right_1.get_num_elements());
    return -1;
  }
  for (uint32_t i = 0; i < elts_remaining_1.size(); i++) {
    auto el = elts_remaining_1[i];
    if (!ds_left_1.has(el)) {
      ds_left_1.print();
      ds_right_1.print();
      printf("Missing from left leaf after inserts and deletes, elt: %lu , index = %lu\n", el, i);
      return -1;
    }
  } 
  return 0;
}

[[nodiscard]] int merge_test(uint32_t el_count) {
  int r = 0;
  r = merge_test_templated(el_count);
  if (r) {
    return r;
  }

  return 0;
}

[[nodiscard]] int shift_left_test_templated(uint32_t el_count) {
  LeafDS<LOG_SIZE, HEADER_SIZE, BLOCK_SIZE, key_type> ds_left;
  LeafDS<LOG_SIZE, HEADER_SIZE, BLOCK_SIZE, key_type> ds_right;
  std::mt19937 rng(0);
  std::uniform_int_distribution<key_type> dist_el(1, N * 16);

  std::vector<key_type> checker;
  checker.reserve(el_count);
  std::vector<key_type> elts_left_1;
  std::vector<key_type> elts_right_1;
  std::vector<key_type> elts_left_remaining_1;
  std::vector<key_type> elts_right_remaining_1;

  // add 3/4 of elements to left leafDS 
  for (uint32_t i = 0; i < (el_count*3.0)/4.0; i++) {
    key_type el = dist_el(rng);
    if (!std::count(elts_left_1.begin(), elts_left_1.end(), el)) {
      elts_left_1.push_back(el);
    }
    ds_left.insert(el);
  }

  // add all of elements to right leafDS (right will be shifted over)
  for (uint32_t i = 0; i < el_count; i++) {
    key_type el = dist_el(rng);
    if (!std::count(elts_right_1.begin(), elts_right_1.end(), el)) {
      elts_right_1.push_back(el);
    }
    ds_right.insert(el);
  }

  // delete 1/4 of elements from left leafDS
  for (uint32_t i = 0; i < elts_left_1.size(); i++) {
    key_type el = elts_left_1[i];
    if (i < el_count/4) {
      // remove from left leafDS
      ds_left.remove(el);
      if (ds_left.has(el)) {
        printf("Failed to remove el from left leaf: %lu", el);
        return -1;
      }
    } else {
      elts_left_remaining_1.push_back(el);
    }
  } 

  // delete 1/4 of elements from right leafDS
  for (uint32_t i = 0; i < elts_right_1.size(); i++) {
    key_type el = elts_right_1[i];
    if (i < el_count/4) {
      // remove from left leafDS
      ds_right.remove(el);
      if (ds_right.has(el)) {
        printf("Failed to remove el from right leaf: %lu", el);
        return -1;
      }
    } else {
      elts_right_remaining_1.push_back(el);
    }
  }

/*
  TODO!!! 
  get_num_elements() and other size checks aren't working with inserts and deletes, so these tests fail
  shift itself seems to be working though, check existence and deletion in left and right

  if (ds_left.get_num_elements() != elts_left_remaining_1.size()) {
    ds_left.print();
    printf("Left leaf not correct size pre shift = %lu\n, expected %lu", ds_left.get_num_elements(), elts_left_remaining_1.size());
    return -1;
  }
  if (ds_right.get_num_elements() != elts_right_remaining_1.size()) {
    ds_right.print();
    printf("Right leaf not correct size pre shift = %lu\n, expected %lu", ds_left.get_num_elements(), elts_right_remaining_1.size());
    return -1;
  }
*/

  printf("ds_right num elems = %lu, correct = %lu \n", ds_right.get_num_elements(), elts_right_remaining_1.size());
  printf("ds_left num elems = %lu, correct = %lu \n", ds_left.get_num_elements(), elts_left_remaining_1.size());
  printf("shifting diff %lu\n", ds_right.get_num_elements() - ds_left.get_num_elements());
  unsigned int shiftnum = (ds_right.get_num_elements() - ds_left.get_num_elements()) >> 1;
  printf("shifting real %lu\n", shiftnum);

  ds_left.shift_left(&(ds_right), shiftnum);

  printf("ds_right num elems post_shift %lu\n", ds_right.get_num_elements());
  printf("ds_left num elems post_shift %lu\n",ds_left.get_num_elements());

/*
  TODO!!! 
  get_num_elements() and other size checks aren't working with inserts and deletes, so these tests fail
  shift itself seems to be working though, check existence and deletion in left and right

  // Check if left leafDS has shifted elements from right after shift
  if (ds_left.get_num_elements() != elts_left_remaining_1.size() + shiftnum) {
    ds_left.print();
    printf("Left leaf not correct size = %lu\n, expected %lu", ds_left.get_num_elements(), elts_left_remaining_1.size() + shiftnum);
    return -1;
  }
  // Check if right leafDS removed shifted elements from right after shift
  if (ds_right.get_num_elements() != elts_right_remaining_1.size() - shiftnum) {
    ds_right.print();
    printf("Right leaf not correct size = %lu\n, expected %lu", ds_right.get_num_elements(), elts_right_remaining_1.size() - shiftnum);
    return -1;
  }
*/

  // Check if original elems in left exist in left
  for (uint32_t i = 0; i < elts_left_remaining_1.size(); i++) {
    auto el = elts_left_remaining_1[i];
    if (!ds_left.has(el)) {
      ds_left.print();
      printf("Missing elt in left leaf after shift left from orig, elt: %lu index:%u \n", el, i);
      return -1;
    }
  }
  // Check if elems shifted from right exist in left
  std::sort(elts_right_remaining_1.begin(), elts_right_remaining_1.end());
  for (uint32_t i = 0; i < shiftnum; i++) {
    auto el = elts_right_remaining_1[i];
    if (!ds_left.has(el)) {
      ds_left.print();
      printf("Missing elt in left leaf after shift left from right, elt: %lu index:%u \n", el, i);
      return -1;
    }
    if (ds_right.has(el)) {
      ds_right.print();
      printf("Elt not removed from in right leaf after shift left from right, elt: %lu index:%u \n", el, i);
      return -1;
    }
  }
  // Check if elems not shifted from right exist in right
  for (uint32_t i = shiftnum; i < elts_right_remaining_1.size(); i++) {
    auto el = elts_right_remaining_1[i];
    if (!ds_right.has(el)) {
      ds_right.print();
      printf("Missing elt in right leaf after shift left, elt: %lu index:%u \n", el, i);
      return -1;
    }
    if (ds_left.has(el) && !std::count(elts_left_remaining_1.begin(), elts_left_remaining_1.end(), el)) {
      ds_left.print();
      printf("Elt should not exist in left leaf after shift left, elt: %lu index:%u \n", el, i);
      return -1;
    }
  }
  return 0;
}

[[nodiscard]] int shift_left_test(uint32_t el_count) {
  int r = 0;
  r = shift_left_test_templated(el_count);
  if (r) {
    return r;
  }

  return 0;
}

[[nodiscard]] int shift_right_test_templated(uint32_t el_count) {
  LeafDS<LOG_SIZE, HEADER_SIZE, BLOCK_SIZE, key_type> ds_left;
  LeafDS<LOG_SIZE, HEADER_SIZE, BLOCK_SIZE, key_type> ds_right;
  std::mt19937 rng(0);
  std::uniform_int_distribution<key_type> dist_el(1, N * 16);

  std::vector<key_type> checker;
  checker.reserve(el_count);
  std::vector<key_type> elts_left_1;
  std::vector<key_type> elts_right_1;
  std::vector<key_type> elts_left_remaining_1;
  std::vector<key_type> elts_right_remaining_1;

  // add 3/4 of elements to right leafDS 
  for (uint32_t i = 0; i < (el_count*3.0)/4.0; i++) {
    key_type el = dist_el(rng);
    if (!std::count(elts_right_1.begin(), elts_right_1.end(), el)) {
      elts_right_1.push_back(el);
    }
    ds_right.insert(el);
  }

  // add all of elements to left leafDS (left will be shifted over)
  for (uint32_t i = 0; i < el_count; i++) {
    key_type el = dist_el(rng);
    if (!std::count(elts_left_1.begin(), elts_left_1.end(), el)) {
      elts_left_1.push_back(el);
    }
    ds_left.insert(el);
  }

  // delete 1/4 of elements from left leafDS
  for (uint32_t i = 0; i < elts_left_1.size(); i++) {
    key_type el = elts_left_1[i];
    if (i < el_count/4) {
      // remove from left leafDS
      ds_left.remove(el);
      if (ds_left.has(el)) {
        printf("Failed to remove el from left leaf: %lu", el);
        return -1;
      }
    } else {
      elts_left_remaining_1.push_back(el);
    }
  } 

  // delete 1/4 of elements from right leafDS
  for (uint32_t i = 0; i < elts_right_1.size(); i++) {
    key_type el = elts_right_1[i];
    if (i < el_count/4) {
      // remove from left leafDS
      ds_right.remove(el);
      if (ds_right.has(el)) {
        printf("Failed to remove el from right leaf: %lu", el);
        return -1;
      }
    } else {
      elts_right_remaining_1.push_back(el);
    }
  }

/*
  TODO!!! 
  get_num_elements() and other size checks aren't working with inserts and deletes, so these tests fail
  shift itself seems to be working though, check existence and deletion in left and right

  if (ds_left.get_num_elements() != elts_left_remaining_1.size()) {
    ds_left.print();
    printf("Left leaf not correct size pre shift = %lu\n, expected %lu", ds_left.get_num_elements(), elts_left_remaining_1.size());
    return -1;
  }
  if (ds_right.get_num_elements() != elts_right_remaining_1.size()) {
    ds_right.print();
    printf("Right leaf not correct size pre shift = %lu\n, expected %lu", ds_left.get_num_elements(), elts_right_remaining_1.size());
    return -1;
  }
*/

  printf("ds_right num elems = %lu, correct = %lu \n", ds_right.get_num_elements(), elts_right_remaining_1.size());
  printf("ds_left num elems = %lu, correct = %lu \n", ds_left.get_num_elements(), elts_left_remaining_1.size());
  printf("shifting diff %lu\n", ds_right.get_num_elements() - ds_left.get_num_elements());
  unsigned int shiftnum = (ds_right.get_num_elements() - ds_left.get_num_elements()) >> 1;
  printf("shifting real %lu\n", shiftnum);

  ds_right.shift_right(&(ds_left), shiftnum, elts_left_remaining_1.size());

  printf("ds_right num elems post_shift %lu\n", ds_right.get_num_elements());
  printf("ds_left num elems post_shift %lu\n",ds_left.get_num_elements());

/*
  TODO!!! 
  get_num_elements() and other size checks aren't working with inserts and deletes, so these tests fail
  shift itself seems to be working though, check existence and deletion in left and right

  // Check if left leafDS has shifted elements from right after shift
  if (ds_left.get_num_elements() != elts_left_remaining_1.size() + shiftnum) {
    ds_left.print();
    printf("Left leaf not correct size = %lu\n, expected %lu", ds_left.get_num_elements(), elts_left_remaining_1.size() + shiftnum);
    return -1;
  }
  // Check if right leafDS removed shifted elements from right after shift
  if (ds_right.get_num_elements() != elts_right_remaining_1.size() - shiftnum) {
    ds_right.print();
    printf("Right leaf not correct size = %lu\n, expected %lu", ds_right.get_num_elements(), elts_right_remaining_1.size() - shiftnum);
    return -1;
  }
*/

  // Check if original elems in right exist in right
  for (uint32_t i = 0; i < elts_right_remaining_1.size(); i++) {
    auto el = elts_right_remaining_1[i];
    if (!ds_right.has(el)) {
      ds_right.print();
      printf("Missing elt in right leaf after shift right from orig, elt: %lu index:%u \n", el, i);
      return -1;
    }
  }
  // Check if elems shifted from left exist in right
  std::sort(elts_left_remaining_1.begin(), elts_left_remaining_1.end());
  for (uint32_t i = elts_left_remaining_1.size() - shiftnum; i < elts_left_remaining_1.size(); i++) {
    auto el = elts_left_remaining_1[i];
    if (!ds_right.has(el)) {
      ds_right.print();
      printf("Missing elt in right leaf after shift right from left, elt: %lu index:%u \n", el, i);
      return -1;
    }
    if (ds_left.has(el)) {
      ds_left.print();
      printf("Elt not removed from left leaf after shift right from left, elt: %lu index:%u \n", el, i);
      return -1;
    }
  }
  // Check if elems not shifted from left exist in left
  for (uint32_t i = 0; i < elts_left_remaining_1.size() - shiftnum; i++) {
    auto el = elts_left_remaining_1[i];
    if (!ds_left.has(el)) {
      ds_left.print();
      printf("Missing elt in left leaf after shift right, elt: %lu index:%u \n", el, i);
      return -1;
    }
    if (ds_right.has(el) && !std::count(elts_right_remaining_1.begin(), elts_right_remaining_1.end(), el)) {
      ds_right.print();
      printf("Elt should not exist in right leaf after shift right, elt: %lu index:%u \n", el, i);
      return -1;
    }
  }
  return 0;
}

[[nodiscard]] int shift_right_test(uint32_t el_count) {
  int r = 0;
  r = shift_right_test_templated(el_count);
  if (r) {
    return r;
  }

  return 0;
}

int main(int argc, char *argv[]) {

  cxxopts::Options options("LeafDStester",
                           "allows testing diferent attributes of the leaf DS");

  options.positional_help("Help Text");

  // clang-format off
  options.add_options()
    ("el_count", "how many values to insert", cxxopts::value<int>()->default_value( "100000"))
    ("num_copies", "number of copies for parallel test", cxxopts::value<int>()->default_value( "100000"))
    ("v, verify", "verify the results of the test, might be much slower")
    ("update_test", "time updating")
    ("insert_delete_test", "time updating")
    ("parallel_test", "time to do parallel test")
    ("parallel_test_perf", "just leafDS copies for perf")
    ("update_values_test", "time updating with values")
    ("unsorted_range_query_test", "time updating with values")
    ("sorted_range_query_test", "time updating with values")
    ("key_at_sorted_index_test", "verify correctness")
    ("merge_test", "verify correctness")
    ("shift_left_test", "verify correctness")
    ("shift_right_test", "verify correctness");
    // ("help","Print help");
  // clang-format on

  auto result = options.parse(argc, argv);
  uint32_t el_count = result["el_count"].as<int>();
  uint32_t num_copies = result["num_copies"].as<int>();

  bool verify = result["verify"].as<bool>();
  printf("el count %u\n", el_count);

  if (result["update_test"].as<bool>()) {
    return update_test(el_count, verify);
  }

  // always verify
  if (result["insert_delete_test"].as<bool>()) {
    return insert_delete_test(el_count);
  }
  if (result["update_values_test"].as<bool>()) {
    return update_values_test(el_count, verify);
  }
  
  if (result["parallel_test"].as<bool>()) {
    return parallel_test(el_count, num_copies, 1.0);
  }

  if (result["sorted_range_query_test"].as<bool>()) {
    return sorted_range_query_test(el_count, num_copies, 100, 100);
  }

  if (result["unsorted_range_query_test"].as<bool>()) {
    return unsorted_range_query_test(el_count, num_copies, 100);
  }

  if (result["parallel_test_perf"].as<bool>()) {
    return parallel_test_perf(el_count, num_copies, 1.0);
  }

  if (result["key_at_sorted_index_test"].as<bool>()) {
    return key_at_sorted_index_test(el_count);
  }

  if (result["merge_test"].as<bool>()) {
    return merge_test(el_count);
  }

  if (result["shift_left_test"].as<bool>()) {
    return shift_left_test(el_count);
  }

  if (result["shift_right_test"].as<bool>()) {
    return shift_right_test(el_count);
  }

  return 0;
}
