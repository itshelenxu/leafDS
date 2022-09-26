// main test driver for leafDS
#define DEBUG_PRINT 0

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

#define HEADER_SIZE 64
#define LOG_SIZE HEADER_SIZE
#define BLOCK_SIZE 32
#define NUM_TRIALS 5
#define MEDIAN_TRIAL NUM_TRIALS / 2
#define N LOG_SIZE + HEADER_SIZE + BLOCK_SIZE * HEADER_SIZE

#define key_type uint64_t

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

[[nodiscard]] int parallel_test_perf(uint32_t el_count, uint32_t num_copies, double prob_insert) {
	int r = parallel_test_leafDS(el_count, num_copies, prob_insert);
	if (r) {
		return r;
	}
	printf("\n");
	
	return 0;
}
[[nodiscard]] int insert_delete_templated(uint32_t el_count,
                                            bool check = false) {

  LeafDS<LOG_SIZE, HEADER_SIZE, BLOCK_SIZE, key_type> ds;
  std::mt19937 rng(0);
  std::uniform_int_distribution<key_type> dist_el(1, N * 16);

  std::unordered_set<key_type> checker;
	std::vector<key_type> elts;

	// add some elements
	for (uint32_t i = 0; i < el_count; i++) {
    key_type el = dist_el(rng);
		elts.push_back(el);
		ds.insert(el);
		if (check) {
			checker.insert(el);
			if (!ds.has(el)) {
				ds.print();
				printf("don't have something, %u, we inserted while inserting "
							 "elements\n",
							 el);
				return -1;
			}
		}
	} 

	// then remove all the stuff we added
	for (auto el : elts) {
		ds.remove(el);
		if (check) {
			checker.erase(el);
			if (ds.has(el)) {
				ds.print();
				printf("has %u but should have deleted\n", el);
				return -1;
			}
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
	}
	printf("got sum %lu\n", sum);
	printf("got sum direct %lu\n", sum_direct);
  return 0;
}


[[nodiscard]] int insert_delete_test(uint32_t el_count, bool check = false) {
  int r = 0;
  r = insert_delete_templated(el_count, check);
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
			printf("\ninserting %u\n", el);
			ds.insert(el);
			if (check) {
				checker.insert(el);
				if (!ds.has(el)) {
					ds.print();
					printf("don't have something, %u, we inserted while inserting "
								 "elements\n",
								 el);
					return -1;
				}
			}
		} else {
			bool present = ds.has(el);
			if(present) { printf("removing elt %u in DS\n", el); }
			else {
				printf("removing elt %u not in DS\n", el);
			}

      ds.remove(el);
      if (check) {
        checker.erase(el);
        if (ds.has(el)) {
					ds.print();
          printf("have something we removed while removing elements, tried to "
                 "remove %u\n",
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
				printf("missing %u\n", elt);
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
    ("update_values_test", "time updating with values");
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

	if (result["insert_delete_test"].as<bool>()) {
    return insert_delete_test(el_count, verify);
  }
  if (result["update_values_test"].as<bool>()) {
    return update_values_test(el_count, verify);
  }
	
	if (result["parallel_test"].as<bool>()) {
		return parallel_test(el_count, num_copies, 1.0);
	}

	if (result["parallel_test_perf"].as<bool>()) {
		return parallel_test_perf(el_count, num_copies, 1.0);
	}

  return 0;
}
