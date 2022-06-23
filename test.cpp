// main test driver for leafDS
#define DEBUG 0

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

#define LOG_SIZE 32
#define HEADER_SIZE 32
#define BLOCK_SIZE 32

#define N LOG_SIZE + HEADER_SIZE + BLOCK_SIZE * HEADER_SIZE

[[nodiscard]] int parallel_test_leafDS(uint32_t el_count, uint32_t num_copies) {
	std::vector<LeafDS<LOG_SIZE, HEADER_SIZE, BLOCK_SIZE, uint32_t>> dsv(num_copies);
	uint64_t start = get_usecs();
	cilk_for(uint32_t i = 0; i < num_copies; i++) {
	  std::mt19937 rng(0);
  	std::uniform_int_distribution<uint32_t> dist_el(1, N * 16);
		for (uint32_t j = 0; j < el_count; j++) {
			uint32_t el = dist_el(rng);
			dsv[i].insert(el);
		}
	}
	uint64_t end = get_usecs();
	printf("LeafDS: parallel insert time for %u copies of %u elts each = %lu us\n", num_copies, el_count, end - start);


	std::vector<uint64_t> partial_sums(getWorkers() * 8);
	start = get_usecs();

	cilk_for(uint32_t i = 0; i < num_copies; i++) {
		partial_sums[getWorkerNum() * 8] += dsv[i].sum_keys();
	}
	uint64_t count{0};
	for(int i = 0; i < getWorkers(); i++) {
		count += partial_sums[i*8];
	}
	end = get_usecs();
	printf("LeafDS: parallel sum time for %u copies of %u elts each = %lu us\n", num_copies, el_count, end - start);
	printf("total sum %lu\n", count);

	return 0;
}

[[nodiscard]] int parallel_test_sorted_vector(uint32_t el_count, uint32_t num_copies) {
	std::vector<std::vector<uint32_t>> dsv(num_copies);
	uint64_t start = get_usecs();
	cilk_for(uint32_t i = 0; i < num_copies; i++) {
	  std::mt19937 rng(0);
  	std::uniform_int_distribution<uint32_t> dist_el(1, N * 16);
		for (uint32_t j = 0; j < el_count; j++) {
			uint32_t el = dist_el(rng);

			// find the elt at most the thing to insert
			size_t idx = 0;
			for(; idx < dsv[i].size(); idx++) {
				if(dsv[i][idx] == el) {
					break;
				} else if (dsv[i][idx] > el) {
					break;
				}
			}

			// vector does before
			if(dsv[i].size() == 0 || dsv[i][idx] != el) {
				dsv[i].insert(dsv[i].begin() + idx, el);
			}

#if DEBUG
			// test sortedness
			for(idx = 1; idx < dsv[i].size(); idx++) {
				assert(dsv[i][idx] > dsv[i][idx-1]);
			}
#endif
		}
	}
	uint64_t end = get_usecs();
	printf("Sorted vector: parallel insert time for %u copies of %u elts each = %lu us\n", num_copies, el_count, end - start);

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
	printf("Sorted vector: parallel sum time for %u copies of %u elts each = %lu us\n", num_copies, el_count, end - start);
	printf("total sum %lu\n", count);

	return 0;
}

[[nodiscard]] int parallel_test_unsorted_vector(uint32_t el_count, uint32_t num_copies) {
	std::vector<std::vector<uint32_t>> dsv(num_copies);
	uint64_t start = get_usecs();
	cilk_for(uint32_t i = 0; i < num_copies; i++) {
	  std::mt19937 rng(0);
  	std::uniform_int_distribution<uint32_t> dist_el(1, N * 16);
		for (uint32_t j = 0; j < el_count; j++) {
			uint32_t el = dist_el(rng);

			// find the elt at most the thing to insert
			size_t idx = 0;
			for(; idx < dsv[i].size(); idx++) {
				if(dsv[i][idx] == el) {
					break;
				} 
			}

			// if not found, add it to the end
			if(idx == dsv[i].size()) {
				dsv[i].push_back(el);
			}
		}
	}
	uint64_t end = get_usecs();
	printf("Unsorted vector: parallel insert time for %u copies of %u elts each = %lu us\n", num_copies, el_count, end - start);

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
	printf("Unsorted vector: parallel sum time for %u copies of %u elts each = %lu us\n", num_copies, el_count, end - start);
	printf("total sum %lu\n", count);

	return 0;
}


[[nodiscard]] int parallel_test(uint32_t el_count, uint32_t num_copies) {
	printf("doing parallel test\n");
	int r = parallel_test_leafDS(el_count, num_copies);
	if (r) {
		return r;
	}

	r = parallel_test_sorted_vector(el_count, num_copies);
	if (r) {
		return r;
	}

	r = parallel_test_unsorted_vector(el_count, num_copies);
	if (r) {
		return r;
	}
	return 0;
}

[[nodiscard]] int update_test_templated(uint32_t el_count,
                                            bool check = false) {

  LeafDS<LOG_SIZE, HEADER_SIZE, BLOCK_SIZE, uint32_t> ds;
  std::mt19937 rng(0);
  std::uniform_int_distribution<uint32_t> dist_el(1, N * 16);
  std::uniform_real_distribution<double> dist_flip(.25, .75);

  std::unordered_set<uint32_t> checker;

	for (uint32_t i = 0; i < el_count; i++) {
    // be more likely to insert when we are more empty
    uint32_t el = dist_el(rng);
		// printf("inserting %u, num elts so far %lu\n", el, ds.get_num_elts());
		// if (ds.has(el)) {
	  //	printf("\t DS already has %u\n", el);
		// }
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
			for(auto elt : checker) {
				if(!ds.has(elt)) {
					ds.print();
					printf("don't have something, %u, we inserted while inserting "
								 "elements\n",
								 el);
					return -1;
				}
			}
		}
	}


	uint64_t sum = ds.sum_keys();

	if (check) {
		uint64_t correct_sum = 0;
		for (auto elt : checker) {
			correct_sum += elt;
		}
		if (correct_sum != sum) {
			ds.print();
			tbassert(correct_sum == sum, "got sum %lu, should be %lu\n", sum, correct_sum);
		}
	}
	printf("got sum %lu\n", sum);
/*
  uint64_t start = get_usecs();
  for (uint32_t i = 0; i < el_count; i++) {
    // be more likely to insert when we are more empty
    uint32_t el = dist_el(rng);
    // printf("inserting %u\n", el);

    if (dist_flip(rng) < ((double)(N - ds.get_num_elts()) / N)) {
      // printf("inserting %u\n", el);
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
      // printf("removing %u\n", el);
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
    if (check) {
      if (ds.get_num_elts() != checker.size()) {
        printf("wrong number of elements\n");
        return -1;
      }
      for (auto el : checker) {
        if (!ds.has(el)) {
          ds.print_pma();
          printf("we removed %u when we shouldn't have\n", el);
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
    }
    // ds.print_pma();
  }
  uint64_t end = get_usecs();
  printf("took %lu micros\n", end - start);
*/
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
		("parallel_test", "time to do parallel test")
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

  if (result["update_values_test"].as<bool>()) {
    return update_values_test(el_count, verify);
  }
	
	if (result["parallel_test"].as<bool>()) {
		printf("doing parallel test\n");
		return parallel_test(el_count, num_copies);
	}
  return 0;
}
