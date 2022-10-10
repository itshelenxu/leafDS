#pragma once
#include "StructOfArrays/SizedInt.hpp"
#include "StructOfArrays/soa.hpp"
#include "helpers.hpp"
#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <tuple>
#include <type_traits>
#include <immintrin.h>
#include <typeinfo>


#if DEBUG==1
#define ASSERT(PREDICATE, ...)                                                 \
  do {                                                                         \
    if (!(PREDICATE)) {                                                        \
      fprintf(stderr,                                                          \
              "%s:%d (%s) Assertion " #PREDICATE " failed: ", __FILE__,        \
              __LINE__, __PRETTY_FUNCTION__);                                  \
      fprintf(stderr, __VA_ARGS__);                                            \
      abort();                                                                 \
    }                                                                          \
  } while (0)
#else
#define ASSERT(PREDICATE, ...)                                                 
#endif

#define STATS 0
#define DEBUG_PRINT 0

static uint64_t one[128] = {
  1ULL << 0, 1ULL << 1, 1ULL << 2, 1ULL << 3, 1ULL << 4, 1ULL << 5, 1ULL << 6, 1ULL << 7, 1ULL << 8, 1ULL << 9,
  1ULL << 10, 1ULL << 11, 1ULL << 12, 1ULL << 13, 1ULL << 14, 1ULL << 15, 1ULL << 16, 1ULL << 17, 1ULL << 18, 1ULL << 19, 
  1ULL << 20, 1ULL << 21, 1ULL << 22, 1ULL << 23, 1ULL << 24, 1ULL << 25, 1ULL << 26, 1ULL << 27, 1ULL << 28, 1ULL << 29, 
  1ULL << 30, 1ULL << 31, 1ULL << 32, 1ULL << 33, 1ULL << 34, 1ULL << 35, 1ULL << 36, 1ULL << 37, 1ULL << 38, 1ULL << 39, 
  1ULL << 40, 1ULL << 41, 1ULL << 42, 1ULL << 43, 1ULL << 44, 1ULL << 45, 1ULL << 46, 1ULL << 47, 1ULL << 48, 1ULL << 49, 
  1ULL << 50, 1ULL << 51, 1ULL << 52, 1ULL << 53, 1ULL << 54, 1ULL << 55, 1ULL << 56, 1ULL << 57, 1ULL << 58, 1ULL << 59, 
  1ULL << 60, 1ULL << 61, 1ULL << 62, 1ULL << 63, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };


enum range_type {INSERTS, DELETES, BLOCK};

template <size_t log_size, size_t header_size, size_t block_size, typename key_type, typename... Ts>
class LeafDS {

  static constexpr bool binary = (sizeof...(Ts) == 0);

  using element_type =
      typename std::conditional<binary, std::tuple<key_type>,
                                std::tuple<key_type, Ts...>>::type;

  using value_type =
      typename std::conditional<binary, std::tuple<>, std::tuple<Ts...>>::type;

  template <int I>
  using NthType = typename std::tuple_element<I, value_type>::type;
  static constexpr int num_types = sizeof...(Ts);

  using SOA_type = typename std::conditional<binary, SOA<key_type>,
                                             SOA<key_type, Ts...>>::type;

#if AVX512
	static constexpr size_t keys_per_vector = 64 / sizeof(key_type);
	
	static constexpr uint32_t all_ones_vec = keys_per_vector - 1;

	using mask_type = 
		typename std::conditional<sizeof(key_type) == 4, __mmask16, __mmask8>::type;

	// idk why it complained about this one
	using lr_mask_type = 
		typename std::conditional<sizeof(key_type) == 4, uint16_t, uint8_t>::type;

	// TODO: can precompute the masks
	lr_mask_type get_left_mask(size_t start) {
		assert(start < keys_per_vector);
#if DEBUG_PRINT
		printf("\tget left mask start = %lu\n", start);
#endif
		lr_mask_type mask;
		if constexpr (std::is_same<key_type, uint32_t>::value) { // 32-bit keys
			mask = 0xFFFF;
		} else {
			mask = 0xFF;
		}
		mask <<= start;
		mask >>= start;
		return mask;
	}

	// TODO: can precompute the masks
	auto get_right_mask(size_t end) {
		assert(end < keys_per_vector);
#if DEBUG_PRINT
		printf("\tget right mask end = %lu\n", end);
#endif
		lr_mask_type mask;
		if constexpr (std::is_same<key_type, uint32_t>::value) { // 32-bit keys
			mask = 0xFFFF;
		} else {
			mask = 0xFF;
		}
		mask >>= end;
		mask <<= end;
		return mask;
	}


	// 64 bytes per cache line / 4 bytes per elt = 16-bit vector
	// wherever there is a match, it will set those bits
	static inline __mmask16 slot_mask_32(uint8_t * array, uint32_t key) {
		__m512i bcast = _mm512_set1_epi32(key);
		__m512i block = _mm512_loadu_si512((const __m512i *)(array));
		return _mm512_cmp_epu32_mask(bcast, block, _MM_CMPINT_EQ);
	}

	static inline __mmask8 slot_mask_64(uint8_t * array, uint64_t key) {
		__m512i bcast = _mm512_set1_epi64(key);
		__m512i block = _mm512_loadu_si512((const __m512i *)(array));
		return _mm512_cmp_epu64_mask(bcast, block, _MM_CMPINT_EQ);
	}

	static inline mask_type slot_mask(uint8_t * array, key_type key) {
		if constexpr (std::is_same<key_type, uint32_t>::value) { // 32-bit keys
			return slot_mask_32(array, key);
		} else {
			return slot_mask_64(array, key);
		}
	}
#endif

private:
	static constexpr key_type NULL_VAL = {};

	static constexpr size_t num_blocks = header_size;
	static constexpr size_t N = log_size + header_size + block_size * num_blocks;

  static_assert(N != 0, "N cannot be 0");

	// start of each section
	// insert and delete log
	static constexpr size_t header_start = log_size;
	static constexpr size_t blocks_start = header_start + header_size;

	// counters
	size_t num_inserts_in_log = 0;
	size_t num_deletes_in_log = 0;
	size_t num_elts_total = 0;

	// max elts allowed in data structure before split
	static constexpr size_t max_density = (int)( 9.0 / 10.0 * N );
	static constexpr size_t min_density = (int)( 4.0 / 10.0 * N );

public:
	// TODO: NOT SAFE!! insert and delete break this because they don't increment num_elts_total until they flush
	// size_t get_num_elements() const { return num_elts_total; }

	std::array<uint8_t, SOA_type::get_size_static(N)> array = {0};

	key_type get_min_after_split() {
		return blind_read_key(header_start);
	}

private:

  inline key_type get_key_array(uint32_t index) const {
    return std::get<0>(
        SOA_type::template get_static<0>(array.data(), N, index));
  }

	size_t count_up_elts() const {
		size_t result = 0;
		for(size_t i = 0; i < N; i++) {
			result += (blind_read_key(i) != NULL_VAL);
		}
		return result;
	}

#if STATS
  uint32_t num_redistributes = 0;
  uint32_t vol_redistributes = 0;
public:
  void report_redistributes() {
    printf("num redistributes = %u, vol redistributes %u\n", num_redistributes, vol_redistributes);
  }
#endif

public:
  key_type split(LeafDS<log_size, header_size, block_size, key_type, Ts...>* right);
  void merge(LeafDS<log_size, header_size, block_size, key_type, Ts...>* right);
  void get_max_2(key_type* max_key, key_type* second_max_key, int manual_num_elts);
  void shift_left(LeafDS<log_size, header_size, block_size, key_type, Ts...>* right, int shiftnum);
  void shift_right(LeafDS<log_size, header_size, block_size, key_type, Ts...>* left, int shiftnum, int left_num_elts);
  key_type& get_key_at_sorted_index(size_t i);
  size_t get_element_at_sorted_index(size_t i);
  size_t get_num_elements();

private:
	// private helpers
	void update_val_at_index(element_type e, size_t index);
	void place_elt_at_index(element_type e, size_t index);
	void clear_range(size_t start, size_t end);
	size_t find_block(key_type key) const;
	size_t find_block_with_hint(key_type key, size_t hint) const;
	void global_redistribute(element_type* log_to_flush, size_t num_to_flush, unsigned short* count_per_block);
	void global_redistribute_blocks(unsigned short* count_per_block);

	void copy_src_to_dest(size_t src, size_t dest);
	void flush_log_to_blocks(size_t max_to_flush);
	void flush_deletes_to_blocks();

	void sort_log();
	void sort_range(size_t start_idx, size_t end_idx);
	
	// helpers for range query
	void sort_array_range(void *base_array, size_t size, size_t start_idx, size_t end_idx);
	inline bool is_in_delete_log(key_type key);
	auto get_sorted_block_copy(size_t block_idx);


	inline std::pair<size_t, size_t> get_block_range(size_t block_idx) const;
	
	template <range_type type>
	bool update_in_range_if_exists(size_t start, size_t end, element_type e);
	std::pair<bool, size_t> find_key_in_range(size_t start, size_t end, key_type e) const;
	bool update_in_block_if_exists(element_type e);
	bool update_in_block_if_exists(element_type e, size_t block_idx);

	unsigned short count_block(size_t block_idx) const;
	void print_range(size_t start, size_t end) const;
	
	void advance_block_ptr(size_t* blocks_ptr, size_t* cur_block, size_t* start_of_cur_block, unsigned short* count_per_block) const;

	// delete helpers
	bool delete_from_header();
	void strip_deletes_and_redistrib();
	void delete_from_block_if_exists(key_type e, size_t block_idx);

	// given a buffer of n elts, spread them evenly in the blocks
	void global_redistribute_buffer(element_type* buffer, size_t n);



public:
  size_t get_num_elts() const { return num_elts_total; }
  bool is_full() const { return num_elts_total >= max_density; }
  bool is_few() const { return num_elts_total <= min_density; }

  void print() const;

  [[nodiscard]] uint64_t sum_keys() const;
  [[nodiscard]] uint64_t sum_keys_with_map() const;
  [[nodiscard]] uint64_t sum_keys_direct() const;
  template <bool no_early_exit, size_t... Is, class F> bool map(F f) const;
  
  // main top-level functions
  // given a key, return the index of the largest elt at least e
  [[nodiscard]] uint32_t search(key_type e) const;

  // insert e, return true if it was not there
  bool insert(element_type e);

  // remove e, return true if it was present
  bool remove(key_type e);

  // whether elt e was in the DS
  [[nodiscard]] bool has(key_type e) const;
  [[nodiscard]] bool has_with_print(key_type e) const;

  // return the next [length] sorted elts greater than or equal to start 
  auto sorted_range(key_type start, size_t length);

  // return all elts in the range [start, end]
  auto unsorted_range(key_type start, key_type end);

  [[nodiscard]] size_t get_index_in_blocks(key_type e) const;

  // index of element e in the DS, N if not found
  [[nodiscard]] size_t get_index(key_type e) const;

  auto blind_read_key(uint32_t index) const {
    return std::get<0>(SOA_type::get_static(array.data(), N, index));
  }

  // min block header is the first elt in the header part
  key_type get_min_block_key() const {
    return blind_read_key(header_start);
  }

  void blind_write_array(void* arr, size_t len, element_type e, uint32_t index) {
    SOA_type::get_static(arr, len, index) = e;
  }

  void blind_write(element_type e, uint32_t index) {
    SOA_type::get_static(array.data(), N, index) = e;
  }

  auto blind_read(uint32_t index) const {
    return SOA_type::get_static(array.data(), N, index);
  }
  auto blind_read_array(void* arr, size_t size, uint32_t index) const {
    return SOA_type::get_static(arr, size, index);
  }

  auto blind_read_key_array(void* arr, size_t size, uint32_t index) const {
    return std::get<0>(SOA_type::get_static(arr, size, index));
  }
};

// precondition - keys at index already match
template <size_t log_size, size_t header_size, size_t block_size, typename key_type, typename... Ts>
void LeafDS<log_size, header_size, block_size, key_type, Ts...>::update_val_at_index(element_type e, size_t index) {
	// update its value if needed
	if constexpr (!binary) {
		if (leftshift_tuple(SOA_type::get_static(array.data(), N, index)) !=
				leftshift_tuple(e)) {
			SOA_type::get_static(array.data(), N, index) = e;
		}
	}
}

// precondition - this slot is empty
template <size_t log_size, size_t header_size, size_t block_size, typename key_type, typename... Ts>
void LeafDS<log_size, header_size, block_size, key_type, Ts...>::place_elt_at_index(element_type e, size_t index) {
	SOA_type::get_static(array.data(), N, index) = e;
	num_elts_total++;
}

template <size_t log_size, size_t header_size, size_t block_size, typename key_type, typename... Ts>
void LeafDS<log_size, header_size, block_size, key_type, Ts...>::clear_range(size_t start, size_t end) {
  SOA_type::map_range_static(
      array.data(), N,
      [](auto &...args) { std::forward_as_tuple(args...) = element_type(); },
      start, end);
}


// given a merged list, put it in the DS
template <size_t log_size, size_t header_size, size_t block_size, typename key_type, typename... ts>
void LeafDS<log_size, header_size, block_size, key_type, ts...>::global_redistribute_buffer(element_type* buffer, size_t n) {
#if STATS
	num_redistributes++;
	vol_redistributes += N;
#endif

#if DEBUG
	assert(n < N);	
#endif
  clear_range(header_start, N); // clear the header and blocks

#if DEBUG_PRINT
	printf("GLOBAL REDISTRIB BUFFER OF SIZE %lu\n", n);
#endif

  // split up the buffer into blocks
  size_t per_block = n / num_blocks;
  size_t remainder = n % num_blocks;
  size_t num_so_far = 0;
  for(size_t i = 0; i < num_blocks; i++) {
    size_t num_to_flush = per_block + (i < remainder);
#if DEBUG
    ASSERT(num_to_flush < block_size, "to flush %lu, block size %lu\n", num_to_flush, block_size);
    assert(num_to_flush >= 1);
#endif
#if DEBUG_PRINT
		printf("block %zu, num to flush %zu\n", i, num_to_flush);
#endif
    // write the header
    blind_write(buffer[num_so_far], header_start + i);
#if DEBUG_PRINT
		printf("\tset buf[%zu] = %lu as header of block %zu at pos %lu\n", num_so_far, std::get<0>(buffer[num_so_far]), i, header_start + i);
#endif
    num_to_flush--;
    num_so_far++;
    // write the rest into block
    size_t start = blocks_start + i * block_size;
    for(size_t j = 0; j < num_to_flush; j++) {
#if DEBUG
			assert(num_so_far < n);
#endif
#if DEBUG_PRINT
		printf("\tset buf[%zu] = %lu in block %zu at pos %lu\n", num_so_far, std::get<0>(buffer[num_so_far]), i, start + j);
#endif
      blind_write(buffer[num_so_far], start + j);
      num_so_far++;
    }
  }
#if DEBUG
  assert(num_so_far == n);
#endif
	num_elts_total = num_so_far + num_inserts_in_log;

#if DEBUG_PRINT
	printf("after global redistrib blocks\n");
	print();
#endif
}

// just redistrib the header/blocks
template <size_t log_size, size_t header_size, size_t block_size, typename key_type, typename... ts>
void LeafDS<log_size, header_size, block_size, key_type, ts...>::global_redistribute_blocks(unsigned short* count_per_block) {

#if STATS
	num_redistributes++;
	vol_redistributes += N - header_size;
#endif

	// sort each block
	// size_t end_blocks = 0;
	for (size_t i = 0; i < num_blocks; i++) {
		auto block_range = get_block_range(i);
		sort_range(block_range.first, block_range.first + count_per_block[i]);
	}

	// merge all elts into sorted order
	std::vector<element_type> buffer;
	size_t blocks_ptr = header_start;
	size_t cur_block = 0;
	size_t start_of_cur_block = 0;
	while(cur_block < num_blocks) {
#if DEBUG
		ASSERT(blind_read_key(blocks_ptr) != NULL_VAL, "block ptr %lu\n", blocks_ptr);
#endif
#if DEBUG_PRINT
		printf("added %lu at idx %lu to buffer\n", blind_read_key(blocks_ptr), blocks_ptr);
#endif
		buffer.push_back(blind_read(blocks_ptr));
		advance_block_ptr(&blocks_ptr, &cur_block, &start_of_cur_block, count_per_block);
	}
#if DEBUG
	assert(buffer.size() < num_blocks * block_size);
#endif
	num_elts_total = buffer.size();

#if DEBUG_PRINT
	printf("*** BUFFER ***\n");
	for(size_t i = 0; i < buffer.size(); i++) {
		printf("%lu\t", std::get<0>(buffer[i]));
	}
	printf("\n");
#endif

#if DEBUG
	// at this point the buffer should be sorted
	for (size_t i = 1; i < buffer.size(); i++) {
		assert(has(std::get<0>(buffer[i])));
		ASSERT(std::get<0>(buffer[i]) > std::get<0>(buffer[i-1]), "buffer[%lu] = %lu, buffer[%lu] = %lu\n", i-1, std::get<0>(buffer[i-1]), i, std::get<0>(buffer[i]));
	}
#endif

	// split up the buffer evenly amongst the rest of the blocks
	global_redistribute_buffer(buffer.data(), buffer.size());
}


// one of the blocks
// input: deduped log to flush, number of elements in the log, count of elements to flush to each block, count of elements per block
// merge all elements from blocks and log in sorted order in the intermediate buffer
// split them evenly amongst the blocks
// TODO: count global redistributes 
template <size_t log_size, size_t header_size, size_t block_size, typename key_type, typename... ts>
void LeafDS<log_size, header_size, block_size, key_type, ts...>::global_redistribute(element_type* log_to_flush, size_t num_to_flush, unsigned short* count_per_block) {
#if DEBUG
	// verify that the log to flush is sorted
	for(size_t i = 1; i < num_to_flush; i++) {
		assert(std::get<0>(log_to_flush[i-1]) < std::get<0>(log_to_flush[i]));
	}
#endif
	// sort each block
	// size_t end_blocks = 0;
	for (size_t i = 0; i < num_blocks; i++) {
		auto block_range = get_block_range(i);
		sort_range(block_range.first, block_range.first + count_per_block[i]);
	}

	// do a merge from sorted log and sorted blocks
	std::vector<element_type> buffer;
	size_t log_ptr = 0; 
	size_t blocks_ptr = header_start;
	size_t cur_block = 0;
	size_t start_of_cur_block = 0;
	size_t log_end = num_to_flush;

#if DEBUG_PRINT
	printf("\n *** LOG TO FLUSH (size = %lu)***\n", num_to_flush);
	for(size_t i = 0; i < num_to_flush; i++) {
		printf("%lu\t", std::get<0>(log_to_flush[i]));
	}
	printf("\n\n\n");
	print();
#endif

	while(log_ptr < log_end && cur_block < num_blocks) {
		assert(blocks_ptr < N);
#if DEBUG_PRINT
		printf("log ptr %lu, cur_block %lu, blocks_ptr %lu\n", log_ptr, cur_block, blocks_ptr);
#endif
		const key_type log_key = std::get<0>(log_to_flush[log_ptr]);
		const key_type block_key = blind_read_key(blocks_ptr);
		assert(log_key != block_key);
		if (log_key < block_key) {
#if DEBUG_PRINT
			printf("pushed %lu from log to buffer\n", std::get<0>(log_to_flush[log_ptr]));
#endif
#if DEBUG
			if (buffer.size() >= 1) {		
				assert(std::get<0>(buffer[buffer.size() - 1]) < std::get<0>(log_to_flush[log_ptr]));
			}
#endif
			buffer.push_back(log_to_flush[log_ptr]);
			log_ptr++;
		} else {
#if DEBUG
			assert(blocks_ptr < N);
			if (buffer.size() >= 1) {		
				ASSERT(std::get<0>(buffer[buffer.size() - 1]) < blind_read_key(blocks_ptr), "buffer end %lu, blocks ptr %lu, blocks elt %lu\n", std::get<0>(buffer[buffer.size() - 1]), blocks_ptr, blind_read_key(blocks_ptr));
			}
#endif
			buffer.push_back(blind_read(blocks_ptr));
			advance_block_ptr(&blocks_ptr, &cur_block, &start_of_cur_block, count_per_block);
		}
	}

	// cleanup if necessary
	while(log_ptr < log_end) {
		buffer.push_back(log_to_flush[log_ptr]);
		log_ptr++;
	}

#if DEBUG_PRINT
	printf("\n*** cleaning up with blocks ***\n");
#endif
	while(cur_block < num_blocks) {
#if DEBUG
		ASSERT(blind_read_key(blocks_ptr) != NULL_VAL, "block ptr %lu\n", blocks_ptr);
#endif
#if DEBUG_PRINT
		printf("added %lu at idx %lu to buffer\n", blind_read_key(blocks_ptr), blocks_ptr);
#endif
		buffer.push_back(blind_read(blocks_ptr));
		advance_block_ptr(&blocks_ptr, &cur_block, &start_of_cur_block, count_per_block);
	}
#if DEBUG
	ASSERT(buffer.size() <= header_size + num_blocks * block_size, "buffer size = %lu, blocks slots = %lu\n", buffer.size(), num_blocks * block_size);
#endif
	num_elts_total = buffer.size();

#if DEBUG_PRINT
	printf("*** BUFFER ***\n");
	for(size_t i = 0; i < buffer.size(); i++) {
		printf("%lu\t", std::get<0>(buffer[i]));
	}
	printf("\n");
#endif

#if DEBUG
	// at this point the buffer should be sorted
	for (size_t i = 1; i < buffer.size(); i++) {
		assert(has(std::get<0>(buffer[i])));
		ASSERT(std::get<0>(buffer[i]) > std::get<0>(buffer[i-1]), "buffer[%lu] = %lu, buffer[%lu] = %lu\n", i-1, std::get<0>(buffer[i-1]), i, std::get<0>(buffer[i]));
	}
#endif

	// we have merged in all the inserts
	num_inserts_in_log = 0;

	// split up the buffer evenly amongst the rest of the blocks
	global_redistribute_buffer(buffer.data(), buffer.size());
}

// return index of the block that this elt would fall in
template <size_t log_size, size_t header_size, size_t block_size, typename key_type, typename... Ts>
size_t LeafDS<log_size, header_size, block_size, key_type, Ts...>::find_block(key_type key) const {
	return find_block_with_hint(key, 0);
}

// return index of the block that this elt would fall in
template <size_t log_size, size_t header_size, size_t block_size, typename key_type, typename... Ts>
size_t LeafDS<log_size, header_size, block_size, key_type, Ts...>::find_block_with_hint(key_type key, size_t hint) const {
	assert(blind_read_key(header_start) != 0);
#if !AVX512 || DEBUG
	// scalar version for debug
	size_t i = header_start + hint;
	ASSERT(key >= blind_read_key(i), "key = %lu, start block = %lu, header key = %lu\n", key, hint, blind_read_key(i));
	size_t correct_ret = hint;

	for( ; i < blocks_start; i++) {
		correct_ret += blind_read_key(i) <= key;
	}
	if (correct_ret == num_blocks || blind_read_key(header_start + correct_ret) > key) {
		correct_ret--;
	}
#if !AVX512
	return correct_ret;
#endif
#endif
#if AVX512
	// vector version
	size_t vector_start = header_start;
	size_t vector_end = blocks_start;
	// printf("vector start = %lu, vector end = %lu, keys per vector = %lu\n", vector_start, vector_end, keys_per_vector);
	size_t ret = 0;
	mask_type mask;
	for(; vector_start < vector_end; vector_start += keys_per_vector) {
		if constexpr (std::is_same<key_type, uint32_t>::value) { // 32-bit keys
			__m512i bcast = _mm512_set1_epi32(key);
			__m512i block = _mm512_loadu_si512((const __m512i *)(array.data() + vector_start * sizeof(key_type)));
			mask = _mm512_cmple_epu32_mask(block, bcast);
		} else {
			__m512i bcast = _mm512_set1_epi64(key);
			// printf("key = %lu, vector_start = %lu, byte start = %lu\n", key, vector_start, vector_start * sizeof(key_type));

			__m512i block = _mm512_loadu_si512((const __m512i *)(array.data() + vector_start * sizeof(key_type)));
			mask = _mm512_cmple_epu64_mask(block, bcast);
			// printf("in 64-bit case, popcount got %lu\n", __builtin_popcount(mask));
			assert((size_t)(__builtin_popcount(mask)) <= keys_per_vector);
		}
		ret += __builtin_popcount(mask);
	}

	// TODO: can you do to the next line faster?
	// printf("num blocks = %lu\n", num_blocks);
	assert(ret <= num_blocks);
	if (ret == num_blocks || blind_read_key(header_start + ret) > key) {
		ret--;
	}

	ASSERT(ret == correct_ret, "searching for key %lu: got %lu, should be %lu\n", key, ret, correct_ret);
#endif
#if DEBUG
	i = header_start + hint;
	for( ; i < blocks_start; i++) {
		if(blind_read_key(i) == key)  {
			return i - header_start;
		} else if (blind_read_key(i) > key) {
			break;
		}
	}
	assert(i - header_start - 1 < num_blocks);
#if AVX512
	ASSERT(i - header_start - 1 == ret, "elt %lu, original found %lu, new is %lu\n", key, i - header_start - 1, ret);
#endif
#endif
#if AVX512
	return ret;
#endif
}

// given a src, dest indices 
// move elt at src into dest
template <size_t log_size, size_t header_size, size_t block_size, typename key_type, typename... Ts>
void LeafDS<log_size, header_size, block_size, key_type, Ts...>::copy_src_to_dest(size_t src, size_t dest) {
	SOA_type::get_static(array.data(), N, dest) =
		SOA_type::get_static(array.data(), N, src);
}


// precondition: range must be packed
// TODO: vectorized sorting
// one way would be to access the key array, sort that in a vectorized way, and apply
// the permutation vector to later value vectors
template <size_t log_size, size_t header_size, size_t block_size, typename key_type, typename... Ts>
void LeafDS<log_size, header_size, block_size, key_type, Ts...>::sort_range(size_t start_idx, size_t end_idx) {
	assert(start_idx <= end_idx);
	auto start = typename LeafDS<log_size, header_size, block_size, key_type, Ts...>::SOA_type::Iterator(array.data(), N, start_idx);
	auto end = typename LeafDS<log_size, header_size, block_size, key_type, Ts...>::SOA_type::Iterator(array.data(), N, end_idx);

	std::sort(start, end, [](auto lhs, auto rhs) { return std::get<0>(typename SOA_type::T(lhs)) < std::get<0>(typename SOA_type::T(rhs)); } );
#if DEBUG
	// check sortedness
	for(size_t i = start_idx + 1; i < end_idx; i++) {
		if (blind_read_key(i-1) >= blind_read_key(i)) {
			print();
		}
		ASSERT(blind_read_key(i-1) < blind_read_key(i), "i: %lu, key at i - 1: %lu, key at i: %lu, num_dels: %lu", i, blind_read_key(i-1), blind_read_key(i), num_deletes_in_log);
		assert(blind_read_key(i-1) < blind_read_key(i));

	}
#endif
}

template <size_t log_size, size_t header_size, size_t block_size, typename key_type, typename... Ts>
void LeafDS<log_size, header_size, block_size, key_type, Ts...>::sort_array_range(void *base_array, size_t size, size_t start_idx, size_t end_idx) {
	auto start = typename LeafDS<log_size, header_size, block_size, key_type, Ts...>::SOA_type::Iterator(base_array, size, start_idx);
	auto end = typename LeafDS<log_size, header_size, block_size, key_type, Ts...>::SOA_type::Iterator(base_array, size, end_idx);

	std::sort(start, end, [](auto lhs, auto rhs) { return std::get<0>(typename SOA_type::T(lhs)) < std::get<0>(typename SOA_type::T(rhs)); } );
}

// sort the log
template <size_t log_size, size_t header_size, size_t block_size, typename key_type, typename... Ts>
void LeafDS<log_size, header_size, block_size, key_type, Ts...>::sort_log() {
	sort_range(0, num_inserts_in_log);
}

// if you expect at most 1 occurrence, use compare to 0
// then use tzcnt to tell you the index of the first 1

// if you could possibly have more than 1 match:
// use popcount to tell you how many matches there are
// __builtin_popcountll(mask64)
// if yes, use select to find the index
static inline uint8_t word_select(uint64_t val, int rank) {
  val = _pdep_u64(one[rank], val);
  return _tzcnt_u64(val);
}

// given a range [start, end), look for elt e 
// if e is in the range, update it and return true
// otherwise return false
// also return index found or index stopped at
template <size_t log_size, size_t header_size, size_t block_size, typename key_type, typename... Ts>
template <range_type type>
bool LeafDS<log_size, header_size, block_size, key_type, Ts...>::update_in_range_if_exists(size_t start, size_t end, element_type e) {
#if DEBUG_PRINT
	printf("*** start = %lu, end = %lu ***\n", start, end);
#endif

	[[maybe_unused]] bool correct_answer = false;
	[[maybe_unused]] size_t test_i = start;
#if !AVX512 || DEBUG
	// scalar version for correctness
	const key_type key = std::get<0>(e);

	for(; test_i < end; test_i++) { 
		if (key == get_key_array(test_i)) {
		#if DEBUG_PRINT
			printf("CORRECT FOUND KEY %lu AT IDX %zu\n", std::get<0>(e), test_i);
			print();
		#endif
			correct_answer = true;
			break;
		} else if (get_key_array(test_i) == NULL_VAL) {
			correct_answer = false;
			break;
		}
	}
#if !AVX512
	return correct_answer;
#endif
#endif
#if AVX512
	// vector version
	if constexpr (type == BLOCK) {
		assert(start % keys_per_vector == 0);
		assert(end % keys_per_vector == 0);
	
		for(size_t vector_start = start; vector_start < end; vector_start += keys_per_vector) {
			mask_type mask = slot_mask(array.data() + vector_start * sizeof(key_type), std::get<0>(e));
			if (mask > 0) {
				auto i = vector_start + _tzcnt_u64(mask);

				assert(test_i == i);
				assert((correct_answer == true));

				update_val_at_index(e, i);
				return true;
			}
		}
		return false;
	} 

	// if you are in the log
	size_t vector_start = start & ~(all_ones_vec);
	size_t vector_end = (end + all_ones_vec) & ~(all_ones_vec);
#if DEBUG_PRINT
	printf("\tvector start %lu, vector end %lu\n", vector_start, vector_end);
#endif
	mask_type mask = slot_mask(array.data() + vector_start * sizeof(key_type), std::get<0>(e));

	if constexpr (type == DELETES) {
		// TODO: check that this is right
		lr_mask_type left_mask = get_right_mask(start % keys_per_vector);
		mask &= left_mask;
		if (mask > 0) {
			auto idx = vector_start + _tzcnt_u64(mask); // check if there is an off-by-1 in tzcnt

			ASSERT(test_i == idx, "test_i = %zu, i = %llu, mask = %d\n", test_i, idx, mask);
			assert((correct_answer == true));

			update_val_at_index(e, idx);
			return true;
		}
		vector_start += keys_per_vector;
	}
	
	if (vector_end < keys_per_vector) {
		assert((correct_answer == false));
		return false;
	}

	// do blocks or any full blocks of the log
	// will miss the last full block, if there is one
	while(vector_start + keys_per_vector <= end) {
		auto mask = slot_mask(array.data() + vector_start * sizeof(key_type), std::get<0>(e));
		if (mask > 0) {
			auto i = vector_start + _tzcnt_u64(mask);

			assert(test_i == i);
			assert((correct_answer == true));
#if DEBUG_PRINT
			printf("\tVECTOR LOOP FOUND AT IDX %lu\n", i);
#endif
			update_val_at_index(e, i);
			return true;
		}
		vector_start += keys_per_vector;
	}
#if DEBUG_PRINT
	printf("\tvector start %lu, end %lu\n", vector_start, end);
#endif

	// do the remaining ragged right, if there is any.
	if constexpr (type == INSERTS) {
#if DEBUG_PRINT
		printf("\tINSERTS: vector start %lu, end %lu\n", vector_start, end);
#endif
		if (vector_start < end) { // this is not exactly optimal, TBD how to remove it
			// vector fills from the right (tzcnt), so use get_left_mask
			lr_mask_type left_mask = get_left_mask(end % keys_per_vector);
			mask = slot_mask(array.data() + vector_start * sizeof(key_type), std::get<0>(e));
#if DEBUG_PRINT
			printf("\tleft mask %u, mask %u\n", left_mask, mask);
#endif
			mask &= left_mask;
			if (mask > 0) {
				auto i = vector_start + _tzcnt_u64(mask);
				assert(test_i == i);
#if DEBUG_PRINT
				printf("\tVECTOR END FOUND AT IDX %lu\n", i);
#endif
				update_val_at_index(e, i);
				assert((correct_answer == true));
				return true;
			}
		}
	}

	ASSERT((correct_answer == false), "searching for key %lu in range [%lu, %lu)\n", std::get<0>(e), start, end);
	return false;
#endif
} 

// (only used in delete)
// given a range [start, end), look for elt e 
// if e is in the range, update it and return true
// otherwise return false
// also return index found or index stopped at
template <size_t log_size, size_t header_size, size_t block_size, typename key_type, typename... Ts>
std::pair<bool, size_t> LeafDS<log_size, header_size, block_size, key_type, Ts...>::find_key_in_range(size_t start,
	size_t end, key_type key) const {
	size_t i = start;

	for(; i < end; i++) { // TODO: vectorize this for loop using AVX-512
		// if found, update the val and return	
		if (key == get_key_array(i)) {
			return {true, i};
		} else if (get_key_array(i) == NULL_VAL) {
			return {false, i};
		}
		
	}
	return {false, end};
} 


// given a block index, return its range [start, end)
// TODO: precompute this in a table
template <size_t log_size, size_t header_size, size_t block_size, typename key_type, typename... Ts>
inline std::pair<size_t, size_t> LeafDS<log_size, header_size, block_size, key_type, Ts...>::get_block_range(size_t block_idx) const {
	size_t block_start = blocks_start + block_idx * block_size;
	size_t block_end = block_start + block_size;
#if DEBUG
	ASSERT(block_idx < num_blocks, "block idx %lu\n", block_idx);
	assert(block_start < N);
	assert(block_end <= N);
#endif
	return {block_start, block_end};
}

// count up the number of elements in this b lock
template <size_t log_size, size_t header_size, size_t block_size, typename key_type, typename... Ts>
unsigned short LeafDS<log_size, header_size, block_size, key_type, Ts...>::count_block(size_t block_idx) const {
	size_t block_start = blocks_start + block_idx * block_size;
	size_t block_end = block_start + block_size;

#if !AVX512 || DEBUG
	// count number of nonzero elts in this block
	uint64_t correct_count = 0;
	SOA_type::template map_range_static(array.data(), N, [&correct_count](auto key) {correct_count += key != 0;}, block_start, block_end);
#if !AVX512
	return correct_count;
#endif
#endif
#if AVX512
	uint64_t num_zeroes = 0;
	mask_type mask;
	for(; block_start < block_end; block_start += keys_per_vector) {
		if constexpr (std::is_same<key_type, uint32_t>::value) { // 32-bit keys
			__m512i bcast = _mm512_set1_epi32(0);
			__m512i block = _mm512_loadu_si512((const __m512i *)(array.data() + block_start * sizeof(key_type)));
			mask = _mm512_cmpeq_epu32_mask(block, bcast);
		} else {
			__m512i bcast = _mm512_set1_epi64(0);
			__m512i block = _mm512_loadu_si512((const __m512i *)(array.data() + block_start * sizeof(key_type)));
			mask = _mm512_cmpeq_epu64_mask(block, bcast);
		}
		num_zeroes += __builtin_popcount(mask);
	}

	ASSERT(num_zeroes <= block_size, "counted zeroes %lu, block size %lu\n", num_zeroes, block_size);

	uint64_t count = block_size - num_zeroes;

	ASSERT(correct_count == count, "counting block %lu, got count %lu, should be %lu\n", block_idx, count, correct_count);
	return count;
#endif
}

// flush the log to the blocks
// precondition: log has been sorted
template <size_t log_size, size_t header_size, size_t block_size, typename key_type, typename... Ts>
void LeafDS<log_size, header_size, block_size, key_type, Ts...>::flush_log_to_blocks(size_t max_to_flush) {

#if DEBUG_PRINT
	printf("BEFORE FLUSH\n");
	print();
#endif
	// dedup the log wrt the blocks, putting all new elts in log_to_flush
	element_type log_to_flush[log_size];
	unsigned short num_to_flush = 0;
	unsigned short num_to_flush_per_block[num_blocks];
	size_t hint = 0;
	memset(num_to_flush_per_block, 0, num_blocks * sizeof(unsigned short));

	// look for the sorted log in the blocks
	for(size_t i = 0; i < max_to_flush; i++) {
		key_type key_to_flush = blind_read_key(i);

#if DEBUG
		ASSERT(key_to_flush >= blind_read_key(header_start), "key to flush = %lu, first header = %lu\n", key_to_flush, blind_read_key(header_start));
#endif

		size_t block_idx = find_block_with_hint(key_to_flush, hint);
#if DEBUG
		assert(block_idx < num_blocks);
#endif
		// if it is in the header, update the header
		if (blind_read_key(header_start + block_idx) == key_to_flush) {
				copy_src_to_dest(i, header_start + block_idx);
#if DEBUG_PRINT
				printf("found duplicate in header idx %zu of elt %lu\n", block_idx, key_to_flush);
#endif
				num_elts_total--;
		} else {
			// otherwise, look for it in the block
			// TODO: update this to take in the block index because you already have it
			auto update_block = update_in_block_if_exists(blind_read(i));
			// block_idx = (update_block.second - blocks_start) / block_size; // floor div
			// update hint 
			if (hint < block_idx) { hint = block_idx; }

			// if it was in the block, do nothing bc we have already updated it
			// if not found, add to the deduped log_to_flush
			if (!update_block) {
				// if not found, the second thing in the pair is the index at the end
#if DEBUG_PRINT
				printf("\tflushing elt %lu to block %lu, header %lu\n", blind_read_key(i), block_idx, blind_read_key(header_start + block_idx));
				printf("set log_to_flush[%hu] = %lu\n", num_to_flush, blind_read_key(i));
#endif
				log_to_flush[num_to_flush] = blind_read(i);
				num_to_flush++;
				num_to_flush_per_block[block_idx]++;
			} else {
				num_elts_total--;
#if DEBUG_PRINT
				printf("found duplicate in block %lu of elt %zu\n", block_idx, key_to_flush);
				printf("num elts now %zu\n", num_elts_total);
#endif
			}
		}
	}

	// count the number of elements in each block
	unsigned short count_per_block[num_blocks];

	// TODO: merge these loops and count the rest in global redistribute
	for (size_t i = 0; i < num_blocks; i++) {
		count_per_block[i] = count_block(i);
	}

	// if any of them overflow, redistribute
	// TODO: can vectorize this part
	bool need_global_redistrubute = false;
	for (size_t i = 0; i < num_blocks; i++) {
		if (count_per_block[i] + num_to_flush_per_block[i] >= block_size) {
			need_global_redistrubute = true;
#if DEBUG_PRINT
			printf("*** GLOBAL REDISTRIB FROM OVERFLOW ***\n");
			printf("at block %lu, count_per_block = %u, num to flush = %u\n", i, count_per_block[i], num_to_flush_per_block[i]);
#endif
		}
	}
	if (need_global_redistrubute) {
		global_redistribute(log_to_flush, num_to_flush, count_per_block);
		return; // log gets taken care of in global redistribute
	}

	// otherwise, flush the log to the blocks
	size_t idx_in_log = 0;
	for(size_t i = 0; i < num_blocks; i++) {
		// pointer to start of block
		size_t write_start = blocks_start + i * block_size + count_per_block[i];
		for(size_t j = 0; j < num_to_flush_per_block[i]; j++) {
			blind_write(log_to_flush[idx_in_log + j], write_start + j);
		}
		idx_in_log += num_to_flush_per_block[i];
	}

#if DEBUG
	ASSERT(idx_in_log == num_to_flush, "flushed %lu, should have flushed %u\n", idx_in_log, num_to_flush);
#endif

#if DEBUG_PRINT
	printf("AFTER FLUSH\n");
	print();
#endif
}

template <size_t log_size, size_t header_size, size_t block_size, typename key_type, typename... Ts>
bool LeafDS<log_size, header_size, block_size, key_type, Ts...>::update_in_block_if_exists(element_type e) {
	const key_type key = std::get<0>(e);
	// if key is in the current range of this node, try to find it in the block
	auto block_idx = find_block(key);
	auto block_range = get_block_range(block_idx);
	// if found, update and return
	return update_in_range_if_exists<BLOCK>(block_range.first, block_range.second, e);
}

// take in the block idx
template <size_t log_size, size_t header_size, size_t block_size, typename key_type, typename... Ts>
bool LeafDS<log_size, header_size, block_size, key_type, Ts...>::update_in_block_if_exists(element_type e, size_t block_idx) {
	// if key is in the current range of this node, try to find it in the block
	auto block_range = get_block_range(block_idx);
	// if found, update and return
	return update_in_range_if_exists(block_range.first, block_range.second, e);
}

// take in the block idx
template <size_t log_size, size_t header_size, size_t block_size, typename key_type, typename... Ts>
void LeafDS<log_size, header_size, block_size, key_type, Ts...>::delete_from_block_if_exists(key_type e, size_t block_idx) {
	// if key is in the current range of this node, try to find it in the block
	auto block_range = get_block_range(block_idx);
	// if found, shift everything left by 1
	for(size_t i = block_range.first; i < block_range.second; i++) {
		if(blind_read_key(i) == e) {
#if DEBUG_PRINT
			printf("found elt %zu to delete in block %zu, idx %zu\n", e, block_idx, i);
#endif
			for(size_t j = i; j < block_range.second - 1; j++) {
				SOA_type::get_static(array.data(), N, j) = SOA_type::get_static(array.data(), N, j+1);
			}

			// TODO: is there a better way to clear a single element at the end?
			clear_range(block_range.second - 1, block_range.second);
			break;
		}
	}
}

// return true if the element was inserted, false otherwise
// may return true if the element was already there due to it being a pseudo-set
template <size_t log_size, size_t header_size, size_t block_size, typename key_type, typename... Ts>
bool LeafDS<log_size, header_size, block_size, key_type, Ts...>::insert(element_type e) {
	assert(std::get<0>(e) != 0);
	// first try to update the key if it is in the log
	auto result = update_in_range_if_exists<INSERTS>(0, num_inserts_in_log, e);
	if (result) { 
		return false; 
	}

#if DEBUG
	// there should always be space in the log
	assert(num_inserts_in_log + num_deletes_in_log < log_size);
#endif

	// if not found, add it to the log
	blind_write(e, num_inserts_in_log);
	num_elts_total++; // num elts (may be duplicates in log and block)
	num_inserts_in_log++;
	
	if (num_inserts_in_log + num_deletes_in_log == log_size) { // we filled the log
		sort_range(0, num_inserts_in_log); // sort inserts

		// if this is the first time we are flushing the log, just make the sorted log the header
		if (get_min_block_key() == 0) {
			if (num_deletes_in_log > 0) { // if we cannot fill the header
				clear_range(num_inserts_in_log, log_size); // clear deletes
				num_deletes_in_log = 0;
				return true;
			} else {
	#if DEBUG_PRINT
				printf("\nmake sorted log the header\n");
				print();
	#endif

				for(size_t i = 0; i < log_size; i++) {
					SOA_type::get_static(array.data(), N, i + header_start) =
						SOA_type::get_static(array.data(), N, i);
				}
			}
		} else { // otherwise, there are some elements in the block / header part
#if DEBUG
			assert(num_inserts_in_log > 0);
#endif
			if(num_deletes_in_log > 0) {
				sort_range(log_size - num_deletes_in_log, log_size); // sort deletes
			}
			// if inserting min, swap out the first header into the first block
			if (blind_read_key(0) < get_min_block_key()) {

#if DEBUG
		    size_t i = blocks_start;
				for(; i < blocks_start + block_size; i++) {
					if (blind_read_key(i) == 0) {
						break;
					}
				}
#endif
				
				size_t j = blocks_start + block_size;
				// find the first zero slot in the block
				// TODO: this didn't work (didn't find the first zero)
				SOA_type::template map_range_with_index_static(array.data(), N, [&j](auto index, auto key) {
					if (key == 0) {
						j = std::min(index, j);
					}
				}, blocks_start, blocks_start + block_size);
#if DEBUG
				ASSERT(i == j, "got %zu, should be %zu\n", j, i);
				assert(i < blocks_start + block_size);
#endif

				// put the old min header in the block where it belongs
				// src = header start, dest = i
				copy_src_to_dest(header_start, j);

				// make min elt the new first header
				copy_src_to_dest(0, header_start);
				num_elts_total++;
			}

			// flush the log
			// note: at this point, the min key is repeated in the
			// header and log (if there was a new min).  in the flush, it will just
			// get deduped
			// TODO: first flush the deletes
			if (num_deletes_in_log > 0) {
				if (delete_from_header()) {
					strip_deletes_and_redistrib();
				} else {
					flush_deletes_to_blocks();
				}
			}
			flush_log_to_blocks(num_inserts_in_log);
		}

		// clear log
		num_deletes_in_log = 0;
		num_inserts_in_log = 0;
		clear_range(0, log_size);
	}

	return true;
}


template <size_t log_size, size_t header_size, size_t block_size, typename key_type, typename... Ts>
inline void LeafDS<log_size, header_size, block_size, key_type, Ts...>::advance_block_ptr(size_t* blocks_ptr, size_t* cur_block, size_t* start_of_cur_block, unsigned short* count_per_block) const {
#if DEBUG_PRINT
		if (blind_read_key(*blocks_ptr) == NULL_VAL) {
			printf("null blocks ptr %zu\n", *blocks_ptr);
			assert(false);
		}
		printf("\tpushed %lu from blocks to buffer\n", blind_read_key(*blocks_ptr));
#endif
#if DEBUG
		size_t prev_blocks_ptr = *blocks_ptr;
#endif

		// if we are in the header, go to the block if the block is nonempty
		if (*blocks_ptr < blocks_start) {
			*start_of_cur_block = blocks_start + (*cur_block) * block_size;

			if (blind_read_key(*start_of_cur_block) != NULL_VAL) {
				*blocks_ptr = *start_of_cur_block;
			} else { // if this block is empty, move to the next header
				(*cur_block)++;
				(*blocks_ptr)++;
			}
		} else if (*blocks_ptr == *start_of_cur_block + count_per_block[*cur_block] - 1) {
			// if we have merged in this entire block, go back to the header
			(*cur_block)++;
			*blocks_ptr = header_start + *cur_block;
		} else { // if we are still in this block, keep going
			(*blocks_ptr)++;
		}
#if DEBUG
		assert(prev_blocks_ptr != *blocks_ptr); // made sure we advanced
#endif
}

// precondition: we are not deleting from the header
template <size_t log_size, size_t header_size, size_t block_size, typename key_type, typename... Ts>
void LeafDS<log_size, header_size, block_size, key_type, Ts...>::flush_deletes_to_blocks() {
#if DEBUG_PRINT
	printf("flushing deletes, num deletes = %lu\n", num_deletes_in_log);
#endif
	size_t hint = 0;
	// process the deletes
	for(size_t i = log_size - num_deletes_in_log; i < log_size; i++) {
		key_type key_to_delete = blind_read_key(i);
#if DEBUG_PRINT
		printf("\tflushing delete %lu\n", key_to_delete);
#endif
		size_t block_idx = find_block_with_hint(key_to_delete, hint);
#if DEBUG_PRINT
		printf("\tflushing delete %lu to block %lu\n", key_to_delete, block_idx);
#endif
		// try to delete it from the blocks if it exists
		delete_from_block_if_exists(key_to_delete, block_idx);

		if (hint < block_idx) { hint = block_idx; }
	}

	// clear delete log
	clear_range(log_size - num_deletes_in_log, log_size);

	// count the blocks
	unsigned short count_per_block[num_blocks];
	bool redistribute = false;
	for (size_t i = 0; i < num_blocks; i++) {
    		count_per_block[i] = count_block(i);
		if (count_per_block[i] == 0) { redistribute = true; }
	}

	// if any of the blocks become empty, do a global redistribute
	if (redistribute) {
	  // just redistribute the header/blocks
	  global_redistribute_blocks(count_per_block);
	}
}

template <size_t log_size, size_t header_size, size_t block_size, typename key_type, typename... Ts>
void LeafDS<log_size, header_size, block_size, key_type, Ts...>::strip_deletes_and_redistrib() {
  // count the number of elements in each block
#if DEBUG_PRINT
  printf("\nstrip deletes and redistrib, num deletes = %lu\n", num_deletes_in_log);
  print();
#endif
  unsigned short count_per_block[num_blocks];
  size_t total_count = 0;
  for (size_t i = 0; i < num_blocks; i++) {
    count_per_block[i] = count_block(i);
    total_count += count_per_block[i];
  }

  // sort the blocks
  for (size_t i = 0; i < num_blocks; i++) {
    auto block_range = get_block_range(i);
    sort_range(block_range.first, block_range.first + count_per_block[i]);
  }

  // merge into buffer
  element_type buffer[total_count + header_size];

  // sort delete log
  // printf("sort range [%lu, %lu)\n", log_size - num_deletes_in_log, log_size);
  // sort_range(log_size - num_deletes_in_log, log_size);
#if DEBUG
  for(size_t i = log_size - num_deletes_in_log + 1; i < log_size; i++) {
		assert(blind_read_key(i) > blind_read_key(i-1));
  }
#endif

  // two-finger strip of log from blocks/header
  size_t log_ptr = log_size - num_deletes_in_log;
  size_t blocks_ptr = header_start;
  size_t cur_block = 0;
  size_t start_of_cur_block = 0;
  size_t log_end = log_size;
  size_t buffer_ptr = 0;

  while(log_ptr < log_end && cur_block < num_blocks) {
    const key_type log_key = blind_read_key(log_ptr);
    const key_type block_key = blind_read_key(blocks_ptr);
    // if we are deleting this key
    if (log_key == block_key) {
#if DEBUG_PRINT
	printf("\tstrip %lu from log\n", log_key);
#endif
	log_ptr++;
	advance_block_ptr(&blocks_ptr, &cur_block, &start_of_cur_block, count_per_block);
	// increment block pointer
      } else if (log_key < block_key) {
	log_ptr++;
      } else { // merge in elts that we are keeping
	buffer[buffer_ptr++] = blind_read(blocks_ptr);
	advance_block_ptr(&blocks_ptr, &cur_block, &start_of_cur_block, count_per_block);
      }
    }

    // cleanup by merging in rest of DS
    while(cur_block < num_blocks) {
      buffer[buffer_ptr++] = blind_read(blocks_ptr);
      advance_block_ptr(&blocks_ptr, &cur_block, &start_of_cur_block, count_per_block);
    }

    // we have removed all the deletes. there may still be stuff in the insert log.

    // num elts total is num_insert_log + from blocks
    // but there might be some repetitions between log/blocks
    // split up buffer into blocks if there is enough
    if (buffer_ptr > header_size) {
#if DEBUG_PRINT
      printf("large buffer case\n");
#endif
      global_redistribute_buffer(buffer, buffer_ptr);
    } else { // otherwise, put it in the header
#if DEBUG_PRINT
	printf("before small buffer case, buffer_ptr = %lu\n", buffer_ptr);
	print();
#endif
	    // make sure to delete the repetitions if you are going to the log
	    sort_log();

	    // merge in inserts log
	    element_type buffer2[num_inserts_in_log + buffer_ptr];
	    log_ptr = 0;
	    log_end = num_inserts_in_log;
	    auto buffer_end = buffer_ptr;
	    buffer_ptr = 0;
	    size_t out_ptr = 0;
	    while(log_ptr < log_end && buffer_ptr < buffer_end) {
		    const key_type buffer_key = std::get<0>(buffer[buffer_ptr]);
		    const key_type log_key = blind_read_key(log_ptr);
		    // if they are equal, the one the ds is newest
		    if (log_key == buffer_key) {
			    buffer2[out_ptr++] = blind_read(log_ptr);
			    log_ptr++;
			    buffer_ptr++;
		    } else if (log_key > buffer_key) {
			    buffer2[out_ptr++] = buffer[buffer_ptr++];
		    } else {
			    buffer2[out_ptr++] = blind_read(log_ptr);
			    log_ptr++;
		    }
	    } 
	    // finish up
	    while(log_ptr < log_end) {
		    buffer2[out_ptr++] = blind_read(log_ptr);
		    log_ptr++;
	    }
	    while(buffer_ptr < buffer_end) {
		    buffer2[out_ptr++] = buffer[buffer_ptr++];
	    }

	    num_deletes_in_log = 0;
	    clear_range(0, N);
	    printf("num left after merging log and block = %lu\n", out_ptr);
	    if (out_ptr <= log_size) { // if they can all fit in the log
		    size_t i = 0;
		    
		    for(; i < out_ptr; i++) {
			    place_elt_at_index(buffer2[i], i);
		    }
		    num_inserts_in_log = i;
		    num_elts_total = i;
	    } else { //otherwise, put the first some into the headers and the rest into log
		    size_t i = 0;
		    for(; i < log_size; i++) {
			    place_elt_at_index(buffer2[i], header_start + i);
		    }
		    for(; i < out_ptr; i++) {
			    place_elt_at_index(buffer2[i], i - log_size);
		    }
		    num_elts_total = out_ptr;
		    num_inserts_in_log = out_ptr - log_size;
	    }

#if DEBUG_PRINT
	    printf("\nput the rest into the insert log\n");
	    print();
#endif
    }
}

// return whether we are deleting from the header or not
template <size_t log_size, size_t header_size, size_t block_size, typename key_type, typename... Ts>
bool LeafDS<log_size, header_size, block_size, key_type, Ts...>::delete_from_header() {

	// rebuild if we are deleting from the header
	size_t log_ptr = log_size - num_deletes_in_log;
	size_t header_ptr = header_start;

	while(log_ptr < log_size && header_ptr < header_start + header_size) {
		if(blind_read_key(header_ptr) == blind_read_key(log_ptr)) {
			return true;
		} else if (blind_read_key(header_ptr) > blind_read_key(log_ptr)) {
			log_ptr++;
		} else {
			header_ptr++;
		}
	}
	return false;
}

// return true if the element was deleted, false otherwise
// may return false even if the elt is there due to it being a pseudo-set
// return N if not found, otherwise return the slot the key is at
// TODO: what if key_type is not elt_type? do you make a fake elt?
// TODO: this breaks get_num_elements() prior to flushing
template <size_t log_size, size_t header_size, size_t block_size, typename key_type, typename... Ts>
bool LeafDS<log_size, header_size, block_size, key_type, Ts...>::remove(key_type e) {

#if DEBUG
	assert(num_deletes_in_log + num_inserts_in_log < log_size);
#endif

	// check if the element is in the insert log
	bool was_insert = false;
	for(size_t i = 0; i < num_inserts_in_log; i++) {
		// if so, shift everything left by 1 (cancel the insert)
		if (blind_read_key(i) == e) {
			was_insert = true;
#if DEBUG_PRINT
			printf("\tfound in insert log\n");
#endif
			for(size_t j = i; j < num_inserts_in_log - 1; j++) {
				SOA_type::get_static(array.data(), N, j) = SOA_type::get_static(array.data(), N, j+1); 
			}
			clear_range(num_inserts_in_log - 1, num_inserts_in_log);
			num_inserts_in_log--;
			break;
		}
	}

	// if it was not an insert, look through the delete log for it
	if(!was_insert) {
		auto result = find_key_in_range(log_size - num_deletes_in_log, log_size, e);
		if (!result.first) { // if not in delete log, add it
			num_deletes_in_log++;
			blind_write(e, log_size - num_deletes_in_log); // grow left
		} else {
			printf("\tfound in delete log\n");	
		}
	} else { // otherwise, just add it to the delete log
		num_deletes_in_log++;
		blind_write(e, log_size - num_deletes_in_log);
	}

	// now check if the log is full
	if (num_deletes_in_log + num_inserts_in_log == log_size) {
#if DEBUG_PRINT
		printf("flushing delete log because full\n");
		print();
#endif
		// if the header is empty, the deletes just disappear
		// only do the flushing if there is stuff later in the DS
#if DEBUG_PRINT
		printf("\tmin block key = %lu\n", get_min_block_key());
#endif
		if (get_min_block_key() != 0) {
			sort_range(log_size - num_deletes_in_log, log_size);
			// if we are deleting from the header, do a global rewrite
			if (delete_from_header()) {
#if DEBUG_PRINT
				printf("deleting from header\n");
#endif
				// strip deletes and redistrib
				strip_deletes_and_redistrib();
			} else {
#if DEBUG_PRINT
				printf("\tflush deletes to blocks\n");
#endif
				flush_deletes_to_blocks();
			}
		}

		clear_range(log_size - num_deletes_in_log, log_size);
		num_deletes_in_log = 0;
	}
	return true; // ?
}

// return N if not found, otherwise return the slot the key is at
template <size_t log_size, size_t header_size, size_t block_size, typename key_type, typename... Ts>
inline size_t LeafDS<log_size, header_size, block_size, key_type, Ts...>::get_index_in_blocks(key_type e) const {
	// if there is no header / stuff in blocks, key is not there
	// if less than current min, should not be in ds
	if (blind_read_key(header_start) == 0 || e < blind_read_key(header_start)) {
		return N;
	}
	assert( e >= blind_read_key(header_start) );
	size_t block_idx = find_block(e);
#if DEBUG_PRINT
	printf("\tin has for elt %lu, find block returned %lu\n", e, block_idx);
#endif
	// check the header
	assert(e >= blind_read_key(header_start + block_idx));
	if (e == blind_read_key(header_start + block_idx)) {
		return header_start + block_idx;
	}
	
	// check the block
	// TODO: vectorize this search
	auto range = get_block_range(block_idx);

#if DEBUG_PRINT
	printf("\tblock range [%lu, %lu)\n", range.first, range.second);
#endif
	for(size_t i = range.first; i < range.second; i++) {
		if (blind_read_key(i) == NULL_VAL) {
			return N;
		}
		if (blind_read_key(i) == e) {
			return i;
		}
	}
	return N;
}


// return N if not found, otherwise return the slot the key is at
template <size_t log_size, size_t header_size, size_t block_size, typename key_type, typename... Ts>
size_t LeafDS<log_size, header_size, block_size, key_type, Ts...>::get_index(key_type e) const {
	// check the log
	// TODO: vectorize the search in the log
	// replace it with the vectorized search update_if_exists
	for(size_t i = 0; i < num_inserts_in_log; i++) {
		if(e == blind_read_key(i)) {
			return i;
		}
	}

	return get_index_in_blocks(e);
}

// return true iff element exists in the data structure
template <size_t log_size, size_t header_size, size_t block_size, typename key_type, typename... Ts>
bool LeafDS<log_size, header_size, block_size, key_type, Ts...>::has(key_type e) const {
	// first check if it is in the delete log
#if DEBUG_PRINT
	printf("in has for %lu\n", e);
#endif
	for(size_t i = log_size - 1; i > log_size - num_deletes_in_log - 1; i--) {
		if(blind_read_key(i) == e) {
#if DEBUG_PRINT
			printf("\tfound %lu in delete log\n", e);
			print();
#endif
			return false;
		}
	}
	
	// otherwise search in insert log and rest of DS
	auto idx = get_index(e);
	return (idx != N);
}

// return true iff element exists in the data structure
template <size_t log_size, size_t header_size, size_t block_size, typename key_type, typename... Ts>
bool LeafDS<log_size, header_size, block_size, key_type, Ts...>::has_with_print(key_type e) const {
	// first check if it is in the delete log
	for(size_t i = log_size; i > log_size - num_deletes_in_log; i--) {
		if(blind_read_key(i) == e) {
			printf("found %u in delete log\n", e);
			print();
			return false;
		}
	}
	
	// otherwise search in insert log and rest of DS
	auto idx = get_index(e);
	if (idx == N) { printf("%lu not found\n", e); print(); }
	return (idx != N);
}
// print the range [start, end)
template <size_t log_size, size_t header_size, size_t block_size, typename key_type, typename... Ts>
void LeafDS<log_size, header_size, block_size, key_type, Ts...>::print_range(size_t start, size_t end) const {
	SOA_type::map_range_with_index_static(
			(void *)array.data(), N,
			[](size_t index, key_type key, auto... args) {
				if (key != NULL_VAL) {
					if constexpr (binary) {
						std::cout << key << ", ";
					} else {
						std::cout << "((_" << index << "_)" << key << ", ";
						((std::cout << ", " << args), ...);
						std::cout << "), ";
					}
				} else {
					std::cout << "_" << index << "_,";
				}
			},
			start, end);
	printf("\n");
}

// print the entire thing
template <size_t log_size, size_t header_size, size_t block_size, typename key_type, typename... Ts>
void LeafDS<log_size, header_size, block_size, key_type, Ts...>::print() const {
  auto num_elts = count_up_elts();
  printf("total num elts via count_up_elts %lu\n", num_elts);
  printf("total num elts %lu\n", num_elts_total);
  printf("num inserts in log = %lu\n", num_inserts_in_log);
  printf("num deletes in log = %lu\n", num_deletes_in_log);
  SOA_type::print_type_details();

	if (num_elts == 0) {
    printf("the ds is empty\n");
  }

	printf("\nlog: \n");
	print_range(0, log_size);

	printf("\nheaders:\n");
	print_range(header_start, blocks_start);

	for (uint32_t i = blocks_start; i < N; i += block_size) {
		printf("\nblock %lu (header = %lu)\n", (i - blocks_start) / block_size, blind_read_key(header_start + (i - blocks_start) / block_size));
		print_range(i, i + block_size);
	}
	printf("\n");
}

// apply the function F to the entire data structure
// most general map function without inverse
template <size_t log_size, size_t header_size, size_t block_size, typename key_type, typename... Ts>
template <bool no_early_exit, size_t... Is, class F>
bool LeafDS<log_size, header_size, block_size, key_type, Ts...>::map(F f) const {
  // read-only version of map
  // for each elt in the log, search for it in the blocks
  // if it was found in the blocks, add the index of it into the duplicates list
  size_t skip_index[log_size];
  size_t num_to_skip = 0;

#if DEBUG_PRINT
  print();
#endif
  // add inserts to remove, if any
  for(size_t i = 0; i < num_inserts_in_log; i++) {
      key_type key = blind_read_key(i);
      size_t idx = get_index_in_blocks(key);
      if (idx < N) {
#if DEBUG_PRINT
	printf("\tskip %lu from insert log\n", key);
#endif
	skip_index[num_to_skip] = idx;
	num_to_skip++;
      }
  } 
  // add deletes to skip, if any
  for(size_t i = log_size - 1; i > log_size - num_deletes_in_log - 1; i--) {
    key_type key = blind_read_key(i);
    size_t idx = get_index_in_blocks(key);
    if (idx < N) {
#if DEBUG_PRINT
      printf("\tskip %lu from delete log\n", key);
#endif
      skip_index[num_to_skip] = idx;
      num_to_skip++;
    }
  }

  static_assert(std::is_invocable_v<decltype(&F::operator()), F &, uint32_t,
                                    NthType<Is>...>,
                "update function must match given types");

  // map over insert log
  for (size_t i = 0; i < num_inserts_in_log; i++) {
    auto element =
	    SOA_type::template get_static<0, (Is + 1)...>(array.data(), N, i);
    if constexpr (no_early_exit) {
	    std::apply(f, element);
    } else {
	    if (std::apply(f, element)) {
		    return true;
	    }
    }
  }
	
  // map over the rest after the delete log
  for (size_t i = header_start; i < N; i++) {
    auto index = get_key_array(i);
    // skip over deletes
    if (index != NULL_VAL) {
			// skip if duplicated
			bool skip = false;
			for(size_t j = 0; j < num_to_skip; j++) {
				if(i == skip_index[j]) { 
					skip = true; 
#if DEBUG_PRINT
					printf("skip elt %lu at idx %lu\n", index, i);
#endif
				}
			}
	if(skip) { continue; }

      auto element =
          SOA_type::template get_static<0, (Is + 1)...>(array.data(), N, i);
      if constexpr (no_early_exit) {
        std::apply(f, element);
      } else {
        if (std::apply(f, element)) {
          return true;
        }
      }
    }
  }
  return false;
}

template <size_t log_size, size_t header_size, size_t block_size, typename key_type, typename... Ts>
uint64_t LeafDS<log_size, header_size, block_size, key_type, Ts...>::sum_keys_with_map() const {
  uint64_t result = 0;
  map<true>([&](key_type key) { result += key; });
  return result;
}

template <size_t log_size, size_t header_size, size_t block_size, typename key_type, typename... Ts>
uint64_t LeafDS<log_size, header_size, block_size, key_type, Ts...>::sum_keys_direct() const {
#if DEBUG_PRINT
  printf("*** sum with subtraction ***\n");
#endif
  uint64_t result = 0;

  size_t skip_index[log_size];
  size_t num_to_skip = 0;

  // add inserts to remove, if any
  for(size_t i = 0; i < num_inserts_in_log; i++) {
    key_type key = blind_read_key(i);
    size_t idx = get_index_in_blocks(key);
    if (idx < N) {
	    skip_index[num_to_skip] = idx;
	    num_to_skip++;
    }
  } 
  // add deletes to skip, if any
  for(size_t i = log_size - 1; i > log_size - num_deletes_in_log - 1; i--) {
    key_type key = blind_read_key(i);
    size_t idx = get_index_in_blocks(key);
    if (idx < N) {
#if DEBUG_PRINT
	    printf("\tskip key %lu from deletes\n", key);
#endif
	    skip_index[num_to_skip] = idx;
	    num_to_skip++;
    }
  }
  assert(num_to_skip < log_size);

  // do inserts
  for (size_t i = 0; i < num_inserts_in_log; i++) {
    result += blind_read_key(i);
  }

  for (size_t i = header_start; i < N; i++) {
    result += blind_read_key(i);
  }

  for(size_t i = 0; i < num_to_skip; i++) {
    result -= blind_read_key(skip_index[i]);
  }
  return result;
}


// return true if the given key is going to be deleted
template <size_t log_size, size_t header_size, size_t block_size, typename key_type, typename... Ts>
inline bool LeafDS<log_size, header_size, block_size, key_type, Ts...>::is_in_delete_log(key_type key) {
	for(size_t j = log_size - 1; j > log_size - 1 - num_deletes_in_log; j++) {
		if(blind_read_key(j) == key) { return true; }
	}
	return false;
}

template <size_t log_size, size_t header_size, size_t block_size, typename key_type, typename... Ts>
auto LeafDS<log_size, header_size, block_size, key_type, Ts...>::get_sorted_block_copy(size_t block_idx) {
	assert(block_idx < num_blocks);
#if DEBUG_PRINT
	printf("get sorted block copy of block %lu\n", block_idx);
#endif
	size_t elts_in_block_copy = 0;
	// now find the corresponding block
	std::array<uint8_t, SOA_type::get_size_static(block_size)> block_copy = {0};
	
	// copy only if its not in the delete log
	if (!is_in_delete_log(blind_read_key(header_start + block_idx))) {
#if DEBUG_PRINT
		printf("\tcopy[0] = %lu\n", blind_read_key(header_start + block_idx));
#endif
		blind_write_array(block_copy.data(), block_size, blind_read(header_start + block_idx), 0);
		elts_in_block_copy++;
	}

	size_t elts_in_block = count_block(block_idx);
	// now copy in the rest of the block
	for(size_t i = 0; i < elts_in_block; i++) {
		// copy only if its not in the delete log
		size_t idx = blocks_start + block_idx * block_size + i;
		key_type block_key = blind_read_key(idx);
		if(!is_in_delete_log(block_key)) {
#if DEBUG_PRINT
			printf("\tcopy[%lu] = %lu\n", elts_in_block_copy, block_key);
#endif
			blind_write_array(block_copy.data(), block_size, blind_read(idx), elts_in_block_copy);
			elts_in_block_copy++;
		}
	}
#if DEBUG_PRINT
	printf("elts in copy = %lu\n", elts_in_block_copy);
#endif
	sort_array_range(block_copy.data(), block_size, 0, elts_in_block_copy);
	return std::make_pair(block_copy, elts_in_block_copy);
}

// return a vector of element type
template <size_t log_size, size_t header_size, size_t block_size, typename key_type, typename... Ts>
auto LeafDS<log_size, header_size, block_size, key_type, Ts...>::unsorted_range(key_type start, key_type end) {
#if DEBUG_PRINT
	printf("\n\n*** unsorted range [%lu, %lu] ***\n", start, end);
#endif
	// first apply it to everything in the insert log
	std::vector<element_type> output;
	// make a copy of the insert
	std::array<uint8_t, SOA_type::get_size_static(log_size)> log_copy = {0};
	std::array<uint8_t, SOA_type::get_size_static(log_size)> deletes_copy = {0};

	for(size_t i = 0; i < num_inserts_in_log; i++) {
		blind_write_array(log_copy.data(), log_size, blind_read(i), i);
	}
	sort_array_range(log_copy.data(), log_size, 0, num_inserts_in_log);

	// copy all deletes into a log
	size_t dest = 0;
	for(size_t i = log_size - 1; i > log_size - num_deletes_in_log - 1; i++) {
		blind_write_array(deletes_copy.data(), log_size, blind_read(i), dest);
		dest++;
	}

	// sort delete log
	assert(dest == num_deletes_in_log);
	sort_array_range(deletes_copy.data(), log_size, 0, num_deletes_in_log);

	// pre-process per block if something exists in the range, and the number of elts in the log that fall in the range
	unsigned short log_per_block[num_blocks] = {0};
	unsigned short deletes_per_block[num_blocks] = {0};

	size_t block_idx = 0;
	key_type block_start = blind_read_key(header_start + block_idx);
	key_type block_end = blind_read_key(header_start + block_idx + 1);
	size_t i = 0;
	while(i < num_inserts_in_log) {
		key_type log_key = blind_read_key_array(log_copy.data(), log_size, i);
		if (log_key >= block_start && log_key < block_end) {
			log_per_block[block_idx]++;
			// TODO: make this do the function thing
			if (log_key >= start && log_key <= end) {
				output.push_back(blind_read_array(log_copy.data(), log_size, i));
			}
			i++;
		} else { // otherwise, advance the block
			assert(log_key > block_end);
			block_idx++;
			
			block_start = blind_read_key(header_start + block_idx);
			if (block_idx == num_blocks - 1) {
				block_end = std::numeric_limits<key_type>::max();
			} else {
				block_end = blind_read_key(header_start + block_idx + 1);
			}
		}	
	}

	i = 0;
	block_start = blind_read_key(header_start + block_idx);
	block_end = blind_read_key(header_start + block_idx + 1);

	while(i < num_deletes_in_log) {
		key_type delete_key = blind_read_key_array(deletes_copy.data(), log_size, i);
		if (delete_key >= block_start && delete_key < block_end) {
			deletes_per_block[block_idx]++;
			i++;
		} else { // otherwise, advance the block
			assert(delete_key > block_end);
			block_idx++;
			
			block_start = blind_read_key(header_start + block_idx);
			if (block_idx == num_blocks - 1) {
				block_end = std::numeric_limits<key_type>::max();
			} else {
				block_end = blind_read_key(header_start + block_idx + 1);
			}
		}	
	}
	// do prefix sum
#if DEBUG_PRINT
	printf("\tlog_per_block[0] = %hu, delete_per_block[0] = %hu\n", log_per_block[0], deletes_per_block[0]);
#endif
	for(i = 1; i < num_blocks; i++) {
		log_per_block[i] += log_per_block[i-1];
#if DEBUG_PRINT
		printf("\tlog_per_block[%lu] = %hu, deletes_per_block[%lu] = %hu\n", i, log_per_block[i], i, deletes_per_block[i]);
#endif
		deletes_per_block[i] += deletes_per_block[i-1];
	}
	assert(log_per_block[num_blocks - 1] == num_inserts_in_log);
	assert(deletes_per_block[num_blocks - 1] == num_deletes_in_log);

	// go through blocks, only checking for skips if there was anything in the block
	size_t start_block = find_block(start);
	size_t end_block = find_block(end);
#if DEBUG_PRINT
	printf("start block = %lu, start header = %lu, end block  = %lu, end header = %lu\n", start_block, blind_read_key(header_start + start_block), end_block, blind_read_key(header_start + end_block));
#endif
	size_t delete_start, delete_end, log_start, log_end, j;
	for(block_idx = start_block; block_idx <= end_block; block_idx++) {
		auto num_elts_in_block = count_block(block_idx);
		auto block_start = get_block_range(block_idx).first;
#if DEBUG_PRINT
		printf("block idx = %lu, num elts = %hu\n", block_idx, num_elts_in_block);
#endif
		size_t log_overlap = log_per_block[block_idx];
		assert(log_overlap <= num_inserts_in_log);
		size_t delete_overlap = deletes_per_block[block_idx];
		assert(delete_overlap <= num_deletes_in_log);
		if (block_idx > 0) { 
			log_overlap = log_per_block[block_idx] - log_per_block[block_idx - 1]; 
			delete_overlap = deletes_per_block[block_idx] - deletes_per_block[block_idx - 1];
		}
#if DEBUG_PRINT
		printf("log overlap %lu, delete overlap %lu\n", log_overlap, delete_overlap);
#endif
		if (log_overlap) {
			if (delete_overlap) {
				delete_start = 0;
				delete_end = deletes_per_block[block_idx]; 
				log_start = 0;
				log_end = log_per_block[block_idx]; 

				if (block_idx > 0) { 
					log_start = log_per_block[block_idx-1];
					delete_start = deletes_per_block[block_idx-1];
				}
				// check against both deletes and log
				for(i = block_start; i < block_start + num_elts_in_block; i++) {
					key_type key = blind_read_key(i);
					for(j = log_start; j < log_end; j++) {
						if (blind_read_key_array(log_copy.data(), log_size, j) == key) {
							continue;
						}
					}
					for(j = delete_start; j < delete_end; j++) {
						if (blind_read_key_array(deletes_copy.data(), log_size, j) == key) {
							continue;
						}
					}
					output.push_back(blind_read(i));
				}
			} else {
				// just check against log
				// start and end in log
				log_start = 0;
				log_end = log_per_block[block_idx]; 
				if (block_idx > 0) { log_start = log_per_block[block_idx-1]; }
#if DEBUG_PRINT
				printf("\tlog range [%lu, %lu)\n", log_start, log_end);
#endif
				assert(log_end <= log_size);
				assert(log_start <= log_size);
				// add header
				bool add_header = false;

				// add if it is in the range
				if (blind_read_key(header_start + block_idx) >= start && blind_read_key(header_start + block_idx) <= end) { 
					add_header = true;
					for(j = log_start; j < log_end; j++) {
						if (blind_read_key_array(log_copy.data(), log_size, j) == blind_read_key(header_start + block_idx)) {
							add_header = false;
						}
					}
				}
				if (add_header) { output.push_back(blind_read(header_start + block_idx)); }
				// then do the rest of the block
				for(i = block_start; i < block_start + num_elts_in_block; i++) {
					key_type key = blind_read_key(i);
#if DEBUG_PRINT
					printf("\t\tkey = %lu\n", key);
#endif
					for(j = log_start; j < log_end; j++) {
						if (blind_read_key_array(log_copy.data(), log_size, j) == key) {
#if DEBUG_PRINT
							printf("\t\t\tfound in log[%lu] = %lu\n", j, blind_read_key_array(log_copy.data(), log_size, j));
#endif
							continue;
						}
					}
					if(key >= start && key <= end) {
#if DEBUG_PRINT
						printf("\t\t\tadd %lu to output, not found in log\n", key);
#endif
						output.push_back(blind_read(i)); 
					}
				}
			}
		} else {
			assert(!log_overlap);
			if (delete_overlap) { // check against deletes 
				delete_start = 0;
				delete_end = deletes_per_block[block_idx]; 
				if (block_idx > 0) { delete_start = deletes_per_block[block_idx-1]; }
				for(i = block_start; i < block_start + num_elts_in_block; i++) {
					key_type key = blind_read_key(i);
					for(j = delete_start; j < delete_end; j++) {
						if (blind_read_key_array(deletes_copy.data(), log_size, j) == key) {
							continue;
						}
					}
					if (blind_read_key(i) >= start && blind_read_key(i) <= end) {
						output.push_back(blind_read(i));
					}
				}
			} else { // no overlap with either insert or delete log
				// first do the header
				if (blind_read_key(header_start + block_idx) >= start && blind_read_key(header_start + block_idx) <= end) { output.push_back(blind_read(header_start + block_idx)); }
				// then do the rest of the block
				for(i = block_start; i < block_start + num_elts_in_block; i++) {
					if(blind_read_key(i) >= start && blind_read_key(i) <= end) {
#if DEBUG_PRINT
						printf("\tno overlap, adding %lu to output\n", blind_read_key(i));
#endif
						output.push_back(blind_read(i));
					}
				}
			}
		}
#if DEBUG_PRINT
		printf("after processed block %zu\n", block_idx);
		for(i = 0; i < output.size(); i++) {
			printf("\tout[%lu] = %lu\n", i, std::get<0>(output[i]));
		}
#endif
	}
#if DEBUG_PRINT
	printf("on return, output size = %lu\n", output.size());
#endif
	return output;
}

// return a vector of element type
// TODO: make this apply function f to everything in the range
template <size_t log_size, size_t header_size, size_t block_size, typename key_type, typename... Ts>
auto LeafDS<log_size, header_size, block_size, key_type, Ts...>::sorted_range(key_type start, size_t length) {
#if DEBUG_PRINT
	printf("\n\n*** sorted range starting at %lu of length %lu ***\n", start, length);
#endif

	// copy log out-of-place and sort it
	std::vector<element_type> output;
	std::array<uint8_t, SOA_type::get_size_static(log_size)> log_copy = {0};
	for(size_t i = 0; i < num_inserts_in_log; i++) {
		blind_write_array(log_copy.data(), log_size, blind_read(i), i);
	}

	sort_array_range(log_copy.data(), log_size, 0, num_inserts_in_log);

	// move pointer to first elt at least start
	size_t log_ptr = 0;
	while(log_ptr < num_inserts_in_log && blind_read_key_array(log_copy.data(), log_size, log_ptr) < start) {
#if DEBUG_PRINT
		printf("moving log ptr, log[%lu] = %lu\n", log_ptr, blind_read_key_array(log_copy.data(), log_size, log_ptr));
#endif
		log_ptr++;
	}

	// make a copy of the block that the start key is in
	size_t block_idx = find_block(start);
	auto ret = get_sorted_block_copy(block_idx);
	std::array<uint8_t, SOA_type::get_size_static(block_size)> block_copy = ret.first;
	size_t elts_in_block_copy = ret.second;

	size_t block_ptr = 0;
	while(block_ptr < elts_in_block_copy && blind_read_key_array(block_copy.data(), block_size, block_ptr) < start) {
		block_ptr++;
	}

	// actually we should have been in the next one, key falls between end of block i and start of i+1
	if (block_ptr == elts_in_block_copy && block_idx < num_blocks) {
		block_idx++;
		ret = get_sorted_block_copy(block_idx);
		block_copy = ret.first;
		elts_in_block_copy = ret.second;
		block_ptr = 0;
	}
	// merge 
	while(log_ptr < num_inserts_in_log && block_ptr < elts_in_block_copy) {
		key_type log_key = blind_read_key_array(log_copy.data(), log_size, log_ptr);
		key_type block_key = blind_read_key_array(block_copy.data(), block_size, block_ptr);
		assert(output.size() < length);
		assert(log_key >= start);
		assert(block_key >= start);
		if (log_key == block_key) { // duplicate in log and blocks
			// log is the more recent one
			output.push_back(blind_read_array(log_copy.data(), log_size, log_ptr));
			log_ptr++;

		} else if (log_key < block_key) {
#if DEBUG_PRINT
			printf("\toutput[%lu] = %lu from log\n", output.size(), log_key);
#endif
			output.push_back(blind_read_array(log_copy.data(), log_size, log_ptr));
			log_ptr++;
		} else {
#if DEBUG_PRINT
			printf("\toutput[%lu] = %lu from blocks\n", output.size(), block_key);
#endif
			output.push_back(blind_read_array(block_copy.data(), block_size, block_ptr));
			block_ptr++;
			// if we ran out of block, copy in the next one
			if (block_ptr == elts_in_block_copy) {
				block_idx++;
				if (block_idx == num_blocks) { break; }
				ret = get_sorted_block_copy(block_idx);
				block_copy = ret.first;
				elts_in_block_copy = ret.second;
				block_ptr = 0;
#if DEBUG
				for(size_t i = 0; i < elts_in_block_copy; i++) {
					tbassert(blind_read_key_array(block_copy.data(), block_size, i) > start, "block_copy[%lu] = %lu, start = %lu\n", i, blind_read_key_array(block_copy.data(), block_size, i), start);
				}
#endif
			}
		}
		if (output.size() == length) { return output; }
	}
	assert(output.size() < length);
	// cleanup with log
	while(log_ptr < num_inserts_in_log) {
#if DEBUG
		key_type log_key = blind_read_key_array(log_copy.data(), log_size, log_ptr);
#if DEBUG_PRINT
		printf("\toutput[%lu] = %lu from log\n", output.size(), log_key);
#endif
		assert(output.size() < length);
		assert(log_key >= start);
#endif
		output.push_back(blind_read_array(log_copy.data(), log_size, log_ptr));
		log_ptr++;
		if (output.size() == length) { return output; }
	}
	assert(output.size() < length);
	// cleanup with blocks
	
	while(block_ptr < elts_in_block_copy && block_idx < num_blocks) {
#if DEBUG_PRINT
		printf("\toutput[%lu] = %lu from log\n", output.size(), blind_read_key_array(block_copy.data(), block_size, block_ptr));
#endif
		output.push_back(blind_read_array(block_copy.data(), block_size, block_ptr));
		block_ptr++;
		if (output.size() == length) { return output; }
		// go to the next block if we reached the end of this one
		if (block_ptr == elts_in_block_copy) {
				block_idx++;
				if (block_idx == num_blocks) { break; }
				ret = get_sorted_block_copy(block_idx);
				block_copy = ret.first;
				elts_in_block_copy = ret.second;
#if DEBUG
				for(size_t i = 0; i < elts_in_block_copy; i++) {
					assert(blind_read_key_array(block_copy.data(), block_size, i) > start);
				}
#endif
				block_ptr = 0;
		}
	}

	return output;
}

// split for b+-tree
template <size_t log_size, size_t header_size, size_t block_size, typename key_type, typename... Ts>
key_type LeafDS<log_size, header_size, block_size, key_type, Ts...>::split(LeafDS<log_size, header_size, block_size, key_type, Ts...>* right) {
#if DEBUG_PRINT
	printf("\n*** SPLIT ***\n");
	print();
#endif
	// flush log to blocks
	sort_range(0, num_inserts_in_log);

	if (num_inserts_in_log > 0 && blind_read_key(0) < get_min_block_key()) {
#if DEBUG_PRINT
		printf("first key in log = %lu, min block key = %lu\n", blind_read_key(0), get_min_block_key());
#endif
		size_t j = blocks_start + block_size;
		// find the first zero slot in the block
		SOA_type::template map_range_with_index_static(array.data(), N, [&j](auto index, auto key) {
			if (key == 0) {
				j = std::min(index, j);
			}
		}, blocks_start, blocks_start + block_size);
#if DEBUG_PRINT
		printf("first empty slot = %lu\n", j);
		print();
#endif

		// put the old min header in the block where it belongs
		// src = header start, dest = i
		copy_src_to_dest(header_start, j);
		// make min elt the new first header
		copy_src_to_dest(0, header_start);
	}

	flush_log_to_blocks(num_inserts_in_log);
	num_inserts_in_log = 0;

	// sort blocks
	// count the number of elements in each block
	unsigned short count_per_block[num_blocks];
	for (size_t i = 0; i < num_blocks; i++) {
		count_per_block[i] = count_block(i);
	}

	// sort the blocks
	for (size_t i = 0; i < num_blocks; i++) {
		auto block_range = get_block_range(i);
		sort_range(block_range.first, block_range.first + count_per_block[i]);
	}

	// merge all elts into sorted order
	std::vector<element_type> buffer;
	size_t blocks_ptr = header_start;
	size_t cur_block = 0;
	size_t start_of_cur_block = 0;
	while(cur_block < num_blocks) {
#if DEBUG
		ASSERT(blind_read_key(blocks_ptr) != NULL_VAL, "block ptr %lu\n", blocks_ptr);
#endif
		
		buffer.push_back(blind_read(blocks_ptr));
		advance_block_ptr(&blocks_ptr, &cur_block, &start_of_cur_block, count_per_block);
	}
#if DEBUG
	assert(buffer.size() < num_blocks * block_size);
#endif
	num_elts_total = buffer.size();

#if DEBUG_PRINT
	printf("*** BUFFER ***\n");
	for(size_t i = 0; i < buffer.size(); i++) {
		printf("%u\t", std::get<0>(buffer[i]));
	}
	printf("\n");
#endif

#if DEBUG
	// at this point the buffer should be sorted
	for (size_t i = 1; i < buffer.size(); i++) {
		assert(has(std::get<0>(buffer[i])));
		ASSERT(std::get<0>(buffer[i]) > std::get<0>(buffer[i-1]), "buffer[%lu] = %u, buffer[%lu] = %u\n", i-1, std::get<0>(buffer[i-1]), i, std::get<0>(buffer[i]));
	}
#endif
	clear_range(0, N);

	// pick a midpoint to return
#if DEBUG_PRINT
	printf("buffer size = %lu\n", buffer.size());
#endif
	size_t midpoint = buffer.size() / 2;	
	size_t left_size = midpoint;
	size_t elts_per_block = left_size / num_blocks;
	size_t remainder = left_size % num_blocks;
	// put the first half of the array into left, put the second half into right
	int cur_idx = 0;
	// size_t idx_in_block = 0;
#if DEBUG_PRINT
  printf("left size = %lu, elts per block = %lu, remainder = %lu\n", left_size, elts_per_block, remainder);
#endif
	for(size_t i = 0; i < num_blocks; i++) {
		assert(cur_idx < left_size);
		blind_write(buffer[cur_idx], header_start + i);
#if DEBUG_PRINT
		printf("\twrote buffer[%lu] = %lu as header %lu\n", cur_idx, std::get<0>(buffer[cur_idx]), i);
#endif
		size_t j;
		for(j = 1; j < elts_per_block + (i < remainder); j++) {
#if DEBUG_PRINT
			printf("\twrote buffer[%lu] = %lu at position %lu \n", cur_idx + j, std::get<0>(buffer[cur_idx + j]), blocks_start + i*block_size + j -1);	
#endif
			blind_write(buffer[cur_idx + j], blocks_start + i * block_size + j - 1);
		}
		cur_idx += j;
#if DEBUG_PRINT
		printf("cur idx at end of left loop %lu\n", cur_idx);
#endif
	}

	num_elts_total = left_size;
	// do the same for right side
	size_t right_size = buffer.size() - midpoint;
	elts_per_block = right_size / num_blocks;	
	remainder = right_size % num_blocks;
#if DEBUG_PRINT	
	printf("right size = %lu, elts per block = %lu\n", left_size, elts_per_block);
#endif
	for(size_t i = 0; i < num_blocks; i++) {
		assert(cur_idx < buffer.size());
		right->blind_write(buffer[cur_idx], header_start + i);
		size_t j;
		for(j = 1; j < elts_per_block + (i < remainder); j++) {
			right->blind_write(buffer[cur_idx + j], blocks_start + i * block_size + j - 1);
		}
		cur_idx += j;
#if DEBUG_PRINT
		printf("cur idx at end of right loop = %lu\n", cur_idx);
#endif
	}
	assert(cur_idx == buffer.size());
	right->num_elts_total = right_size;

	// end
	key_type mid_elt = std::get<0>(buffer[midpoint - 1]);
#if DEBUG_PRINT
	printf("mid elt = %lu\n", mid_elt);
	printf("\n\n**LEFT**\n");
	print();
	printf("\n\n**RIGHT**\n");
	right->print();
#endif

	return mid_elt;
}

//! Merge two leaf nodes. The function moves all key/data pairs from right
//! to left and sets right's slotuse to zero. The right slot is then removed
//! by the calling parent node.
// Assumes that right's log_size, header_size, and block_size is the same as this
template <size_t log_size, size_t header_size, size_t block_size, typename key_type, typename... Ts>
void LeafDS<log_size, header_size, block_size, key_type, Ts...>::merge(LeafDS<log_size, header_size, block_size, key_type, Ts...>* right) {
	// Flush existing deletes in right leaf to blocks
	if (right->num_deletes_in_log > 0 && right->blind_read_key(right->header_start) != 0) {
		// right->print();
		assert(right->blind_read_key(right->header_start) != 0);
		right->sort_range(log_size - right->num_deletes_in_log, log_size);
		if (right->delete_from_header()) {
			printf("deleting from right header, stripping and deleting");
			right->strip_deletes_and_redistrib();
		} else {
			right->flush_deletes_to_blocks();
		}
	}
	right->clear_range(log_size - right->num_deletes_in_log, log_size);
	right->num_deletes_in_log = 0;

	// Insert existing items from right into left (no need to sort them beforehand)

	// Insert into left from right log
	size_t log_ptr = 0;
	while (log_ptr < right->num_inserts_in_log) {
		insert(right->blind_read(log_ptr));
		log_ptr++;
	}

	// Count + sort the blocks
	unsigned short count_per_block[right->num_blocks];
	for (size_t i = 0; i < right->num_blocks; i++) {
		count_per_block[i] = right->count_block(i);
	}
	
	// Insert into left from right blocks, handle edge case of pre-first insert flush
	size_t blocks_ptr = header_start;
	size_t cur_block = 0;
	size_t start_of_cur_block = 0;
	if (right->blind_read_key(blocks_ptr) != 0 || cur_block != 0) {
		while (cur_block < right->num_blocks) {
			// printf("inserting elt: %lu\n", right->blind_read_key(blocks_ptr));
			assert(right->blind_read_key(blocks_ptr) != 0);
			insert(right->blind_read(blocks_ptr));
			right->advance_block_ptr(&blocks_ptr, &cur_block, &start_of_cur_block, count_per_block);
		}
	}

	// Clear the right leaf
	right->clear_range(0, right->N);
	right->num_elts_total = 0;
	right->num_inserts_in_log = 0;
	return;
}

// assumes manual_num_elts == true size of leafds
template <size_t log_size, size_t header_size, size_t block_size, typename key_type, typename... Ts>
void LeafDS<log_size, header_size, block_size, key_type, Ts...>::get_max_2(key_type* max_key, key_type* second_max_key, int manual_num_elts) {
	*max_key = get_key_at_sorted_index(manual_num_elts - 1);
	*second_max_key = get_key_at_sorted_index(manual_num_elts - 2);
	return;
}


// Balance two leaf nodes. The function moves key/data pairs from right to
// left so that both nodes are equally filled. The parent node is updated
// if possible.
// assumes shiftnum < num_elts, value of right's log_size, header_size, block_size is same as this
template <size_t log_size, size_t header_size, size_t block_size, typename key_type, typename... Ts>
void LeafDS<log_size, header_size, block_size, key_type, Ts...>::shift_left(LeafDS<log_size, header_size, block_size, key_type, Ts...>* right, int shiftnum) {
	// Flush deletes to right leaf blocks
	if (right->num_deletes_in_log > 0 && right->blind_read_key(right->header_start) != 0) {
		right->sort_range(log_size - right->num_deletes_in_log, log_size);
		if (right->delete_from_header()) {
			printf("deleting from right header, stripping and deleting");
			right->strip_deletes_and_redistrib();
		} else {
			right->flush_deletes_to_blocks();
		}
	}
	right->clear_range(log_size - right->num_deletes_in_log, log_size);
	right->num_deletes_in_log = 0;

	// Iterate over right leaf elems up to shiftnum, insert into left
	std::vector<key_type> elts_to_shift;
	for (int i = 0; i < shiftnum; i++) {
		auto ith_elem = right->blind_read(right->get_element_at_sorted_index(i));
		elts_to_shift.push_back(std::get<0>(ith_elem));
		insert(ith_elem);
	}

	// Iterate over right leaf elems up to shiftnum, delete from right
	for (int i = 0; i < shiftnum; i++) {
		right->remove(elts_to_shift[i]);
	}
	return;
}

// Balance two leaf nodes. The function moves key/data pairs from left to
// right so that both nodes are equally filled. The parent node is updated
// if possible.
// TODO: for now, since get_num_elements() is unsafe, pass in the num of elements in left so we can index into it
template <size_t log_size, size_t header_size, size_t block_size, typename key_type, typename... Ts>
void LeafDS<log_size, header_size, block_size, key_type, Ts...>::shift_right(LeafDS<log_size, header_size, block_size, key_type, Ts...>* left, int shiftnum, int left_num_elts) {
	// Flush deletes to left leaf blocks
	if (left->num_deletes_in_log > 0 && left->blind_read_key(left->header_start) != 0) {
		left->sort_range(log_size - left->num_deletes_in_log, log_size);
		if (left->delete_from_header()) {
			printf("deleting from left header, stripping and deleting");
			left->strip_deletes_and_redistrib();
		} else {
			left->flush_deletes_to_blocks();
		}
	}
	left->clear_range(log_size - left->num_deletes_in_log, log_size);
	left->num_deletes_in_log = 0;

	// Iterate over left leaf elems from -shiftnum to end, insert into right
	std::vector<key_type> elts_to_shift;
	for (int i = left_num_elts - shiftnum; i < left_num_elts; i++) {
		auto ith_elem = left->blind_read(left->get_element_at_sorted_index(i));
		elts_to_shift.push_back(std::get<0>(ith_elem));
		insert(ith_elem);
	}

	// Iterate over right leaf elems up to shiftnum, delete from right
	for (int i = 0; i < shiftnum; i++) {
		left->remove(elts_to_shift[i]);
	}
	return;
}

// Assumes i < get_num_elts
template <size_t log_size, size_t header_size, size_t block_size, typename key_type, typename... Ts>
key_type& LeafDS<log_size, header_size, block_size, key_type, Ts...>::get_key_at_sorted_index(size_t i) {
	// flush delete log so we can do two ptr sorted check on inserts
	if (num_deletes_in_log > 0 && blind_read_key(header_start) != 0) {
		sort_range(log_size - num_deletes_in_log, log_size);
		if (delete_from_header()) {
#if DEBUG_PRINT
			printf("deleting from header, stripping and deleting");
#endif
			strip_deletes_and_redistrib();
		} else {
			flush_deletes_to_blocks();
		}
	}
	clear_range(log_size - num_deletes_in_log, log_size);
	num_deletes_in_log = 0;

	// Sort log
	sort_range(0, num_inserts_in_log);

	// Count + sort the blocks
	unsigned short count_per_block[num_blocks];
	for (size_t i = 0; i < num_blocks; i++) {
		count_per_block[i] = count_block(i);
	}

	for (size_t i = 0; i < num_blocks; i++) {
		auto block_range = get_block_range(i);
		sort_range(block_range.first, block_range.first + count_per_block[i]);
	}

	// two pointer search for key at sorted index
	size_t log_ptr = 0;
	size_t blocks_ptr = header_start;
	size_t cur_block = 0;
	size_t start_of_cur_block = 0;
	size_t curr_index = 0;
	while ((log_ptr < num_inserts_in_log || cur_block < num_blocks) && curr_index < i) {
		// check which ptr to increment
		if (log_ptr < num_inserts_in_log && cur_block < num_blocks) {
			if (blind_read_key(log_ptr) <= blind_read_key(blocks_ptr)) {
				log_ptr++;
			} else if (blind_read_key(blocks_ptr) == 0 && cur_block == 0) {
				// edge case of first header block being 0 pre-first flush, we still want to increment log_ptr here
				log_ptr++;
			} else {
				advance_block_ptr(&blocks_ptr, &cur_block, &start_of_cur_block, count_per_block);
			}
		} else if (log_ptr < num_inserts_in_log) {
			log_ptr++;
		} else {
			advance_block_ptr(&blocks_ptr, &cur_block, &start_of_cur_block, count_per_block);
		}
		curr_index++;
	}
#if DEBUG_PRINT
	auto log_ptr_val = blind_read_key(log_ptr);
	auto blocks_ptr_val = blind_read_key(blocks_ptr);
	printf("looking for index %lu\n", i);
	printf("\tlogptr %lu, val %lu\n", log_ptr, log_ptr_val);
	printf("\tblocks_ptr %lu, val %lu\n", blocks_ptr, blocks_ptr_val);
#endif
	if (log_ptr < num_inserts_in_log && cur_block < num_blocks) {
		if (blind_read(log_ptr) <= blind_read(blocks_ptr)) {
			return std::get<0>(blind_read(log_ptr));
		} else if (blind_read_key(blocks_ptr) == 0 && cur_block == 0) {
			// edge case of first header block being 0 pre-first flush
			return std::get<0>(blind_read(log_ptr));
		} else {
			return std::get<0>(blind_read(blocks_ptr));
		}
	} else if (log_ptr < num_inserts_in_log) {
		return std::get<0>(blind_read(log_ptr));
	} else {
		return std::get<0>(blind_read(blocks_ptr));
	}
}

// Assumes i < get_num_elts
// returns index to get elem using blind_read()
template <size_t log_size, size_t header_size, size_t block_size, typename key_type, typename... Ts>
size_t LeafDS<log_size, header_size, block_size, key_type, Ts...>::get_element_at_sorted_index(size_t i) {
	// flush delete log so we can do two ptr sorted check on inserts
	if (num_deletes_in_log > 0) {
		sort_range(log_size - num_deletes_in_log, log_size);
		if (delete_from_header()) {
			printf("deleting from header, stripping and deleting");
			strip_deletes_and_redistrib();
		} else {
			flush_deletes_to_blocks();
		}
	}
	clear_range(log_size - num_deletes_in_log, log_size);
	num_deletes_in_log = 0;

	// Sort log
	sort_range(0, num_inserts_in_log);

	// Count + sort the blocks
	unsigned short count_per_block[num_blocks];
	for (size_t i = 0; i < num_blocks; i++) {
		count_per_block[i] = count_block(i);
	}

	for (size_t i = 0; i < num_blocks; i++) {
		auto block_range = get_block_range(i);
		sort_range(block_range.first, block_range.first + count_per_block[i]);
	}

	// two pointer search for key at sorted index
	size_t log_ptr = 0;
	size_t blocks_ptr = header_start;
	size_t cur_block = 0;
	size_t start_of_cur_block = 0;
	size_t curr_index = 0;
	while ((log_ptr < num_inserts_in_log || cur_block < num_blocks) && curr_index < i) {
		// check which ptr to increment
		if (log_ptr < num_inserts_in_log && cur_block < num_blocks) {
			if (blind_read_key(log_ptr) <= blind_read_key(blocks_ptr)) {
				log_ptr++;
			} else if (blind_read_key(blocks_ptr) == 0 && cur_block == 0) {
				// edge case of first header block being 0 pre-first flush, we still want to increment log_ptr here
				log_ptr++;
			} else {
				advance_block_ptr(&blocks_ptr, &cur_block, &start_of_cur_block, count_per_block);
			}
		} else if (log_ptr < num_inserts_in_log) {
			log_ptr++;
		} else {
			advance_block_ptr(&blocks_ptr, &cur_block, &start_of_cur_block, count_per_block);
		}
		curr_index++;
	}
#if DEBUG_PRINT
	auto log_ptr_val = blind_read_key(log_ptr);
	auto blocks_ptr_val = blind_read_key(blocks_ptr);
	printf("looking for index %lu\n", i);
	printf("\tlogptr %lu, val %lu\n", log_ptr, log_ptr_val);
	printf("\tblocks_ptr %lu, val %lu\n", blocks_ptr, blocks_ptr_val);
#endif
	if (log_ptr < num_inserts_in_log && cur_block < num_blocks) {
		if (blind_read(log_ptr) <= blind_read(blocks_ptr)) {
			return log_ptr;
		} else if (blind_read_key(blocks_ptr) == 0 && cur_block == 0) {
			// edge case of first header block being 0 pre-first flush
			return log_ptr;
		} else {
			return blocks_ptr;
		}
	} else if (log_ptr < num_inserts_in_log) {
		return log_ptr;
	} else {
		return blocks_ptr;
	}
}

// TODO: fix, gives wrong sizes sometimes
template <size_t log_size, size_t header_size, size_t block_size, typename key_type, typename... Ts>
size_t LeafDS<log_size, header_size, block_size, key_type, Ts...>::get_num_elements() {
	// flush delete log
	if (num_deletes_in_log > 0) {
		sort_range(log_size - num_deletes_in_log, log_size);
		if (delete_from_header()) {
#if DEBUG_PRINT
			printf("deleting from header, stripping and deleting");
#endif
			strip_deletes_and_redistrib();
		} else {
			flush_deletes_to_blocks();
		}
	}
	clear_range(log_size - num_deletes_in_log, log_size);
	num_deletes_in_log = 0;

	// flush insert log
	if (num_inserts_in_log > 0) {
		sort_range(0, num_inserts_in_log); // sort inserts

		// if this is the first time we are flushing the log, just make the sorted log the header
		if (get_min_block_key() == 0) {
			if (num_deletes_in_log > 0) { // if we cannot fill the header
				clear_range(num_inserts_in_log, log_size); // clear deletes
				num_deletes_in_log = 0;
				return true;
			} else {
				for(size_t i = 0; i < log_size; i++) {
					SOA_type::get_static(array.data(), N, i + header_start) =
						SOA_type::get_static(array.data(), N, i);
				}
			}
		} else { // otherwise, there are some elements in the block / header part
			if(num_deletes_in_log > 0) {
				sort_range(num_inserts_in_log, log_size); // sort deletes
			}
			// if inserting min, swap out the first header into the first block
			if (blind_read_key(0) < get_min_block_key()) {
				size_t j = blocks_start + block_size;
				// find the first zero slot in the block
				// TODO: this didn't work (didn't find the first zero)
				SOA_type::template map_range_with_index_static(array.data(), N, [&j](auto index, auto key) {
					if (key == 0) {
						j = std::min(index, j);
					}
				}, blocks_start, blocks_start + block_size);

				// put the old min header in the block where it belongs
				// src = header start, dest = i
				copy_src_to_dest(header_start, j);

				// make min elt the new first header
				copy_src_to_dest(0, header_start);
				num_elts_total++;
			}
			// flush the log
			flush_log_to_blocks(num_inserts_in_log);
		}

		// clear insert log
		num_inserts_in_log = 0;
		clear_range(0, log_size);
	}

	// num_elts_total should be safe now
	// printf("count via count up elets : %lu ", count_up_elts());
	return num_elts_total;
}
