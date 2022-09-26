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
		return _mm512_cmp_epi32_mask(bcast, block, _MM_CMPINT_EQ);
	}

	static inline __mmask8 slot_mask_64(uint8_t * array, uint64_t key) {
		__m512i bcast = _mm512_set1_epi64(key);
		__m512i block = _mm512_loadu_si512((const __m512i *)(array));
		return _mm512_cmp_epi64_mask(bcast, block, _MM_CMPINT_EQ);
	}

	static inline mask_type slot_mask(uint8_t * array, key_type key) {
		if constexpr (std::is_same<key_type, uint32_t>::value) { // 32-bit keys
			return slot_mask_32(array, key);
		} else {
			return slot_mask_64(array, key);
		}
	}


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

	std::array<uint8_t, SOA_type::get_size_static(N)> array = {0};

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

  [[nodiscard]] size_t get_index_in_blocks(key_type e) const;

	// index of element e in the DS, N if not found
  [[nodiscard]] size_t get_index(key_type e) const;

  auto blind_read_key(uint32_t index) const {
    return std::get<0>(SOA_type::get_static(array.data(), N, index));
  }

	// min block header is the first elt in the header part
	auto get_min_block_key() const {
		return blind_read_key(header_start);
  }

  void blind_write(element_type e, uint32_t index) {
    SOA_type::get_static(array.data(), N, index) = e;
  }

  auto blind_read(uint32_t index) const {
    return SOA_type::get_static(array.data(), N, index);
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
    assert(num_to_flush < block_size);
		assert(num_to_flush >= 1);
#endif
#if DEBUG_PRINT
		printf("block %u, num to flush %u\n", i, num_to_flush);
#endif
    // write the header
    blind_write(buffer[num_so_far], header_start + i);
    num_to_flush--;
    num_so_far++;
#if DEBUG_PRINT
		printf("\tset buf[%u] = %u as header of block %u at pos %u\n", num_so_far, std::get<0>(buffer[num_so_far]), i, header_start + i);
#endif
    // write the rest into block
    size_t start = blocks_start + i * block_size;
    for(size_t j = 0; j < num_to_flush; j++) {
#if DEBUG
			assert(num_so_far < n);
#endif
#if DEBUG_PRINT
		printf("\tset buf[%u] = %u in block %u at pos %u\n", num_so_far, std::get<0>(buffer[num_so_far]), i, start + j);
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
		tbassert(blind_read_key(blocks_ptr) != NULL_VAL, "block ptr %lu\n", blocks_ptr);
#endif
#if DEBUG_PRINT
		printf("added %u at idx %lu to buffer\n", blind_read_key(blocks_ptr), blocks_ptr);
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
		tbassert(std::get<0>(buffer[i]) > std::get<0>(buffer[i-1]), "buffer[%lu] = %u, buffer[%lu] = %u\n", i-1, std::get<0>(buffer[i-1]), i, std::get<0>(buffer[i]));
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
		printf("%u\t", std::get<0>(log_to_flush[i]));
	}
	printf("\n");
#endif

	while(log_ptr < log_end && cur_block < num_blocks) {
		const key_type log_key = std::get<0>(log_to_flush[log_ptr]);
		const key_type block_key = blind_read_key(blocks_ptr);
#if DEBUG
		assert(log_key != block_key);
#endif
		if (log_key < block_key) {
#if DEBUG_PRINT
			printf("pushed %u from log to buffer\n", std::get<0>(log_to_flush[log_ptr]));
#endif
			buffer.push_back(log_to_flush[log_ptr]);
			log_ptr++;
		} else {
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
		tbassert(blind_read_key(blocks_ptr) != NULL_VAL, "block ptr %lu\n", blocks_ptr);
#endif
#if DEBUG_PRINT
		printf("added %u at idx %lu to buffer\n", blind_read_key(blocks_ptr), blocks_ptr);
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
		tbassert(std::get<0>(buffer[i]) > std::get<0>(buffer[i-1]), "buffer[%lu] = %u, buffer[%lu] = %u\n", i-1, std::get<0>(buffer[i-1]), i, std::get<0>(buffer[i]));
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

#if DEBUG
	// scalar version for debug
	size_t i = header_start + hint;
	assert(key >= blind_read_key(i));
	size_t correct_ret = hint;

	for( ; i < blocks_start; i++) {
		correct_ret += blind_read_key(i) <= key;
	}
	if (correct_ret == num_blocks || blind_read_key(header_start + correct_ret) > key) {
		correct_ret--;
	}
#endif

	// vector version
	size_t vector_start = header_start;
	size_t vector_end = blocks_start;
	size_t ret = 0;
	mask_type mask;
	for(; vector_start < vector_end; vector_start += keys_per_vector) {
		if constexpr (std::is_same<key_type, uint32_t>::value) { // 32-bit keys
			__m512i bcast = _mm512_set1_epi32(key);
			__m512i block = _mm512_loadu_si512((const __m512i *)(array.data() + vector_start * sizeof(key_type)));
			mask = _mm512_cmple_epi32_mask(block, bcast);
		} else {
			__m512i bcast = _mm512_set1_epi64(key);
			__m512i block = _mm512_loadu_si512((const __m512i *)(array.data() + vector_start * sizeof(key_type)));
			mask = _mm512_cmple_epi64_mask(block, bcast);
		}
		ret += __builtin_popcount(mask);
	}

	// TODO: can you do to the next line faster?
	if (ret == num_blocks || blind_read_key(header_start + ret) > key) {
		ret--;
	}


	ASSERT(ret == correct_ret, "got %lu, should be %lu\n", ret, correct_ret);
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
	assert(i - header_start - 1 == ret);
	// printf("elt %lu, original found %lu, new is %lu\n", key, i - header_start - 1, ret);
#endif

	return ret;
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
	auto start = typename LeafDS<log_size, header_size, block_size, key_type, Ts...>::SOA_type::Iterator(array.data(), N, start_idx);
	auto end = typename LeafDS<log_size, header_size, block_size, key_type, Ts...>::SOA_type::Iterator(array.data(), N, end_idx);

	std::sort(start, end, [](auto lhs, auto rhs) { return std::get<0>(typename SOA_type::T(lhs)) < std::get<0>(typename SOA_type::T(rhs)); } );
#if DEBUG
	// check sortedness
	for(size_t i = start_idx + 1; i < end_idx; i++) {
		assert(blind_read_key(i-1) < blind_read_key(i));
	}
#endif
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
bool LeafDS<log_size, header_size, block_size, key_type, Ts...>::update_in_range_if_exists(size_t start,
	size_t end, element_type e) {
#if DEBUG_PRINT
	printf("*** start = %lu, end = %lu ***\n", start, end);
#endif

	[[maybe_unused]] bool correct_answer = false;
	[[maybe_unused]] size_t test_i = start;
#if DEBUG
	// scalar version for correctness
	const key_type key = std::get<0>(e);

	for(; test_i < end; test_i++) { 
		if (key == get_key_array(test_i)) {
		#if DEBUG_PRINT
			printf("CORRECT FOUND KEY %u AT IDX %u\n", std::get<0>(e), test_i);
			print();
		#endif
			correct_answer = true;
			break;
		} else if (get_key_array(test_i) == NULL_VAL) {
			correct_answer = false;
			break;
		}
	}
#endif
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

	ASSERT((correct_answer == false), "searching for key %u in range [%lu, %lu)\n", std::get<0>(e), start, end);
	return false;
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
	tbassert(block_idx < num_blocks, "block idx %lu\n", block_idx);
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

#if DEBUG
	// count number of nonzero elts in this block
	uint64_t correct_count = 0;
	SOA_type::template map_range_static(array.data(), N, [&correct_count](auto key) {correct_count += key != 0;}, block_start, block_end);
#endif

	uint64_t num_zeroes = 0;
	mask_type mask;
	for(; block_start < block_end; block_start += keys_per_vector) {
		if constexpr (std::is_same<key_type, uint32_t>::value) { // 32-bit keys
			__m512i bcast = _mm512_set1_epi32(0);
			__m512i block = _mm512_loadu_si512((const __m512i *)(array.data() + block_start * sizeof(key_type)));
			mask = _mm512_cmpeq_epi32_mask(block, bcast);
		} else {
			__m512i bcast = _mm512_set1_epi64(0);
			__m512i block = _mm512_loadu_si512((const __m512i *)(array.data() + block_start * sizeof(key_type)));
			mask = _mm512_cmpeq_epi64_mask(block, bcast);
		}
		num_zeroes += __builtin_popcount(mask);
	}

	ASSERT(num_zeroes <= block_size, "counted zeroes %lu, block size %lu\n", num_zeroes, block_size);

	uint64_t count = block_size - num_zeroes;

	ASSERT(correct_count == count, "counting block %lu, got count %lu, should be %lu\n", block_idx, count, correct_count);
	return count;
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
		assert(key_to_flush >= blind_read_key(log_size));
#endif

		size_t block_idx = find_block_with_hint(key_to_flush, hint);
#if DEBUG
		assert(block_idx < num_blocks);
#endif
		// if it is in the header, update the header
		if (blind_read_key(header_start + block_idx) == key_to_flush) {
				copy_src_to_dest(i, header_start + block_idx);
#if DEBUG_PRINT
				printf("found duplicate in header idx %zu of elt %u\n", block_idx, key_to_flush);
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
				printf("\tflushing elt %u to block %lu, header %u\n", blind_read_key(i), block_idx, blind_read_key(header_start + block_idx));
#endif
				log_to_flush[num_to_flush] = blind_read(i);
				num_to_flush++;
				num_to_flush_per_block[block_idx]++;
			} else {
				num_elts_total--;
#if DEBUG_PRINT
				printf("found duplicate in block %u of elt %zu\n", block_idx, key_to_flush);
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
			print();
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
	tbassert(idx_in_log == num_to_flush, "flushed %lu, should have flushed %u\n", idx_in_log, num_to_flush);
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
			printf("found elt %u to delete in block %zu, idx %u\n", e, block_idx, i);
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
				sort_range(num_inserts_in_log, log_size); // sort deletes
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
				tbassert(i == j, "got %zu, should be %zu\n", j, i);
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
				flush_deletes_to_blocks();
			}
			flush_log_to_blocks(num_inserts_in_log);
		}

		// clear log
		num_inserts_in_log = 0;
		clear_range(0, log_size);
	}

	return true;
}


template <size_t log_size, size_t header_size, size_t block_size, typename key_type, typename... Ts>
inline void LeafDS<log_size, header_size, block_size, key_type, Ts...>::advance_block_ptr(size_t* blocks_ptr, size_t* cur_block, size_t* start_of_cur_block, unsigned short* count_per_block) const {
#if DEBUG_PRINT
		if (blind_read_key(*blocks_ptr) == NULL_VAL) {
			printf("null blocks ptr %lu\n", blocks_ptr);
			assert(false);
		}
		printf("pushed %u from blocks to buffer\n", blind_read_key(*blocks_ptr));
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
	printf("flushing deletes\n");
	print();
#endif

	// sort deletes
	sort_range(log_size, header_start);

	size_t hint = 0;
	// process the deletes
	for(size_t i = 0; i < num_deletes_in_log; i++) {
		key_type key_to_delete = blind_read_key(i + log_size);
		size_t block_idx = find_block_with_hint(key_to_delete, hint);

		// try to delete it from the blocks if it exists
		delete_from_block_if_exists(key_to_delete, block_idx);

		if (hint < block_idx) { hint = block_idx; }
	}

	// clear delete log
	// clear_range(log_size, header_start);

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
	sort_range(log_size, log_size + num_deletes_in_log);

	// two-finger strip of log from blocks/header
	size_t log_ptr = log_size;
	size_t blocks_ptr = header_start;
	size_t cur_block = 0;
	size_t start_of_cur_block = 0;
  size_t log_end = log_size + num_deletes_in_log;
	size_t buffer_ptr = 0;

  while(log_ptr < log_end && cur_block < num_blocks) {
    const key_type log_key = blind_read_key(log_ptr);
    const key_type block_key = blind_read_key(blocks_ptr);
		// if we are deleting this key
		if (log_key == block_key) {
#if DEBUG_PRINT
			printf("\tstrip %u from log\n", log_key);
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

	// num elts total is num_insert + from blocks
	// but there might be some repetitions between log/blocks
	// split up buffer into blocks if there is enough
	if (buffer_ptr > header_size) {
		global_redistribute_buffer(buffer, buffer_ptr);
	} else { // otherwise, put it in the header
#if DEBUG_PRINT
		printf("before small buffer case\n");
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
			} else if (log_key < buffer_key) {
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

		assert(out_ptr < log_size);

		clear_range(0, log_size);
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

		// clear rest of the DS
		clear_range(out_ptr, N);
	}
}

// return whether we are deleting from the header or not
template <size_t log_size, size_t header_size, size_t block_size, typename key_type, typename... Ts>
bool LeafDS<log_size, header_size, block_size, key_type, Ts...>::delete_from_header() {
	auto log_end = log_size + num_deletes_in_log;
	sort_range(log_size, log_end);

	// rebuild if we are deleting from the header
	size_t log_ptr = log_size;
	size_t header_ptr = header_start;

	while(log_ptr < log_end && header_ptr < header_start + header_size) {
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
			printf("found in insert log\n");
			print();
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
		auto result = find_key_in_range(num_inserts_in_log, num_deletes_in_log, e);
		if (!result.first) { // if not in delete log, add it
			num_deletes_in_log++;
			blind_write(e, log_size - num_deletes_in_log); // grow left
		}
	} else { // otherwise, just add it to the delete log
		num_deletes_in_log++;
		blind_write(e, log_size - num_deletes_in_log);
	}

	// now check if the log is full
	if (num_deletes_in_log + num_inserts_in_log == log_size) {
		// if the header is empty, the deletes just disappear
		// only do the flushing if there is stuff later in the DS
		if (get_min_block_key() != 0) {
			// if we are deleting from the header, do a global rewrite
			if (delete_from_header()) {
#if DEBUG_PRINT
				printf("deleting from header\n");
#endif
				// strip deletes and redistrib
				strip_deletes_and_redistrib();
			} else {
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
	// if less than current min, should not be in ds
	if (e < blind_read_key(header_start)) {
		return N;
	}
	assert( e >= blind_read_key(header_start) );
	size_t block_idx = find_block(e);
#if DEBUG_PRINT
	printf("\tin has for elt %u, find block returned %lu\n", e, block_idx);
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
	for(size_t i = log_size; i > log_size - num_deletes_in_log; i--) {
		if(blind_read_key(i) == e) {
			printf("found %u in delete log\n", e);
			return false;
		}
	}
	
	// otherwise search in insert log and rest of DS
	return (get_index(e) != N);
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
	printf("total num elts %lu\n", num_elts);
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
		printf("\nblock %lu\n", (i - blocks_start) / block_size);
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
	size_t skip_index[2*log_size];
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
	for(size_t i = log_size; i < log_size + num_deletes_in_log; i++) {
		key_type key = blind_read_key(i);
		size_t idx = get_index_in_blocks(key);
		if (idx < N) {
			skip_index[num_to_skip] = idx;
			num_to_skip++;
		}
	}

  static_assert(std::is_invocable_v<decltype(&F::operator()), F &, uint32_t,
                                    NthType<Is>...>,
                "update function must match given types");

	// map over insert log
	for (uint32_t i = 0; i < num_inserts_in_log; i++) {
    // auto index = get_key_array(i);
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
  for (uint32_t i = header_start; i < N; i++) {
    auto index = get_key_array(i);
		// skip over deletes
    if (index != NULL_VAL) {
			// skip if duplicated
			bool skip = false;
			for(uint32_t j = 0; j < num_to_skip; j++) {
				if(i == skip_index[j]) { skip = true; }
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
  uint64_t result = 0;

	size_t skip_index[2*log_size];
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
	for(size_t i = log_size; i < log_size + num_deletes_in_log; i++) {
		key_type key = blind_read_key(i);
		size_t idx = get_index_in_blocks(key);
		if (idx < N) {
			skip_index[num_to_skip] = idx;
			num_to_skip++;
		}
	}

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
