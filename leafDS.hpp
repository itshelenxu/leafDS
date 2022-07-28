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


private:
	static constexpr key_type NULL_VAL = {};

	static constexpr size_t num_blocks = header_size;
	static constexpr size_t N = log_size + header_size + header_size * num_blocks;

  static_assert(N != 0, "N cannot be 0");
	
	// start of each section
	static constexpr size_t header_start = log_size;
	static constexpr size_t blocks_start = log_size + header_size;

	// counters
	size_t num_inserts_in_log = 0;
	size_t num_elts_total = 0;

	// max elts allowed in data structure before split
	static constexpr size_t max_density = (int)( 9.0 / 10.0 * N );

	std::array<uint8_t, SOA_type::get_size_static(N)> array = {0};

  inline key_type get_key_array(uint32_t index) const {
    return std::get<0>(
        SOA_type::template get_static<0>(array.data(), N, index));
  }

	size_t count_up_elts() {
		size_t result = 0;
		for(size_t i = 0; i < N; i++) {
			result += (blind_read_key(i) != NULL_VAL);
		}
		return result;
	}

	// private helpers
	void update_val_at_index(element_type e, size_t index);
	void place_elt_at_index(element_type e, size_t index);
	void clear_range(size_t start, size_t end);
	size_t find_block(key_type key);
	size_t find_block_with_hint(key_type key, size_t hint);
	void global_redistribute(element_type* log_to_flush, size_t num_to_flush, unsigned short* count_per_block);
	void copy_src_to_dest(size_t src, size_t dest);
	void flush_log_to_blocks(size_t max_to_flush);
	void sort_log();
	void sort_range(size_t start_idx, size_t end_idx);
	inline std::pair<size_t, size_t> get_block_range(size_t block_idx);
	std::pair<bool, size_t> update_in_range_if_exists(size_t start, size_t end, element_type e);
	std::pair<bool, size_t> update_in_block_if_exists(element_type e);
	unsigned short count_block(size_t block_idx);
	void print_range(size_t start, size_t end);

public:
	size_t get_num_elts() const { return num_elts_total; }
	bool is_full() { return num_elts_total >= max_density; }
	void print();

	[[nodiscard]] uint64_t sum_keys();
	[[nodiscard]] uint64_t sum_keys_with_map();
	[[nodiscard]] uint64_t sum_keys_direct();
	template <bool no_early_exit, size_t... Is, class F> bool map(F f);
	// main top-level functions
	// given a key, return the index of the largest elt at least e
	[[nodiscard]] uint32_t search(key_type e) const;

	// insert e, return true if it was not there
  bool insert(element_type e);

	// remove e, return true if it was present
  bool remove(key_type e);

	// whether elt e was in the DS
  [[nodiscard]] bool has(key_type e);

  [[nodiscard]] size_t get_index_in_blocks(key_type e);

	// index of element e in the DS, N if not found
  [[nodiscard]] size_t get_index(key_type e);

  auto blind_read_key(uint32_t index) const {
    return std::get<0>(SOA_type::get_static(array.data(), N, index));
  }

	// min block header is the first elt in the header part
	auto get_min_block_key() {
		return blind_read_key(log_size);
  }

  void blind_write(element_type e, uint32_t index) {
    SOA_type::get_static(array.data(), N, index) = e;
  }

  auto blind_read(uint32_t index) {
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

// one of the blocks
// input: deduped log to flush, number of elements in the log, count of elements to flush to each block, count of elements per block
// merge all elements from blocks and log in sorted order in the intermediate buffer
// split them evenly amongst the blocks
// TODO: count global redistributes 
template <size_t log_size, size_t header_size, size_t block_size, typename key_type, typename... Ts>
void LeafDS<log_size, header_size, block_size, key_type, Ts...>::global_redistribute(element_type* log_to_flush, size_t num_to_flush, unsigned short* count_per_block) {
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
#if DEBUG_PRINT
			if (blind_read_key(blocks_ptr) == NULL_VAL) {
				printf("null blocks ptr %lu\n", blocks_ptr);
				// pr

				printf("\n***BUFFER***\n");
				for(size_t i = 0; i < buffer.size(); i++) {
					printf("%u\t", std::get<0>(buffer[i]));
				}
				printf("\n");
				assert(false);
			}
			printf("pushed %u from blocks to buffer\n", blind_read_key(blocks_ptr));
#endif
			buffer.push_back(blind_read(blocks_ptr));
			size_t prev_blocks_ptr = blocks_ptr;

			if (blocks_ptr < blocks_start) {
				start_of_cur_block = blocks_start + cur_block * block_size;

				// if we are in the header, go to the block if the block is nonempty
				if (blind_read_key(start_of_cur_block) != NULL_VAL) {
					blocks_ptr = start_of_cur_block;
				} else { // if this block is empty, move to the next header
					cur_block++;
					blocks_ptr++;
				}
			} else if (blocks_ptr == start_of_cur_block + count_per_block[cur_block] - 1) {
				// if we have merged in this entire block, go back to the header
				cur_block++;
				blocks_ptr = header_start + cur_block;
			} else { // if we are still in this block, keep going
				blocks_ptr++;
			}
#if DEBUG
			assert(prev_blocks_ptr != blocks_ptr); // made sure we advanced
#endif
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
		// if it was in the header, go to the block if the block is nonempty
		if (blocks_ptr < blocks_start) {
			start_of_cur_block = blocks_start + cur_block * block_size;
			if (blind_read_key(start_of_cur_block) != NULL_VAL) {
				blocks_ptr = start_of_cur_block;
			} else {
				cur_block++;
				blocks_ptr++;
			}
		} else if (blocks_ptr == start_of_cur_block + count_per_block[cur_block] - 1) {
			// if we have merged in this entire block, go back to the header
			cur_block++;
			blocks_ptr = header_start + cur_block;
		} else {
			// if we are still in this block, keep going
			blocks_ptr++;
		}
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

	clear_range(header_start, N); // clear the header and blocks

	// split up the buffer into blocks
	size_t per_block = buffer.size() / num_blocks;
	size_t remainder = buffer.size() % num_blocks;
	size_t num_so_far = 0;
	for(size_t i = 0; i < num_blocks; i++) {
		size_t num_to_flush = per_block + (i < remainder);
		assert(num_to_flush < block_size);
		// write the header
		blind_write(buffer[num_so_far], header_start + i);
		num_to_flush--;
		num_so_far++;

		// write the rest into block
		size_t start = blocks_start + i * num_blocks;
		for(size_t j = 0; j < num_to_flush; j++) {
			blind_write(buffer[num_so_far], start + j);
			num_so_far++;
		}
	}
#if DEBUG
	assert(num_so_far == buffer.size());
#endif
}

// return index of the block that this elt would fall in
template <size_t log_size, size_t header_size, size_t block_size, typename key_type, typename... Ts>
size_t LeafDS<log_size, header_size, block_size, key_type, Ts...>::find_block(key_type key) {
	return find_block_with_hint(key, 0);
}

// return index of the block that this elt would fall in
template <size_t log_size, size_t header_size, size_t block_size, typename key_type, typename... Ts>
size_t LeafDS<log_size, header_size, block_size, key_type, Ts...>::find_block_with_hint(key_type key, size_t hint) {
	size_t i = header_start + hint;
#if DEBUG
	assert(key >= blind_read_key(i));
#endif
	// TODO: vectorize with leq vector instruction
	for( ; i < blocks_start; i++) {
		if(blind_read_key(i) == key)  {
			return i - header_start;
		} else if (blind_read_key(i) > key) {
			break;
		}
	}
#if DEBUG
	assert(i - header_start - 1 < num_blocks);
#endif
	return i - header_start - 1;
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

// given a range [start, end), look for elt e 
// if e is in the range, update it and return true
// otherwise return false
// also return index found or index stopped at
template <size_t log_size, size_t header_size, size_t block_size, typename key_type, typename... Ts>
std::pair<bool, size_t> LeafDS<log_size, header_size, block_size, key_type, Ts...>::update_in_range_if_exists(size_t start,
	size_t end, element_type e) {
	const key_type key = std::get<0>(e);
	size_t i = start;

	// size_t location = end;
	// bool found = false;

	for(; i < end; i++) { // TODO: vectorize this for loop using AVX-512
		// if found, update the val and return
		/*
		if (key == get_key_array(i) || get_key_array(i) == NULL_VAL) {
			if (!found) { location = i; found = true; }
		}
		*/

		
		if (key == get_key_array(i)) {
			update_val_at_index(e, i);
			return {true, i};
		} else if (get_key_array(i) == NULL_VAL) {
			return {false, i};
		}
		
	}
	/*
	if (location == end || get_key_array(location) == NULL_VAL) {
		return {false, location};
	} else {
		return {true, location};
	}
	*/
	return {false, end};
} 

// given a block index, return its range [start, end)
template <size_t log_size, size_t header_size, size_t block_size, typename key_type, typename... Ts>
inline std::pair<size_t, size_t> LeafDS<log_size, header_size, block_size, key_type, Ts...>::get_block_range(size_t block_idx) {
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
unsigned short LeafDS<log_size, header_size, block_size, key_type, Ts...>::count_block(size_t block_idx) {
	size_t block_start = blocks_start + block_idx * block_size;
	size_t block_end = block_start + block_size;

	// count number of nonzero elts in this block
	unsigned short count = 0;
	for(size_t i = block_start; i < block_end; i++) {
		count += (blind_read_key(i) != 0);
	}
	return count;
}

// flush the log to the blocks
// precondition: log has been sorted
template <size_t log_size, size_t header_size, size_t block_size, typename key_type, typename... Ts>
void LeafDS<log_size, header_size, block_size, key_type, Ts...>::flush_log_to_blocks(size_t max_to_flush) {
#if DEBUG_PRINT
	print();
	printf("at beginning of flush, num elts total = %lu\n", num_elts_total);
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
			if (!update_block.first) {
				// if not found, the second thing in the pair is the index at the end
#if DEBUG_PRINT
				printf("\tflushing elt %u to block %lu, header %u\n", blind_read_key(i), block_idx, blind_read_key(header_start + block_idx));
#endif
				log_to_flush[num_to_flush] = blind_read(i);
				num_to_flush++;
				num_to_flush_per_block[block_idx]++;
			} else {
#if DEBUG_PRINT
				printf("found duplicate in block %u of elt %u\n", block_idx, key_to_flush);
#endif
				num_elts_total--;
			}
		}
	}

	// count the number of elements in each block
	unsigned short count_per_block[num_blocks];

	// TODO: merge these loops and count the rest in global redistribute
	for (size_t i = 0; i < num_blocks; i++) {
		// TODO: vectorize count_block by counting empty slots
		count_per_block[i] = count_block(i);
	}

	// if any of them overflow, redistribute
	// TODO: can vectorize this part
	for (size_t i = 0; i < num_blocks; i++) {
		if (count_per_block[i] + num_to_flush_per_block[i] >= block_size) {
#if DEBUG_PRINT
			printf("at block %lu, count_per_block = %u, num to flush = %u\n", i, count_per_block[i], num_to_flush_per_block[i]);
			print();
#endif
			global_redistribute(log_to_flush, num_to_flush, count_per_block);
			return; // log gets taken care of in global redistribute
		}
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
}

template <size_t log_size, size_t header_size, size_t block_size, typename key_type, typename... Ts>
std::pair<bool, size_t> LeafDS<log_size, header_size, block_size, key_type, Ts...>::update_in_block_if_exists(element_type e) {
	const key_type key = std::get<0>(e);
	// if key is in the current range of this node, try to find it in the block
	auto block_idx = find_block(key);
	auto block_range = get_block_range(block_idx);
	// if found, update and return
	return update_in_range_if_exists(block_range.first, block_range.second, e);
}


// return true if the element was inserted, false otherwise
// may return true if the element was already there due to it being a pseudo-set
template <size_t log_size, size_t header_size, size_t block_size, typename key_type, typename... Ts>
bool LeafDS<log_size, header_size, block_size, key_type, Ts...>::insert(element_type e) {
	// first try to update the key if it is in the log
	// TODO: the simplest vectorized version looks in all of the cache lines in the log
	// another vectorized version fills one cache line at a time
	auto result = update_in_range_if_exists(0, num_inserts_in_log, e);
	if (result.first) { return false; }

#if DEBUG
	// there should always be space in the log
	assert(num_inserts_in_log < log_size);
#endif

	// if not found, add it to the log
	blind_write(e, num_inserts_in_log);
	num_elts_total++; // num elts (may be duplicates in log)
	num_inserts_in_log++;
	
	if (num_inserts_in_log == log_size) { // we filled the log
		sort_log();

		// if this is the first time we are flushing the log, just make the sorted log the header
		if (get_min_block_key() == 0) {
			for(size_t i = 0; i < log_size; i++) {
		    SOA_type::get_static(array.data(), N, i + log_size) =
	        SOA_type::get_static(array.data(), N, i);
			}
		} else { // otherwise, there are some elements in the block / header part
			// if inserting min, swap out the first header into the first block
			if (blind_read_key(0) < get_min_block_key()) {
		    size_t i = blocks_start;
				for(; i < blocks_start + block_size; i++) {
					if (blind_read_key(i) == 0) {
						break;
					}
				}
				assert(i < blocks_start + block_size);

				// src = header start, dest = i
				copy_src_to_dest(header_start, i);

				// make min elt the new first header
				copy_src_to_dest(0, header_start);
				num_elts_total++;
			}

			// flush the log
			// note: at this point, the min key is repeated in the
			// header and log (if there was a new min).  in the flush, it will just
			// get deduped
			flush_log_to_blocks(log_size);
		}

		// clear log
		num_inserts_in_log = 0;
		clear_range(0, log_size);
	}

#if DEBUG
	if (count_up_elts() != num_elts_total) {
		print();
		printf("counted %lu, num elts %lu\n", count_up_elts(), num_elts_total);
		assert(false);
	}
#endif
	return true;
}


// return true if the element was deleted, false otherwise
// may return false even if the elt is there due to it being a pseudo-set
// return N if not found, otherwise return the slot the key is at
template <size_t log_size, size_t header_size, size_t block_size, typename key_type, typename... Ts>
bool LeafDS<log_size, header_size, block_size, key_type, Ts...>::remove(key_type e) {
	// if less than current min, should not be in ds
	if (e < blind_read_key(header_start)) {
		return N;
	}
	assert( e >= blind_read_key(header_start) );
	size_t block_idx = find_block(e);
#if DEBUG_PRINT
	printf("\tin has, find block returned %lu\n", block_idx);
#endif
	// check the header
	if (e == blind_read_key(header_start + block_idx)) {
		return header_start + block_idx;
	}
	
	// check the block
	auto range = get_block_range(block_idx);
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
size_t LeafDS<log_size, header_size, block_size, key_type, Ts...>::get_index_in_blocks(key_type e) {
	// if less than current min, should not be in ds
	if (e < blind_read_key(header_start)) {
		return N;
	}
	assert( e >= blind_read_key(header_start) );
	size_t block_idx = find_block(e);
#if DEBUG_PRINT
	printf("\tin has, find block returned %lu\n", block_idx);
#endif
	// check the header
	if (e == blind_read_key(header_start + block_idx)) {
		return header_start + block_idx;
	}
	
	// check the block
	auto range = get_block_range(block_idx);
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
size_t LeafDS<log_size, header_size, block_size, key_type, Ts...>::get_index(key_type e) {
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
bool LeafDS<log_size, header_size, block_size, key_type, Ts...>::has(key_type e) {
	return (get_index(e) != N);
}

// print the range [start, end)
template <size_t log_size, size_t header_size, size_t block_size, typename key_type, typename... Ts>
void LeafDS<log_size, header_size, block_size, key_type, Ts...>::print_range(size_t start, size_t end) {
	SOA_type::map_range_with_index_static(
			(void *)array.data(), N,
			[](size_t index, key_type key, auto... args) {
				if (key != NULL_VAL) {
					if constexpr (binary) {
						std::cout << key << ", ";
					} else {
						std::cout << "(" << key << ", ";
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
void LeafDS<log_size, header_size, block_size, key_type, Ts...>::print() {
  printf("total num elts %lu\n", num_elts_total);
  SOA_type::print_type_details();

	if (num_elts_total == 0) {
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
bool LeafDS<log_size, header_size, block_size, key_type, Ts...>::map(F f) {
	// read-only version of map
	// for each elt in the log, search for it in the blocks
	// if it was found in the blocks, add the index of it into the duplicates list
	size_t dup_index[log_size];
	size_t num_duplicates = 0;

	for(size_t i = 0; i < num_inserts_in_log; i++) {
		key_type key = blind_read_key(i);
		size_t idx = get_index_in_blocks(key);
		if (idx < N) {
			dup_index[num_duplicates] = idx;
			num_duplicates++;
		}
	} 

  static_assert(std::is_invocable_v<decltype(&F::operator()), F &, uint32_t,
                                    NthType<Is>...>,
                "update function must match given types");

	// loop over all elts
  for (uint32_t i = 0; i < N; i++) {
    auto index = get_key_array(i);
    if (index != NULL_VAL) {
			// skip if duplicated
			for(uint32_t j = 0; j < num_duplicates; j++) {
				if(i == dup_index[j]) { continue; }
			}

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
uint64_t LeafDS<log_size, header_size, block_size, key_type, Ts...>::sum_keys_with_map() {
  uint64_t result = 0;
  map<true>([&](key_type key) { result += key; });
  return result;
}

template <size_t log_size, size_t header_size, size_t block_size, typename key_type, typename... Ts>
uint64_t LeafDS<log_size, header_size, block_size, key_type, Ts...>::sum_keys_direct() {
  uint64_t result = 0;

	// read-only version of map
	// for each elt in the log, search for it in the blocks
	// if it was found in the blocks, add the index of it into the duplicates list
	size_t dup_index[log_size];
	size_t num_duplicates = 0;

	for(size_t i = 0; i < num_inserts_in_log; i++) {
		key_type key = blind_read_key(i);
		size_t idx = get_index_in_blocks(key);
		if (idx < N) {
			dup_index[num_duplicates] = idx;
			num_duplicates++;
		}
	} 

	for (size_t i = 0; i < N; i++) {
		result += blind_read_key(i);
	}
	for(size_t i = 0; i < num_duplicates; i++) {
		result -= blind_read_key(dup_index[i]);
	}
  return result;
}
