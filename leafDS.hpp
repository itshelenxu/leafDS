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

template <size_t N, typename key_type, typename... Ts>
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

  static_assert(N != 0, "N cannot be 0");

private:
	static constexpr key_type NULL_VAL = {};

	// number of slots in the log, header, and each block
	static constexpr size_t log_size = 32;
	static constexpr size_t header_size = log_size;
	static constexpr size_t num_blocks = header_size;

	// block size using the remaining space
	static_assert((N - header_size - log_size) % num_blocks == 0);
	static constexpr size_t block_size = (N - header_size - log_size)/num_blocks;

	// start of each section
	static constexpr size_t header_start = log_size;
	static constexpr size_t blocks_start = log_size + header_size;

	// counters
	size_t num_elts_in_log = 0;
	size_t num_elts_total = 0;

	// max elts allowed in data structure before split
	static constexpr size_t max_density = (int)( 9.0 / 10.0 * N );

	std::array<uint8_t, SOA_type::get_size_static(N)> array = {0};

  inline key_type get_key_array(uint32_t index) const {
    return std::get<0>(
        SOA_type::template get_static<0>(array.data(), N, index));
  }

	// private helpers
	void update_val_at_index(element_type e, size_t index);
	void place_elt_at_index(element_type e, size_t index);
	void clear_range(size_t start, size_t end);
	size_t find_block(key_type key);
	size_t find_block_with_hint(key_type key, size_t hint);
	void global_redistribute(element_type* log_to_flush, size_t num_to_flush, unsigned short* num_to_flush_per_block, unsigned short* count_per_block);
	void copy_src_to_dest(size_t src, size_t dest);
	void flush_log_to_blocks();
	void sort_log();
	void sort_range(size_t start_idx, size_t end_idx);
	std::pair<size_t, size_t> get_block_range(size_t block_idx);
	std::pair<bool, size_t> update_in_range_if_exists(size_t start, size_t end, element_type e);
	std::pair<bool, size_t> update_in_block_if_exists(element_type e);
	unsigned short count_block(size_t block_idx);
	void print_range(size_t start, size_t end);

public:
	size_t get_num_elts() const { return num_elts_total; }
	bool is_full() { return num_elts_total >= max_density; }
	void print();

	// main top-level functions
	// given a key, return the index of the largest elt at least e
	[[nodiscard]] uint32_t search(key_type e) const;

	// insert e, return true if it was not there
  bool insert(element_type e);

	// remove e, return true if it was present
  bool remove(key_type e);

	// whether elt e was in the DS
  [[nodiscard]] bool has(key_type e);

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


	// iterator after here
	// TODO: rename it to be clear this only supports packed regions
	class packed_iterator;

	class packed_iterator {
		//! The key type of the btree. Returned by key().
		// typedef typename LeafDS::key_type key_type;

		typedef LeafDS<N, key_type, Ts...> container_type;
		// typedef typename container_type::iterator iterator_type;

		using iterator_category = std::random_access_iterator_tag;

		//! The value type of the btree. Returned by operator*().
		typedef typename LeafDS::value_type value_type;

		//! Reference to the value_type. STL required.
		typedef value_type& reference;

		//! Pointer to the value_type. STL required.
		typedef value_type* pointer;

		typedef packed_iterator self;

protected:
		uint32_t curr_slot;

public:
		// https://en.cppreference.com/w/cpp/named_req/RandomAccessIterator
		// default
		// iterator_type() : curr_slot(0) { }

		// with slot
		packed_iterator(uint32_t s) : curr_slot(s) { }

		reference operator * () const {
		  return SOA_type::get_static(array.data(), N, curr_slot);

			// return blind_read_key(curr_slot);
		}

		packed_iterator& operator ++ () {
			curr_slot++;

			/*
			// move forward to the next nonempty slot
			while(curr_slot < N) { 
				if (blind_read_key(curr_slot) == 0) {
					++curr_slot;
				} else {
					break;
				}
			}
			*/

			return *this;
		}

		packed_iterator& operator + (size_t s) {
			curr_slot += s;
			return *this;
		}

		packed_iterator& operator - (size_t s) {
			curr_slot -= s;
			return *this;
		}

		bool operator > () {
			return 
		}
	};

};


// precondition - keys at index already match
template <size_t N, typename key_type, typename... Ts>
void LeafDS<N, key_type, Ts...>::update_val_at_index(element_type e, size_t index) {
	// update its value if needed
	if constexpr (!binary) {
		if (leftshift_tuple(SOA_type::get_static(array.data(), N, index)) !=
				leftshift_tuple(e)) {
			SOA_type::get_static(array.data(), N, index) = e;
		}
	}
}

// precondition - this slot is empty
template <size_t N, typename key_type, typename... Ts>
void LeafDS<N, key_type, Ts...>::place_elt_at_index(element_type e, size_t index) {
	SOA_type::get_static(array.data(), N, index) = e;
	num_elts_total++;
}

template <size_t N, typename key_type, typename... Ts>
void LeafDS<N, key_type, Ts...>::clear_range(size_t start, size_t end) {
  SOA_type::map_range_static(
      array.data(), N,
      [](auto &...args) { std::forward_as_tuple(args...) = element_type(); },
      start, end);
}

// one of the blocks
// input: deduped log to flush, number of elements in the log, count of elements to flush to each block, count of elements per block
// merge all elements from blocks and log in sorted order in the intermediate buffer
// split them evenly amongst the blocks
template <size_t N, typename key_type, typename... Ts>
void LeafDS<N, key_type, Ts...>::global_redistribute(element_type* log_to_flush, size_t num_to_flush, unsigned short* num_to_flush_per_block, unsigned short* count_per_block) {
	// sort each block
	size_t end_blocks = 0;
	for (int i = 0; i < num_blocks; i++) {
		auto block_range = get_block_range(i);
		sort_range(block_range.first, block_range.first + count_per_block[i]);
		if (i == num_blocks - 1) { end_blocks = block_range.first + count_per_block[i]; }
	}

	// do a merge from sorted log and sorted blocks
	std::vector<element_type> buffer;
	size_t log_ptr = 0; 
	size_t blocks_ptr = header_start;
	size_t cur_block = 0;
	size_t start_of_cur_block = 0;
	size_t log_end = num_to_flush;

	while(log_ptr < log_end && cur_block < num_blocks) {
		const key_type log_key = std::get<0>(log_to_flush[log_ptr]);
		const key_type block_key = blind_read_key(blocks_ptr);
		assert(log_key != block_key);
		if (log_key < block_key) {
			buffer.push_back(log_to_flush[log_ptr]);
			log_ptr++;
		} else {
			assert(blind_read_key(blocks_ptr) != NULL_VAL);
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
			} else if (blocks_ptr == start_of_cur_block + num_to_flush_per_block[cur_block] - 1) {
				// if we have merged in this entire block, go back to the header
				cur_block++;
				blocks_ptr = header_start + cur_block;
			} else {
				// if we are still in this block, keep going
				blocks_ptr++;
			}
		}
	}
	// cleanup if necessary
	while(log_ptr < log_end) {
		buffer.push_back(log_to_flush[log_ptr]);
		log_ptr++;
	}

	while(cur_block < num_blocks) {
		assert(blind_read_key(blocks_ptr) != NULL_VAL);
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
		} else if (blocks_ptr == start_of_cur_block + num_to_flush_per_block[cur_block] - 1) {
			// if we have merged in this entire block, go back to the header
			cur_block++;
			blocks_ptr = header_start + cur_block;
		} else {
			// if we are still in this block, keep going
			blocks_ptr++;
		}
	}

	assert(buffer.size() < num_blocks * block_size);
	// at this point the buffer should be sorted
	for (int i = 1; i < buffer.size(); i++) {
		assert(std::get<0>(buffer[i]) < std::get<0>(buffer[i-1]));
	}

	clear_range(header_start, N); // clear the header and blocks

	// split up the buffer into blocks
	size_t per_block = buffer.size() / num_blocks;
	size_t remainder = buffer.size() % num_blocks;
	size_t num_so_far = 0;
	for(int i = 0; i < num_blocks; i++) {
		size_t num_to_flush = per_block + (i < remainder);
		assert(num_to_flush < block_size);
		// write the header
		blind_write(buffer[num_so_far], header_start + i);
		num_to_flush--;
		num_so_far++;

		// write the rest into block
		size_t start = blocks_start + i * num_blocks;
		for(int j = 0; j < num_to_flush; j++) {
			blind_write(buffer[num_so_far], start + j);
			num_so_far++;
		}
	}
	assert(num_so_far == buffer.size());
}

// return index of the block that this elt would fall in
template <size_t N, typename key_type, typename... Ts>
size_t LeafDS<N, key_type, Ts...>::find_block(key_type key) {
	return find_block_with_hint(key, 0);
}

// return index of the block that this elt would fall in
template <size_t N, typename key_type, typename... Ts>
size_t LeafDS<N, key_type, Ts...>::find_block_with_hint(key_type key, size_t hint) {
	size_t i = header_start + hint;
	for( ; i < blocks_start; i++) {
		if(blind_read_key(i) == key)  {
			return i;
		} else if (blind_read_key(i) > key) {
			break;
		}
	}
	return i - header_start - 1;
}

// given a src, dest indices 
// move elt at src into dest
template <size_t N, typename key_type, typename... Ts>
void LeafDS<N, key_type, Ts...>::copy_src_to_dest(size_t src, size_t dest) {
	SOA_type::get_static(array.data(), N, dest) =
		SOA_type::get_static(array.data(), N, src);
}


// precondition: range must be packed
template <size_t N, typename key_type, typename... Ts>
void LeafDS<N, key_type, Ts...>::sort_range(size_t start_idx, size_t end_idx) {
	auto start = packed_iterator(start_idx);
	auto end = packed_iterator(end_idx);
	std::sort(start, end);
}

// sort the log
template <size_t N, typename key_type, typename... Ts>
void LeafDS<N, key_type, Ts...>::sort_log() {
	sort_range(0, num_elts_in_log);
}

// given a range [start, end), look for elt e 
// if e is in the range, update it and return true
// otherwise return false
// also return index found or index stopped at
template <size_t N, typename key_type, typename... Ts>
std::pair<bool, size_t> LeafDS<N, key_type, Ts...>::update_in_range_if_exists(size_t start,
	size_t end, element_type e) {
	const key_type key = std::get<0>(e);
	size_t i = start;
	for(; i < end; i++) {
		// if found, update the val and return
		if (key == get_key_array(i)) {
			update_val_at_index(e, i);
			return {true, i};
		} else if (get_key_array(i) == NULL_VAL) {
			return {false, i};
		}
	}
	return {false, end};
} 

// given a block index, return its range [start, end)
template <size_t N, typename key_type, typename... Ts>
std::pair<size_t, size_t> LeafDS<N, key_type, Ts...>::get_block_range(size_t block_idx) {
	size_t block_start = blocks_start + block_idx * block_size;
	size_t block_end = block_start + block_size;
	assert(block_start < N);
	assert(block_end <= N);
	return {block_start, block_end};
}

// count up the number of elements in this b lock
template <size_t N, typename key_type, typename... Ts>
unsigned short LeafDS<N, key_type, Ts...>::count_block(size_t block_idx) {
	size_t block_start = blocks_start + block_idx * block_size;
	size_t block_end = block_start + block_size;

	// count number of nonzero elts in this block
	unsigned short count = 0;
	for(int i = block_start; i < block_end; i++) {
		count += (blind_read_key(i) != 0);
	}
	return count;
}

// flush the log to the blocks
template <size_t N, typename key_type, typename... Ts>
void LeafDS<N, key_type, Ts...>::flush_log_to_blocks() {
	sort_log();

	// dedup the log wrt the blocks
	unsigned short num_dest_per_block[num_blocks];

	element_type log_to_flush[log_size];
	unsigned short num_to_flush = 0;
	unsigned short num_to_flush_per_block[num_blocks];
	size_t hint = 0;
	size_t block_index;

	// dedup the log
	for(size_t i = 0; i < log_size; i++) {
		key_type key_to_flush = blind_read_key(i);
		assert(key_to_flush >= blind_read_key(log_size));

		size_t block_idx = find_block_with_hint(key_to_flush, hint);

		// if it is in the header, update the header
		if (blind_read_key(header_start + block_idx) == key_to_flush) {
				copy_src_to_dest(i, header_start + block_idx);
		} else {
			// otherwise, look for it in the block
			auto update_block = update_in_block_if_exists(blind_read(i));
			size_t block_idx = update_block.second / block_size; // floor div
			// update hint 
			if (hint < block_idx) { hint = block_idx; }

			// if it was in the block, do nothing bc we have already updated it
			// if not found, add to the deduped log_to_flush
			if (!update_block.first) {
				// if not found, the second thing in the pair is the index at the end
				log_to_flush[num_to_flush] = blind_read(i);
				num_to_flush++;
				num_to_flush_per_block[block_idx]++;
			}
		}
	}

	// count the number of elements in each block
	unsigned short count_per_block[num_blocks];
	for (int i = 0; i < num_blocks; i++) {
		count_per_block[i] = count_block(i);
	}

	// if any of them overflow, redistribute
	for (int i = 0; i < num_blocks; i++) {
		if (count_per_block[i] + num_to_flush_per_block[i] >= block_size) {
			global_redistribute(log_to_flush, num_to_flush, num_to_flush_per_block, count_per_block);
			return; // log gets taken care of in global redistribute
		}
	}

	// otherwise, flush the log to the blocks
	size_t idx_in_log = 0;
	for(int i = 0; i < num_blocks; i++) {
		// pointer to start of block
		size_t write_start = blocks_start + i * block_size + count_per_block[i];
		for(int j = 0; j < num_to_flush_per_block[i]; j++) {
			blind_write(log_to_flush[idx_in_log + j], write_start + j);
		}
		idx_in_log += num_to_flush_per_block[i];
	}
	assert(idx_in_log == num_to_flush);
}

template <size_t N, typename key_type, typename... Ts>
std::pair<bool, size_t> LeafDS<N, key_type, Ts...>::update_in_block_if_exists(element_type e) {
	const key_type key = std::get<0>(e);
	// if key is in the current range of this node, try to find it in the block
	auto block_idx = find_block(key);
	auto block_range = get_block_range(block_idx);
	// if found, update and return
	return update_in_range_if_exists(block_range.first, block_range.second, e);
}

template <size_t N, typename key_type, typename... Ts>
bool LeafDS<N, key_type, Ts...>::insert(element_type e) {
	const key_type key = std::get<0>(e);

	// first try to update the key if it is in the log
	auto result = update_in_range_if_exists(0, num_elts_in_log, e);
	if (result.first) { return false; }

	// then try to update the key in the blocks if it would be there / if there
	// are elements in the blocks
	if (get_min_block_key() != NULL_VAL && key >= get_min_block_key()) {
		// check if it matches any of the header
		auto result = update_in_range_if_exists(log_size, log_size + header_size, e);
		if (result.first) { return false; }

		// then check if it is in the blocks
		result = update_in_block_if_exists(e);
		if (result.first) { return false; }
	}

	// there should always be space in the log
	assert(num_elts_in_log < log_size);
	blind_write(e, num_elts_in_log);
	num_elts_total++;
	num_elts_in_log++;
	
	if (num_elts_in_log == log_size) { // we filled the log
		// if this is the first time we are flushing the log, just make the sorted log the header
		if (get_min_block_key() == 0) {
			// move log into header
			for(size_t i = 0; i < log_size; i++) {
		    SOA_type::get_static(array.data(), N, i + log_size) =
	        SOA_type::get_static(array.data(), N, i);
			}
		} else { // otherwise, there are some elements in the block / header part
			// sort the log
			sort_log();

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
			}

			// flush the log
			// note: at this point, the min key is repeated in the
			// header and log (if there was a new min).  in the flush, it will just
			// get deduped
			flush_log_to_blocks();
		}
		// clear log
		clear_range(0, log_size);
	}
	return true;
}


// TODO: handle if you look for something smaller than min
template <size_t N, typename key_type, typename... Ts>
bool LeafDS<N, key_type, Ts...>::has(key_type e) {
	// check the log
	for(size_t i = 0; i < num_elts_in_log; i++) {
		if(e == blind_read_key(i)) {
			return true;
		}
	}

	size_t block_idx = find_block(e);
	// check the header
	if (e == blind_read_key(header_start + block_idx)) {
		return true;
	}
	
	// check the block
	auto range = get_block_range(block_idx);
	for(int i = range.first; i < range.second; i++) {
		if (blind_read_key(i) == NULL_VAL) {
			return false;
		}
		if (blind_read_key(i) == e) {
			return true;
		}
	}

	return false;
}

template <size_t N, typename key_type, typename... Ts>
void LeafDS<N, key_type, Ts...>::print_range(size_t start, size_t end) {
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

template <size_t N, typename key_type, typename... Ts>
void LeafDS<N, key_type, Ts...>::print() {
  printf("total num elts %u\n", num_elts_total);
  SOA_type::print_type_details();

	if (num_elts_total == 0) {
    printf("the ds is empty\n");
  }

	printf("log: \n");
	print_range(0, log_size);
	
	printf("headers:\n");
	print_range(header_start, blocks_start);

	for (uint32_t i = blocks_start; i < N; i += block_size) {
		printf("block %u\n", i);
		print_range(i, i + block_size);
	}
	printf("\n");
}
