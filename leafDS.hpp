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

public:
	size_t get_num_elts() const { return num_elts_total; }
	bool is_full() { return num_elts_total >= max_density; }


	// main top-level functions
	// given a key, return the index of the largest elt at least e
	[[nodiscard]] uint32_t search(key_type e) const;

	// insert e, return true if it was not there
  bool insert(element_type e);

	// remove e, return true if it was present
  bool remove(key_type e);

	// whether elt e was in the DS
  [[nodiscard]] bool has(key_type e) const;

  auto blind_read_key(uint32_t index) const {
    return std::get<0>(SOA_type::get_static(array.data(), N, index));
  }

	// min block header is the first elt in the header part
	auto get_min_block_key() const {
		return blind_read(log_size);
  }

  void blind_write(element_type e, uint32_t index) {
    num_elts_total += get_key_array(index) == NULL_VAL;
    SOA_type::get_static(array.data(), N, index) = e;
  }

  auto blind_read(uint32_t index) {
    return SOA_type::get_static(array.data(), N, index);
  }


// iterator after here
	class iterator;

	class iterator {
		//! The key type of the btree. Returned by key().
		// typedef typename LeafDS::key_type key_type;

		//! The value type of the btree. Returned by operator*().
		typedef typename LeafDS::value_type value_type;

		//! Reference to the value_type. STL required.
		typedef value_type& reference;

		//! Pointer to the value_type. STL required.
		typedef value_type* pointer;

		typedef iterator self;

private:
		uint32_t curr_slot;

public:
		// default
		iterator() : curr_slot(0) { }

		// with slot
		iterator(uint32_t s) : curr_slot(s) { }

		reference operator * () const {
			return blind_read_key(curr_slot);
		}

		iterator& operator ++ () {
			curr_slot++;

			// move forward to the next nonempty slot
			while(curr_slot < N) { 
				if (blind_read_key(curr_slot) == 0) {
					++curr_slot;
				} else {
					break;
				}
			}

			return *this;
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
bool LeafDS<N, key_type, Ts...>::insert(element_type e) {
	const key_type key = std::get<0>(e);

	// look for the element in the log
	for(size_t i = 0; i < num_elts_in_log; i++) {
		// if found update the val in the log and return
		if (e == get_key_array(i)) {
			update_val_at_index(e, i);
			return false;
		}
	}

	// if there is space in the log, add to it
	if (num_elts_in_log < log_size) {
		place_elt_at_index(e, num_elts_in_log);
		num_elts_in_log++;
		return true;
	} else { // otherwise, there is no space in the log
		// if this is the first time we are flushing the log, just make the sorted log the header
		if (get_min_block_key() == 0) {
			


		} else {
			
			// if key is in the current range of this node, try to find it in the block
			
			if(key >= get_min_block_key()) {
				// uint32_t block_idx = find_block(key);
			}
		}


	}
}
