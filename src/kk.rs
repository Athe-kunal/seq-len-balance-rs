use crate::bin::BinPacking;
use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::ops::Add;

// [8,7,6,5,4]
// [0 | 0 | 8] -> sort ascending (first element)
// [7 | 0 | 0] -> sort descending element
// [7 | 0 | 8] -> sort -> [0 | 7 | 8]

// [0 | 7 | 8] -> sorted ascending
// [6 | 0 | 0] -> sorted descending
// [6 | 7 | 8] -> sort ascending -> [6 | 7 | 8]

// [6 | 7 | 8]
// [5 | 0 | 0]
// [(6,5) | 7 | 8] -> sort ascending -> [7 | 8 | 11]

// [7 | 8 | 11]
// [5 | 0 | 0]
// [(7,5) | 8 | (6,5)] -> [8 | 11 | 12]

// [8 | 11 | 12]
// [4 | 0 | 0]
// [(8,4) | 11 | 12] -> [(8,4) | (6,5) | (7,5) ]

pub fn kk_partition<T>(mut items: Vec<T>, k: usize) -> BinPacking<T>
where
    T: Ord + Add<Output = T> + Default + Clone,
{
    assert!(k > 0, "k must be at least 1");
    // placing large items first improves balancing.
    items.sort_unstable_by(|a, b| b.cmp(a));

    let mut bins: Vec<Vec<T>> = vec![vec![]; k];
    let mut sums: Vec<T> = vec![T::default(); k];
    // BinaryHeap<Reverse<...>> gives min-heap behaviour.
    let mut min_heap: BinaryHeap<Reverse<(T, usize)>> =
        (0..k).map(|i| Reverse((T::default(), i))).collect();

    for item in items {
        let Reverse((min_sum, idx)) = min_heap.pop().unwrap();
        let new_sum = min_sum + item.clone();
        bins[idx].push(item);
        sums[idx] = new_sum.clone();
        min_heap.push(Reverse((new_sum, idx)));
    }

    let remaining = vec![T::default(); k];
    BinPacking {
        bins,
        sums,
        remaining,
        capacity: None,
    }
}
