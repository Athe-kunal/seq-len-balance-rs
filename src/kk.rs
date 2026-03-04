use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::fmt;
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

#[derive(Debug)]
pub struct Partition<T> {
    pub groups: Vec<Vec<T>>,
    pub sums: Vec<T>,
}

impl<T> Partition<T>
where
    T: Ord + Clone + std::ops::Sub<Output = T>,
{
    pub fn imbalance(&self) -> T {
        let max = self.sums.iter().max().unwrap().clone();
        let min = self.sums.iter().min().unwrap().clone();
        max - min
    }
}

impl<T: fmt::Display> fmt::Display for Partition<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (i, (group, sum)) in self.groups.iter().zip(self.sums.iter()).enumerate() {
            write!(f, "Group {i} (sum={sum}): [")?;
            for (j, item) in group.iter().enumerate() {
                if j > 0 {
                    write!(f, ", ")?;
                }
                write!(f, "{item}")?;
            }
            writeln!(f, "]")?;
        }
        Ok(())
    }
}

pub fn kk_partition<T>(mut items: Vec<T>, k: usize) -> Partition<T>
where
    T: Ord + Add<Output = T> + Default + Clone,
{
    assert!(k > 0, "k must be at least 1");
    // placing large items first improves balancing.
    items.sort_unstable_by(|a, b| b.cmp(a));

    let mut groups: Vec<Vec<T>> = vec![vec![]; k];
    let mut sums: Vec<T> = vec![T::default(); k];
    // BinaryHeap<Reverse... is for min heap
    let mut min_heap: BinaryHeap<Reverse<(T, usize)>> =
        (0..k).map(|i| Reverse((T::default(), i))).collect();

    for item in items {
        let Reverse((min_sum, idx)) = min_heap.pop().unwrap();
        let new_sum = min_sum + item.clone();
        groups[idx].push(item);
        sums[idx] = new_sum.clone();
        min_heap.push(Reverse((new_sum, idx)));
    }

    Partition { groups, sums }
}
