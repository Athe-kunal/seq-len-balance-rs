use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::fmt;
use std::ops::Add;

// Step	Item	Smallest bucket	Result buckets (sorted by sum)
// 1	8	0	[0\|0\|8]
// 2	7	0	[7\|0\|8] → sort → [0\|7\|8]
// 3	6	0	[6\|7\|8]
// 4	5	6	[(6,5)\|7\|8] → sort → [7\|8\|11]
// 5	4	7	[(4,7)\|8\|11] → final: [11\|8\|11]

/// Result of a balanced k-way partition.
///
/// For `f32`/`f64`, wrap values with `ordered_float::OrderedFloat<f64>` which
/// implements `Ord` and satisfies the bounds below.
#[derive(Debug)]
pub struct Partition<T> {
    pub groups: Vec<Vec<T>>,
    pub sums: Vec<T>,
}

impl<T> Partition<T>
where
    T: Ord + Clone + std::ops::Sub<Output = T>,
{
    /// Difference between the largest and smallest group sum.
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

/// Greedy balanced k-way partition (Karmarkar-Karp / LPT style).
///
/// Sorts items descending, then assigns each item to the group with the
/// smallest current sum using a min-heap — this minimises the spread between
/// the heaviest and lightest groups.
///
/// # Type bounds
/// - `Ord`     — sort items and compare bucket sums in the min-heap
/// - `Add`     — accumulate group sums
/// - `Default` — zero value for empty groups (`0` for integers)
/// - `Clone`   — items move into groups; running sums are cloned
pub fn kk_partition<T>(mut items: Vec<T>, k: usize) -> Partition<T>
where
    T: Ord + Add<Output = T> + Default + Clone,
{
    assert!(k > 0, "k must be at least 1");

    // Largest items first.
    items.sort_unstable_by(|a, b| b.cmp(a));

    let mut groups: Vec<Vec<T>> = vec![vec![]; k];
    let mut sums: Vec<T> = vec![T::default(); k];

    // Min-heap of (current_sum, group_index).
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
