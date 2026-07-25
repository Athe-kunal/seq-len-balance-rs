use std::cmp::Ordering;
use std::collections::BinaryHeap;

/// One sub-partition's running sum and the original indices assigned to it.
#[derive(Debug, Default, Clone)]
struct ItemSet {
    sum: i64,
    items: Vec<(usize, i64)>,
}

impl ItemSet {
    fn add(&mut self, idx: usize, val: i64) {
        self.items.push((idx, val));
        self.sum += val;
    }

    fn merge(&mut self, other: ItemSet) {
        for (idx, val) in other.items {
            self.add(idx, val);
        }
    }
}

impl PartialEq for ItemSet {
    fn eq(&self, other: &Self) -> bool {
        self.sum == other.sum && self.items.len() == other.items.len()
    }
}
impl Eq for ItemSet {}

impl PartialOrd for ItemSet {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for ItemSet {
    fn cmp(&self, other: &Self) -> Ordering {
        self.sum
            .cmp(&other.sum)
            .then_with(|| self.items.len().cmp(&other.items.len()))
    }
}

/// A partial k-way partition of some subset of the items, always kept with
/// `sets` sorted descending (largest sum first).
#[derive(Debug, Clone)]
struct KkState {
    sets: Vec<ItemSet>,
}

impl KkState {
    fn new_seed(items: Vec<(usize, i64)>, k: usize) -> Self {
        let mut sets = vec![ItemSet::default(); k];
        for (i, (idx, val)) in items.into_iter().enumerate() {
            sets[i].add(idx, val);
        }
        sets.sort_by(|a, b| b.cmp(a));
        KkState { sets }
    }

    fn spread(&self) -> i64 {
        let max = self.sets.first().map(|s| s.sum).unwrap_or(0);
        let min = self.sets.last().map(|s| s.sum).unwrap_or(0);
        max - min
    }

    /// Merge `other` into `self`, pairing the largest set in `self` with the
    /// smallest in `other`, next-largest with next-smallest, etc.
    fn merge(mut self, other: KkState) -> KkState {
        let k = self.sets.len();
        let mut other_sets = other.sets;
        for i in 0..k {
            let paired = other_sets.pop().expect("other has k sets");
            self.sets[i].merge(paired);
        }
        self.sets.sort_by(|a, b| b.cmp(a));
        self
    }

    fn into_partitions(self) -> Vec<Vec<usize>> {
        self.sets
            .into_iter()
            .map(|s| s.items.into_iter().map(|(idx, _)| idx).collect())
            .collect()
    }
}

impl PartialEq for KkState {
    fn eq(&self, other: &Self) -> bool {
        self.spread() == other.spread() && self.sets.len() == other.sets.len()
    }
}
impl Eq for KkState {}

impl PartialOrd for KkState {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for KkState {
    fn cmp(&self, other: &Self) -> Ordering {
        // BinaryHeap is a max-heap; the state with the largest spread must
        // be popped first, so it must compare as "greatest".
        self.spread()
            .cmp(&other.spread())
            .then_with(|| self.sets.len().cmp(&other.sets.len()))
    }
}

/// True Karmarkar-Karp largest-differencing partition into `k_partitions`
/// balanced groups. Returns the *original indices* of `seqlen_list` assigned
/// to each partition.
///
/// When `equal_size` is true, every partition ends up with exactly
/// `seqlen_list.len() / k_partitions` items (requires exact divisibility).
pub fn karmarkar_karp_indices(
    seqlen_list: &[i64],
    k_partitions: usize,
    equal_size: bool,
) -> Result<Vec<Vec<usize>>, String> {
    if k_partitions == 0 {
        return Err("k_partitions must be at least 1".to_string());
    }

    let mut sorted_items: Vec<(usize, i64)> = seqlen_list.iter().copied().enumerate().collect();
    sorted_items.sort_by_key(|&(_, v)| v);

    let mut heap: BinaryHeap<KkState> = BinaryHeap::new();

    if equal_size {
        if seqlen_list.len() % k_partitions != 0 {
            return Err(format!(
                "{} % {} != 0 (required when equal_size=True)",
                seqlen_list.len(),
                k_partitions
            ));
        }
        for chunk in sorted_items.chunks(k_partitions) {
            heap.push(KkState::new_seed(chunk.to_vec(), k_partitions));
        }
    } else {
        for item in sorted_items {
            heap.push(KkState::new_seed(vec![item], k_partitions));
        }
    }

    while heap.len() > 1 {
        let s0 = heap.pop().expect("heap has >1 element");
        let s1 = heap.pop().expect("heap has >1 element");
        heap.push(s0.merge(s1));
    }

    let final_state = heap.pop().ok_or("seqlen_list must not be empty")?;
    let partitions = final_state.into_partitions();

    if k_partitions > 1 {
        for (i, p) in partitions.iter().enumerate() {
            if p.is_empty() {
                return Err(format!("the {i}-th partition is empty"));
            }
        }
    }

    Ok(partitions)
}

/// `karmarkar_karp_indices` plus validation and bookkeeping: exactly
/// `k_partitions` non-empty partitions, every original index covered exactly
/// once, each partition's indices sorted ascending.
pub fn get_seqlen_balanced_partitions(
    seqlen_list: &[i64],
    k_partitions: usize,
    equal_size: bool,
) -> Result<Vec<Vec<usize>>, String> {
    if seqlen_list.len() < k_partitions {
        return Err(format!(
            "number of items:[{}] < k_partitions:[{}]",
            seqlen_list.len(),
            k_partitions
        ));
    }

    let partitions = karmarkar_karp_indices(seqlen_list, k_partitions, equal_size)?;

    if partitions.len() != k_partitions {
        return Err(format!(
            "{} != {}",
            partitions.len(),
            k_partitions
        ));
    }

    let mut seen = vec![false; seqlen_list.len()];
    let mut sorted_partitions = Vec::with_capacity(k_partitions);
    for (i, mut p) in partitions.into_iter().enumerate() {
        if p.is_empty() {
            return Err(format!("the {i}-th partition is empty"));
        }
        for &idx in &p {
            seen[idx] = true;
        }
        p.sort_unstable();
        sorted_partitions.push(p);
    }

    if !seen.iter().all(|&s| s) {
        return Err("not every index in seqlen_list was assigned to a partition".to_string());
    }

    Ok(sorted_partitions)
}
