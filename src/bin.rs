use std::fmt;
use std::ops::{Add, Sub};

/// Result of a bin packing algorithm.
#[derive(Debug)]
pub struct BinPacking<T> {
    pub bins: Vec<Vec<T>>,
    pub remaining: Vec<T>,
    pub capacity: T,
}

impl<T: Clone + Sub<Output = T>> BinPacking<T> {
    /// Number of bins used.
    pub fn num_bins(&self) -> usize {
        self.bins.len()
    }

    /// Total wasted space across all bins.
    pub fn waste(&self) -> T
    where
        T: Default + Add<Output = T>,
    {
        self.remaining
            .iter()
            .cloned()
            .fold(T::default(), |acc, r| acc + r)
    }
}

impl<T: fmt::Display + Clone + Sub<Output = T>> fmt::Display for BinPacking<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (i, (bin, rem)) in self.bins.iter().zip(self.remaining.iter()).enumerate() {
            write!(f, "Bin {i} (remaining={rem}): [")?;
            for (j, item) in bin.iter().enumerate() {
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

/// First Fit Decreasing bin packing.
///
/// Sorts items descending, then for each item tries bins in order (0, 1, 2, ...)
/// and places it in the **first** bin with enough remaining capacity.
/// Opens a new bin only when no existing bin fits.
///
/// # Panics
/// Panics if any item exceeds `capacity`.
pub fn first_fit_decreasing<T>(mut items: Vec<T>, capacity: T) -> BinPacking<T>
where
    T: Ord + Add<Output = T> + Sub<Output = T> + Default + Clone,
{
    items.sort_unstable_by(|a, b| b.cmp(a));

    let mut bins: Vec<Vec<T>> = Vec::new();
    let mut remaining: Vec<T> = Vec::new();

    for item in items {
        assert!(
            item <= capacity,
            "item exceeds bin capacity"
        );

        // First bin with enough room.
        let target = bins
            .iter()
            .enumerate()
            .find(|(i, _)| remaining[*i] >= item)
            .map(|(i, _)| i);

        match target {
            Some(i) => {
                remaining[i] = remaining[i].clone() - item.clone();
                bins[i].push(item);
            }
            None => {
                remaining.push(capacity.clone() - item.clone());
                bins.push(vec![item]);
            }
        }
    }

    BinPacking { bins, remaining, capacity }
}

/// Best Fit Decreasing bin packing.
///
/// Sorts items descending, then for each item finds the bin with the
/// **minimum remaining capacity** that still fits the item (tightest fit).
/// This leaves larger gaps available for larger future items.
/// Opens a new bin only when no existing bin fits.
///
/// # Panics
/// Panics if any item exceeds `capacity`.
pub fn best_fit_decreasing<T>(mut items: Vec<T>, capacity: T) -> BinPacking<T>
where
    T: Ord + Add<Output = T> + Sub<Output = T> + Default + Clone,
{
    items.sort_unstable_by(|a, b| b.cmp(a));

    let mut bins: Vec<Vec<T>> = Vec::new();
    let mut remaining: Vec<T> = Vec::new();

    for item in items {
        assert!(item <= capacity, "item exceeds bin capacity");

        // Bin with the smallest remaining space that still fits (tightest fit).
        let best_idx = remaining
            .iter()
            .enumerate()
            .filter(|(_, r)| *r >= &item)
            .min_by(|(_, a), (_, b)| a.cmp(b))
            .map(|(i, _)| i);

        match best_idx {
            Some(i) => {
                remaining[i] = remaining[i].clone() - item.clone();
                bins[i].push(item);
            }
            None => {
                remaining.push(capacity.clone() - item.clone());
                bins.push(vec![item]);
            }
        }
    }

    BinPacking { bins, remaining, capacity }
}
