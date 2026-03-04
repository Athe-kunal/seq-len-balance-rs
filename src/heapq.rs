#[derive(Debug, Default)]
pub struct MinHeap<T: Ord> {
    data: Vec<T>,
}

impl<T: Ord> MinHeap<T> {
    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
    // used after inserting a new element
    pub fn sift_up(&mut self, mut i: usize) {
        while i > 0 {
            let parent = (i - 1) / 2;

            if self.data[parent] <= self.data[i] {
                break;
            }
            self.data.swap(parent, i);
            i = parent;
        }
    }
    // after deleting an element
    pub fn sift_down(&mut self, mut i: usize) {
        let n = self.data.len();

        loop {
            // parent(i) = (i - 1) / 2
            // left(i)   = 2*i + 1
            // right(i)  = 2*i + 2
            let left = 2 * i + 1;
            let right = 2 * i + 2;

            if left >= n {
                break;
            }
            let mut smallest = left;
            if right < n && self.data[right] < self.data[left] {
                smallest = right;
            }

            if self.data[i] <= self.data[smallest] {
                break;
            }
            self.data.swap(i, smallest);
            i = smallest;
        }
    }

    pub fn push(&mut self, val: T) {
        self.data.push(val);
        let i = self.data.len() - 1;
        self.sift_up(i);
    }

    pub fn pop(&mut self) -> Option<T> {
        if self.data.is_empty() {
            return None;
        }
        let last = self.data.len() - 1;
        self.data.swap(0, last);
        let top = self.data.pop();
        if !self.data.is_empty() {
            self.sift_down(0);
        }
        top
    }

    pub fn peek(&self) -> Option<&T> {
        self.data.first()
    }

    pub fn from_vec(data: Vec<T>) -> Self {
        let mut h = Self { data };
        let n = h.data.len();
        for i in (0..n / 2).rev() {
            h.sift_down(i);
        }
        h
    }

    pub fn with_capacity(cap: usize) -> Self {
        Self {
            data: Vec::with_capacity(cap),
        }
    }
}

impl<T: Ord + std::fmt::Display> MinHeap<T> {
    pub fn print_tree(&self) {
        if self.data.is_empty() {
            println!("(empty)");
            return;
        }

        let n = self.data.len();
        let mut level_start = 0;
        let mut level_size = 1;

        while level_start < n {
            let level_end = (level_start + level_size).min(n);
            for i in level_start..level_end {
                if i > level_start {
                    print!("  ");
                }
                print!("{}", self.data[i]);
            }
            println!();
            level_start += level_size;
            level_size *= 2;
        }
    }
}
