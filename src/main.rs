use seq_len_balance::bin::{best_fit_decreasing, first_fit_decreasing};
use seq_len_balance::kk::kk_partition;

fn main() {
    // let mut heap = MaxHeap::from_vec(vec![3, 1, 4, 1, 5, 9, 2, 6]);

    // println!("peek: {:?}", heap.peek());
    // heap.print_tree();

    // heap.push(10);
    // println!("after push 10:");
    // heap.print_tree();

    // print!("sorted desc: ");
    // while let Some(v) = heap.pop() {
    //     print!("{v} ");
    // }
    // println!();

    println!("\n--- kk_partition (integers) ---");
    let items: Vec<i64> = vec![8, 7, 6, 5, 4];
    let p = kk_partition(items, 3);
    print!("{p}");
    println!("imbalance: {}", p.imbalance());

    println!("\n--- kk_partition (usize, e.g. sequence lengths) ---");
    let lengths: Vec<usize> = vec![10, 9, 3, 7, 5, 2, 8];
    let p2 = kk_partition(lengths, 3);
    print!("{p2}");
    println!("imbalance: {}", p2.imbalance());

    println!("\n--- first_fit_decreasing (capacity=10) ---");
    let ffd_items: Vec<u32> = vec![6, 3, 4, 5, 2, 7, 1];
    let ffd = first_fit_decreasing(ffd_items, Some(10));
    print!("{ffd}");
    println!(
        "bins used: {}  total waste: {}",
        ffd.num_bins(),
        ffd.waste()
    );

    println!("\n--- best_fit_decreasing (capacity=10) ---");
    let bfd_items: Vec<u32> = vec![6, 3, 4, 5, 2, 7, 1];
    let bfd = best_fit_decreasing(bfd_items, Some(10));
    print!("{bfd}");
    println!(
        "bins used: {}  total waste: {}",
        bfd.num_bins(),
        bfd.waste()
    );
}
