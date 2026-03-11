pub mod bin;
pub mod heapq;
pub mod kk;

use bin::{best_fit_decreasing, first_fit_decreasing};
use kk::kk_partition;

use ordered_float::OrderedFloat;
use pyo3::prelude::*;
use pyo3::types::PySequence;

fn seq_to_ordered(seq: &Bound<'_, PySequence>) -> PyResult<Vec<OrderedFloat<f64>>> {
    let len = seq.len()?;
    (0..len)
        .map(|i| {
            let val: f64 = seq.get_item(i)?.extract()?;
            Ok(OrderedFloat(val))
        })
        .collect()
}

fn unwrap_bins(bins: Vec<Vec<OrderedFloat<f64>>>) -> Vec<Vec<f64>> {
    bins.into_iter()
        .map(|b| b.into_iter().map(OrderedFloat::into_inner).collect())
        .collect()
}

#[pyfunction]
#[pyo3(name = "karmarkar_karp_partition")]
/// Partition numeric sequence lengths into `k` balanced bins using Karmarkar-Karp.
fn py_karmarkar_karp_partition(
    items: &Bound<'_, PySequence>,
    k: usize,
) -> PyResult<Vec<Vec<f64>>> {
    let ordered = seq_to_ordered(items)?;
    Ok(unwrap_bins(kk_partition(ordered, k).bins))
}

#[pyfunction]
#[pyo3(name = "first_fit_decreasing_pack")]
/// Pack numeric items into bins using First-Fit Decreasing with a fixed bin capacity.
fn py_first_fit_decreasing_pack(
    items: &Bound<'_, PySequence>,
    k: f64,
) -> PyResult<Vec<Vec<f64>>> {
    let ordered = seq_to_ordered(items)?;
    Ok(unwrap_bins(
        first_fit_decreasing(ordered, Some(OrderedFloat(k))).bins,
    ))
}

#[pyfunction]
#[pyo3(name = "best_fit_decreasing_pack")]
/// Pack numeric items into bins using Best-Fit Decreasing with a fixed bin capacity.
fn py_best_fit_decreasing_pack(
    items: &Bound<'_, PySequence>,
    k: f64,
) -> PyResult<Vec<Vec<f64>>> {
    let ordered = seq_to_ordered(items)?;
    Ok(unwrap_bins(
        best_fit_decreasing(ordered, Some(OrderedFloat(k))).bins,
    ))
}

#[pyfunction]
#[pyo3(name = "kk")]
/// Backward-compatible alias for `karmarkar_karp_partition`.
fn py_kk(items: &Bound<'_, PySequence>, k: usize) -> PyResult<Vec<Vec<f64>>> {
    py_karmarkar_karp_partition(items, k)
}

#[pyfunction]
#[pyo3(name = "ffd")]
/// Backward-compatible alias for `first_fit_decreasing_pack`.
fn py_ffd(items: &Bound<'_, PySequence>, k: f64) -> PyResult<Vec<Vec<f64>>> {
    py_first_fit_decreasing_pack(items, k)
}

#[pyfunction]
#[pyo3(name = "bfd")]
/// Backward-compatible alias for `best_fit_decreasing_pack`.
fn py_bfd(items: &Bound<'_, PySequence>, k: f64) -> PyResult<Vec<Vec<f64>>> {
    py_best_fit_decreasing_pack(items, k)
}

#[pymodule]
fn seq_len_balance(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(py_karmarkar_karp_partition, m)?)?;
    m.add_function(wrap_pyfunction!(py_first_fit_decreasing_pack, m)?)?;
    m.add_function(wrap_pyfunction!(py_best_fit_decreasing_pack, m)?)?;
    m.add_function(wrap_pyfunction!(py_kk, m)?)?;
    m.add_function(wrap_pyfunction!(py_ffd, m)?)?;
    m.add_function(wrap_pyfunction!(py_bfd, m)?)?;
    Ok(())
}
