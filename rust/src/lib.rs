pub mod dbscan;
pub mod euclidean_distance;

use numpy::ndarray::ArrayBase;
use numpy::ndarray::{ArrayD, ArrayView2, ArrayViewD, Axis};
use numpy::{
    IntoPyArray, NotContiguousError, PyArrayDyn, PyReadonlyArray1, PyReadonlyArray2,
    PyReadonlyArrayDyn, PyReadwriteArray1, PyReadwriteArray2,
};
use pyo3::{pymodule, types::PyModule, Bound, PyResult, Python};
use std::iter::Enumerate;
use std::ops::{Mul, Sub};

#[pymodule]
fn sklearn_rust_engine<'py>(m: &Bound<'py, PyModule>) -> PyResult<()> {
    // example using immutable borrows producing a new array
    fn axpy_(a: f64, x: ArrayViewD<'_, f64>, y: ArrayViewD<'_, f64>) -> ArrayD<f64> {
        a * &x + &y
    }
    // wrapper of `axpy`
    #[pyfn(m)]
    #[pyo3(name = "truc")]
    fn axpy<'py>(
        py: Python<'py>,
        a: f64,
        x: PyReadonlyArrayDyn<'py, f64>,
        y: PyReadonlyArrayDyn<'py, f64>,
    ) -> Bound<'py, PyArrayDyn<f64>> {
        let x = x.as_array();
        let y = y.as_array();
        let z = axpy_(a, x, y);
        IntoPyArray::into_pyarray_bound(z, py)
    }

    //     def lloyd_iter_chunked_dense(
    //             const floating[:, ::1] X,            # IN
    //             const floating[::1] sample_weight,   # IN
    //             const floating[:, ::1] centers_old,  # IN
    //             floating[:, ::1] centers_new,        # OUT
    //             floating[::1] weight_in_clusters,    # OUT
    //             int[::1] labels,                     # OUT
    //             floating[::1] center_shift,          # OUT
    //             int n_threads,
    //             bint update_centers=True):

    #[pyfn(m)]
    // #[pyo3(name = "truc")]
    fn lloyd_iter_chunked_dense(
        X: PyReadonlyArray2<f64>,
        centers_old: PyReadonlyArray2<f64>,
        mut centers_new: PyReadwriteArray2<f64>,
        mut labels: PyReadwriteArray1<i64>,
        mut center_shift: PyReadwriteArray1<f64>,
        update_centers: bool,
    ) -> PyResult<()> {
        let X_view: ArrayView2<f64> = X.as_array();
        let centers_view: ArrayView2<f64> = centers_old.as_array();

        for (sample_idx, sample) in X_view.axis_iter(Axis(0)).enumerate() {
            // println!("sample {sample_idx}\n============================");
            let mut closest_center_idx = 0;
            let mut smallest_squared_dist = f64::INFINITY;
            for (idx, center) in centers_view.axis_iter(Axis(0)).enumerate() {
                let diff = sample.sub(&center);
                let squared = (&diff).mul(&diff);
                let squared_dist = squared.sum();
                // println!("squared dist: {}", squared_dist);
                if squared_dist < smallest_squared_dist {
                    smallest_squared_dist = squared_dist;
                    closest_center_idx = idx;
                    // println!("closest centroid: {closest_center_idx}");
                }
            }
            (*labels.get_mut(sample_idx).unwrap()) = closest_center_idx as i64;
            // Process each row here
            // For example, print the row
        }
        // dbg!("{}", labels);

        let first_item = centers_new.get_mut([0, 0]).unwrap();
        *first_item = 100.;

        centers_new
            .as_slice_mut()
            .unwrap()
            .iter_mut()
            .for_each(|x| {
                *x = *x + 1.;
            });

        centers_new
            .as_slice_mut()
            .unwrap()
            .iter_mut()
            .map(|x| {
                *x = *x + 1.;
            })
            .count();

        let _ = centers_new
            .as_slice_mut()
            .unwrap()
            .iter_mut()
            .map(|x| {
                *x = *x + 1.;
            })
            .collect::<()>();

        let centers_new_slice = centers_new.as_slice_mut().unwrap();
        for item in centers_new_slice {
            *item = *item + 1.;
        }

        Ok(())
    }

    /// Formats the sum of two numbers as string.
    #[pyfn(m)]
    #[pyo3()]
    fn my_sum_as_string(a: usize, b: usize) -> PyResult<String> {
        Ok((a + b).to_string())
    }

    Ok(())
}

fn distance(x: &Vec<f64>, y: &Vec<f64>) -> f64 {
    x.iter()
        .zip(y.iter())
        .map(|(x_, y_)| (x_ - y_).powf(2.0))
        .sum()
}

#[derive(Debug, PartialEq)]
struct KMeansResult {
    cluster_centroids: Vec<Vec<f64>>,
    // For each point in x, which cluster it is assigned to
    cluster_assignments: Vec<i64>,
}

fn kmeans(x: Vec<Vec<f64>>, n_clusters: usize) -> KMeansResult {
    // Initialize a vec of centroids taken randomly from the dataset
    // NB: assumption is made that our dataset has at least n_clusters row.
    let mut cluster_centroids: Vec<Vec<f64>> = vec![];
    for i in 0..n_clusters {
        cluster_centroids.push(x[i].clone())
    }

    // Expectation step

    let mut closest_centroids_idx: Vec<u64>;
    let mut closest_centroids_distance: Vec<f64>;

    for point in x {
        let mut closest_centroid_idx: u64 = 0;
        let mut closest_centroid_distance: f64 = f64::INFINITY;

        for (idx, centroid) in cluster_centroids.iter().enumerate() {
            distance(&point, &centroid);
        }
    }

    KMeansResult {
        cluster_centroids: vec![],
        cluster_assignments: vec![],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_distance() {
        let x = vec![0.0, 0.0];
        let y = vec![1.0, 0.0];

        assert_eq!(distance(&x, &y), 1.0);
    }

    #[test]
    fn it_works() {
        // let x = vec![vec![0.0, 0.0], vec![1.0, 1.0]];
        let x = vec![
            vec![0.0, 0.0],
            vec![0.5, 0.0],
            vec![0.5, 1.0],
            vec![1.0, 1.0],
        ];
        let result = kmeans(x, 2);
        assert_eq!(
            result,
            KMeansResult {
                cluster_centroids: vec![vec![148.], vec![2.]],
                cluster_assignments: vec![],
            }
        );
    }
}
