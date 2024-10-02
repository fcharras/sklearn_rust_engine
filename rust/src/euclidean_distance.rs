pub fn euclidean_distance(x: &Vec<f64>, y: &Vec<f64>) -> f64 {
    x.iter()
        .zip(y.iter())
        .map(|(x_, y_)| (x_ - y_).powf(2.0))
        .sum::<f64>()
        .sqrt()
}

#[test]
fn test_euclidean_distance() {
    let x = vec![0.0, 0.0];
    let y = vec![1.0, 1.0];
    let z = vec![0.0, 2.0];

    assert_eq!(euclidean_distance(&x, &x), 0.0);
    assert_eq!(euclidean_distance(&x, &y), 2.0_f64.sqrt());
    assert_eq!(euclidean_distance(&x, &z), 2.0);
}
