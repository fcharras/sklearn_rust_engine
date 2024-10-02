use std::collections::HashMap;

pub type Vector = Vec<f64>;
pub type Point = Vector;
pub type Points = Vec<Vector>;
pub type DistanceMetric = fn(x: &Vector, y: &Vector) -> f64;

pub type Neighbors<'a> = Vec<&'a Vector>;

// XXX: Attempt at writing a Neighbors trait
// trait Neighbors {
//     fn add(&mut self, x: Vector);
// }

// struct NeighborsSet {
//     neighbors: HashSet<Point>,
// }

// impl Neighbors for NeighborsSet {
//     fn add(&mut self, x: Vector) {
//         self.neighbors.insert(x);
//     }
// }

pub fn nearest_neighbors<'a>(
    point: &Vector,
    x: &'a Points,
    distance_metric: DistanceMetric,
    epsilon: f64,
) -> Neighbors<'a> {
    let mut neighbors = vec![];
    for p in x {
        if distance_metric(point, p) <= epsilon {
            neighbors.push(p);
        }
    }
    neighbors
}

#[derive(Debug, Clone, PartialEq)]
enum ClusterAssignment {
    Noise,
    Cluster(u32),
}
type PointIndex = usize;
pub type ClusterAssignments = HashMap<PointIndex, ClusterAssignment>;

#[derive(PartialEq, Debug)]
pub struct DBSCANResult {
    cluster_assignments: ClusterAssignments,
}

pub fn dbscan(
    x: &Points,
    distance_metric: DistanceMetric,
    epsilon: f64,
    min_points: usize,
) -> DBSCANResult {
    let mut cluster_number = 0;
    let mut cluster_assignments: ClusterAssignments = HashMap::new();

    use ClusterAssignment::*;

    let nearest_neighbors = |point: &Point| -> Neighbors {
        crate::dbscan::nearest_neighbors(point, x, distance_metric, epsilon)
    };

    for (i, point) in x.iter().enumerate() {
        if cluster_assignments.contains_key(&i) {
            continue;
        }

        let mut neighbors = nearest_neighbors(point);
        if neighbors.len() < min_points {
            cluster_assignments.insert(i, Noise);
            continue;
        }

        cluster_number = cluster_number + 1;

        cluster_assignments.insert(i, Cluster(cluster_number));

        let mut seed_set = neighbors;
        for (j, q) in seed_set.iter_mut().enumerate() {
            if *q == point {
                continue;
            }
            if cluster_assignments[&j] == Noise {
                cluster_assignments.insert(j, Cluster(cluster_number));
            }
            if cluster_assignments.contains_key(&j) {
                continue;
            }

            cluster_assignments.insert(j, Cluster(cluster_number));

            neighbors = nearest_neighbors(q);

            if neighbors.len() >= min_points {
                seed_set.extend_from_slice(&neighbors);
            }
        }
    }

    DBSCANResult {
        cluster_assignments,
    }
}

#[cfg(test)]
mod tests {
    use crate::dbscan::dbscan;
    use crate::dbscan::nearest_neighbors;
    use crate::dbscan::DBSCANResult;
    use crate::euclidean_distance::euclidean_distance;

    #[test]
    fn test_nearest_neighbors() {
        let x = vec![vec![0.0, 0.0], vec![1.0, 1.0]];
        let point = vec![0.0, 2.0];

        assert_eq!(
            nearest_neighbors(&point, &x, euclidean_distance, 1.5),
            vec![&vec![1.0, 1.0]]
        );
    }

    #[test]
    fn test_dbscan() {
        use std::collections::HashMap;
        
        let x = vec![vec![0.0, 0.0], vec![1.0, 1.0]];

        assert_eq!(
            dbscan(&x, euclidean_distance, 1.0, 1),
            DBSCANResult {
                cluster_assignments: HashMap::new()
            }
        );
    }
}
