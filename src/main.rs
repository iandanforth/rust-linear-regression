// Linear Regression Translation from octave into rust
extern crate csv;
extern crate nalgebra as na;

use std::fs::File;
use na::{DMatrix, Vector2};

fn load(
    filename: &'static str,
) -> na::Matrix<f64, na::Dynamic, na::Dynamic, na::MatrixVec<f64, na::Dynamic, na::Dynamic>> {
    let file = File::open(filename).expect("Blur blur");
    let mut rdr = csv::ReaderBuilder::new()
        .has_headers(false)
        .from_reader(file);
    let mut all_vals = Vec::new();
    let mut col_count = 0;
    let mut row_count = 0;
    let mut temp_count = 0;

    for result in rdr.records() {
        let record = result.expect("Bad line");
        for field in record.iter() {
            let f = field.parse::<f64>().expect("Could not parse as f64");
            all_vals.push(f);
            temp_count += 1;
        }
        if col_count == 0 {
            col_count = temp_count;
        }
        row_count += 1;
    }

    // Use this rather than from_iterator because this assumes row major
    // rather than column major orientation of all_vals
    let dmatrix = DMatrix::from_row_slice(row_count, col_count, &all_vals);

    return dmatrix;
}

fn sum(vec: na::Matrix<f64, na::Dynamic, na::U1, na::MatrixVec<f64, na::Dynamic, na::U1>>) -> f64 {
    let mut total = 0.0;
    for val in vec.iter() {
        total += val;
    }

    return total;
}

fn compute_cost(
    X: &na::Matrix<f64, na::Dynamic, na::U2, na::MatrixVec<f64, na::Dynamic, na::U2>>,
    y: &na::Matrix<f64, na::Dynamic, na::U1, na::MatrixVec<f64, na::Dynamic, na::U1>>,
    theta: &na::Matrix<f64, na::U2, na::U1, na::MatrixArray<f64, na::U2, na::U1>>,
) -> f64 {
    let m = y.len();
    let scalar = 1.0 / (2.0 * m as f64);
    let prod = X * theta;
    let err = prod - y;
    let sqr = power(err, 2);
    let total = sum(sqr);
    let J = scalar * total;
    return J;
}

fn power(
    mut vec: na::Matrix<f64, na::Dynamic, na::U1, na::MatrixVec<f64, na::Dynamic, na::U1>>,
    exp: i32,
) -> na::Matrix<f64, na::Dynamic, na::U1, na::MatrixVec<f64, na::Dynamic, na::U1>> {
    for ind in 0..vec.len() {
        vec[ind] = vec[ind].powi(exp);
    }

    return vec;
}

fn gradient_descent(
    X: &na::Matrix<f64, na::Dynamic, na::U2, na::MatrixVec<f64, na::Dynamic, na::U2>>,
    y: &na::Matrix<f64, na::Dynamic, na::U1, na::MatrixVec<f64, na::Dynamic, na::U1>>,
    theta: na::Matrix<f64, na::U2, na::U1, na::MatrixArray<f64, na::U2, na::U1>>,
    alpha: f64,
    iterations: i32,
) -> na::Matrix<f64, na::U2, na::U1, na::MatrixArray<f64, na::U2, na::U1>> {
    let m = y.len();
    let scalar = 1.0 / m as f64;
    // Vectorized gradient calculation
    let mut learned_theta = theta.clone();
    let mut prod;
    let mut grad;
    let mut update;
    for _i in 0..iterations {
        prod = X * learned_theta;
        grad = scalar * (X.transpose() * (prod - y));
        update = alpha * grad;
        learned_theta = learned_theta - update;
    }

    return learned_theta;
}

fn main() {
    // Set up data
    let data = load("ex1data1.txt");
    // Pull out the first column then add a 1's column
    let X = data.column(0).insert_column(0, 1.0);
    // Pull out the final column as the labels
    let y = data.column(1).clone_owned();

    let theta = Vector2::new(0.0, 0.0);

    let iterations = 1500;
    let alpha = 0.01;

    let mut cost = compute_cost(&X, &y, &theta);
    println!("With theta = [0; 0] \nComputed Cost = {:?}", &cost);
    println!("Expected cost value (approx) 32.07");

    let theta_2 = Vector2::new(-1.0, 2.0);
    cost = compute_cost(&X, &y, &theta_2);
    println!("With theta = [-1; 2] \nComputed Cost = {:?}", &cost);
    println!("Expected cost value (approx) 54.24");

    let learned_theta = gradient_descent(&X, &y, theta, alpha, iterations);
    println!("Theta found by gradient descent:");
    println!("{:?}", &learned_theta);
    println!("Expected values (approx)");
    println!(" -3.6303 \n 1.1664");

    let predict1 = learned_theta.transpose() * Vector2::new(1.0, 3.5);
    let predict2 = learned_theta.transpose() * Vector2::new(1.0, 7.0);
    println!("For population = 35,000, we predict a profit of 4519.767868:");
    println!("{:?}", (predict1 * 10000.0)[0]);
    println!("For population = 70,000, we predict a profit of 45342.450129:");
    println!("{:?}", (predict2 * 10000.0)[0]);
}
