use indicatif::{ProgressBar, ProgressBarIter, ProgressIterator, ProgressStyle};
use serde::de::DeserializeOwned;
use serde::Serialize;
use std::f64::consts::E;
use std::fs;

pub type Activation = dyn Fn(&[f64]) -> Vec<f64>;
pub type Loss = dyn Fn(&[f64], &[f64]) -> f64;

pub fn dot(input: &[f64], weights: &[f64]) -> f64 {
    let mut result = weights[0];
    for i in 1..weights.len() {
        result += input[i - 1] * weights[i];
    }

    result
}

// type func = Box<dyn Fn(f64) -> f64>; // might make things look nicer idk

pub fn sigmoid(x: &[f64]) -> Vec<f64> {
    x.iter().map(|i| 1.0 / (1.0 + (-i).exp())).collect()
}

pub fn relu(x: &[f64]) -> Vec<f64> {
    x.iter().map(|i| if *i > 0.0 { *i } else { 0.0 }).collect()
}

pub fn softmax(x: &[f64]) -> Vec<f64> {
    let mut result = vec![];
    let y: Vec<f64> = x.to_vec();
    let sum = y.iter().fold(0.0, |a, b| a + E.powf(*b));
    for j in y {
        result.push(E.powf(j) / sum);
    }
    result
}

pub fn activation_to_str<'a>(a: &'a Activation) -> &'a str {
    let v = vec![0.01, 0.19, 0.03, 0.08, 0.14, 0.27, 0.03, 0.11, 0.12, 0.01];
    let e = a(&v);
    if e == sigmoid(&v) {
        "sigmoid"
    } else if e == relu(&v) {
        "relu"
    } else if e == softmax(&v) {
        "softmax"
    } else {
        "unknown"
    }
}

pub fn str_to_activation<'a>(s: String) -> Option<&'a Activation> {
    match s.as_ref() {
        "sigmoid" => Some(&sigmoid),
        "relu" => Some(&relu),
        "softmax" => Some(&softmax),
        _ => None,
    }
}

// mean squared error: (actually total squared error):
// sum squares of difference between output and actual expected value
// goal is to minimize this
pub fn mse(output: &[f64], expected: &[f64]) -> f64 {
    // gotta be same length
    let mut result = 0.0;
    for i in 0..output.len() {
        result += (output[i] - expected[i]).powf(2.0)
    }

    result
}

pub fn categorical_cross_entropy(output: &[f64], expected: &[f64]) -> f64 {
    // -output[argmax(expected)].log10()
    let p = output[argmax(expected)];
    // -(p.log10() + (1.0 - p).log10())
    -p.log10()
}

pub fn loss_to_str<'a>(a: &'a Loss) -> &'a str {
    let v1 = vec![0.01, 0.19, 0.03, 0.08, 0.14, 0.27, 0.03, 0.11, 0.12, 0.01];
    let v2 = vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    let e = a(&v1, &v2);
    if (e - mse(&v1, &v2)).abs() < 0.01 {
        "mse"
    } else if (e - categorical_cross_entropy(&v1, &v2)).abs() < 0.01 {
        "cce"
    } else {
        "unknown"
    }
}

pub fn str_to_loss<'a>(s: String) -> Option<&'a Loss> {
    match s.as_ref() {
        "mse" => Some(&mse),
        "cce" => Some(&categorical_cross_entropy),
        _ => None,
    }
}

pub trait ToVec<T> {
    fn to_vec(self) -> Vec<T>;
}

pub fn progress_bar<'a, T: 'a>(
    iter: impl Iterator<Item = T>,
    len: u64,
    description: &str,
) -> ProgressBarIter<impl Iterator<Item = T>> {
    let bar: ProgressBar = ProgressBar::new(len).with_style(ProgressStyle::default_bar().template(
        &(description.to_owned() + " [{eta_precise}] {bar:40.cyan/blue} {pos:>7}/{len:7}"),
    ));

    iter.progress_with(bar)
}

pub fn argmax(input: &[f64]) -> usize {
    let mut max = 0.0;
    let mut idx = 0;

    for (i, j) in input.iter().enumerate() {
        if j > &max {
            max = *j;
            idx = i;
        }
    }

    idx
}

pub fn load<T: DeserializeOwned + Serialize>(file: &str) -> Option<T> {
    match fs::read(file) {
        Ok(b) => {
            println!("Loaded from file {}", file);
            Some(bincode::deserialize(&b).unwrap())
        }
        Err(_) => {
            println!("File not found {}", file);
            None
        }
    }
}
