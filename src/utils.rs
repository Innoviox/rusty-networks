use indicatif::{ProgressBar, ProgressBarIter, ProgressIterator, ProgressStyle};
use serde::de::DeserializeOwned;
use serde::Serialize;
use std::f64::consts::E;
use std::fs;

pub fn dot(input: &Vec<f64>, weights: &Vec<f64>) -> f64 {
    let mut result = weights[0];
    for i in 1..weights.len() {
        result += input[i - 1] * weights[i];
    }

    result
}

// type func = Box<dyn Fn(f64) -> f64>; // might make things look nicer idk

pub fn sigmoid(x: Vec<f64>) -> Vec<f64> {
    x.iter().map(|i| 1.0 / (1.0 + (-i).exp())).collect()
}

pub fn relu(x: Vec<f64>) -> Vec<f64> {
    x.iter().map(|i| if *i > 0.0 { *i } else { 0.0 }).collect()
}

pub fn softmax(x: Vec<f64>) -> Vec<f64> {
    let mut result = vec![];
    let y: Vec<f64> = x.iter().cloned().collect();
    let sum = y.iter().fold(0.0, |a, b| a + E.powf(*b));
    for j in y {
        result.push(E.powf(j) / sum);
    }
    result
}

// mean squared error: (actually total squared error):
// sum squares of difference between output and actual expected value
// goal is to minimize this
pub fn mse(output: &Vec<f64>, expected: &Vec<f64>) -> f64 {
    // gotta be same length
    let mut result = 0.0;
    for i in 0..output.len() {
        result += (output[i] - expected[i]).powf(2.0)
    }

    result
}

pub fn categorical_cross_entropy(output: &Vec<f64>, expected: &Vec<f64>) -> f64 {
    // -output[argmax(expected)].log10()
    let p = output[argmax(expected)];
    // -(p.log10() + (1.0 - p).log10())
    -p.log10()
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

pub fn argmax(input: &Vec<f64>) -> usize {
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
        Err(_) => None,
    }
}
