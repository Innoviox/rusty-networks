use indicatif::{ProgressBar, ProgressBarIter, ProgressIterator, ProgressStyle};

pub fn dot(input: &Vec<f64>, weights: &Vec<f64>) -> f64 {
    let mut result = weights[0];
    for i in 1..weights.len() {
        result += input[i - 1] * weights[i];
    }

    result
}

// type func = Box<dyn Fn(f64) -> f64>; // might make things look nicer idk

pub fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

pub fn relu(x: f64) -> f64 {
    if x > 0.0 {
        x
    } else {
        0.0
    }
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

pub trait ToVec<T> {
    fn to_vec(self) -> Vec<T>;
}

pub fn progress_bar<'a, T: 'a>(
    iter: impl Iterator<Item = &'a T>,
    len: u64,
    pre_str: &str,
) -> ProgressBarIter<impl Iterator<Item = &'a T>> {
    let bar: ProgressBar =
        ProgressBar::new(len).with_style(ProgressStyle::default_bar().template(
            &(pre_str.to_owned() + " [{eta_precise}] {bar:40.cyan/blue} {pos:>7}/{len:7}"),
        ));

    iter.progress_with(bar)
}

pub fn progress_bar_into<'a, T: 'a>(
    iter: impl Iterator<Item = T>,
    len: u64,
    pre_str: &str,
) -> ProgressBarIter<impl Iterator<Item = T>> {
    let bar: ProgressBar =
        ProgressBar::new(len).with_style(ProgressStyle::default_bar().template(
            &(pre_str.to_owned() + " [{eta_precise}] {bar:40.cyan/blue} {pos:>7}/{len:7}"),
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
