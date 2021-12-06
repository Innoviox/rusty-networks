pub fn dot(input: &Vec<f64>, weights: &Vec<f64>) -> f64 {
    let mut result = weights[0];
    for i in 1..weights.len() {
        result += input[i - 1] * weights[i];
    }

    result
}

type func = Box<dyn Fn(f64) -> f64>; // might make things look nicer idk

pub fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
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
