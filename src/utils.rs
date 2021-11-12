pub fn dot(a: &Vec<f64>, b: &Vec<f64>) -> f64 {
    if a.len() != b.len() {
        unimplemented!() // forgot how to raise errors
    }

    let mut result = 0.0;
    for i in 0..a.len() {
        result += a[i] * b[i];
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
pub fn mse(output: Vec<f64>, expected: Vec<f64>) -> f64 {
    // gotta be same length
    let mut result = 0.0;
    for i in 0..output.len() {
        result += (output[i] - expected[i]).powf(2.0)
    }

    result
}
