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
