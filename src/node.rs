use crate::utils::dot;

pub struct Node {
    pub weights: Vec<f64>, // first weight is bias
    // pub bias: f64, // i think each node has one bias, not one bias per input
    pub value: i64,
}

impl Node {
    pub fn evaluate(&self, input: &Vec<f64>, activation: &Box<dyn Fn(f64) -> f64>) -> f64 {
        let z = dot(input, &self.weights);
        activation(z)
    }
}
