use crate::convolution::Transform;
use serde::{Deserialize, Serialize};

#[derive(Deserialize, Serialize)]
pub struct Config {
    pub weights: Vec<Vec<Vec<f64>>>,
    pub activation: Vec<String>,
    pub loss: String,
    pub transforms: Vec<Transform>,
    pub shape: Vec<u64>,
}
