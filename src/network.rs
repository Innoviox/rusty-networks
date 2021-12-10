use crate::config::Config;
use crate::convolution;
use crate::optimizers;
use crate::utils;
use crate::utils::{dot, load, progress_bar, Activation, Loss};
use bincode::serialize_into;
use rand::Rng;
use std::fmt;
use std::fs::File;
use std::io::BufWriter;

pub struct Network {
    // layers: Vec<Layer>,
    weights: Vec<Vec<Vec<f64>>>, // vector of layers of nodes of weights
    activation: Vec<&'static Activation>,
    loss: &'static Loss,
    transforms: Vec<convolution::Transform>,
    optimizer: Box<dyn optimizers::Optimizer>,
    shape: Vec<u64>,
}

impl Network {
    pub fn default() -> Network {
        Network {
            weights: vec![],
            activation: vec![],
            loss: &utils::categorical_cross_entropy,
            transforms: vec![],
            optimizer: optimizers::GradDescent::new(),
            // optimizer: optimizers::Adam::new(&vec![625, 10, 10], 0.01, 0.9, 0.99),
            // optimizer: optimizers::Adam::new(&shape, 0.9, 0.9, 0.99),
            shape: vec![],
        }
    }

    pub fn with_weights(weights: Vec<Vec<Vec<f64>>>) -> Network {
        let mut n = Network::default();
        n.weights = weights;

        n
    }

    pub fn from_file(filename: &str) -> Network {
        // Network::with_weights(load(filename).unwrap())
        Network::from_config(load(filename).unwrap())
    }

    pub fn add_layer(&mut self, n: u64, activation: &'static Activation) -> &mut Self {
        self.shape.push(n);
        if self.shape.len() > 1 {
            self.activation.push(activation);
        }

        self
    }

    pub fn predict(&self, input: &[f64]) -> Vec<f64> {
        self.evaluate(&self.apply_transforms(input))
    }

    fn evaluate(&self, input: &[f64]) -> Vec<f64> {
        let mut values: Vec<f64> = input.to_vec();
        for layer_n in 0..self.weights.len() {
            values = (self.activation[layer_n])(
                &self.weights[layer_n]
                    .iter()
                    .map(|node| dot(&values, node))
                    .collect::<Vec<f64>>(),
            );
        }

        values
    }

    // fn change_weight(&mut self, layer_n: usize, node_n: usize, weight_n: usize, delta: f64) {
    //     self.weights[layer_n][node_n][weight_n] += delta;
    // }

    // fn change_bias(&mut self, layer_n: usize, node_n: usize, delta: f64) {
    //     self.layers[layer_n].nodes[node_n].bias += delta;
    // }

    fn train(
        &mut self,
        training_data: &Vec<Vec<f64>>,
        correct_output: &Vec<Vec<f64>>,
        epoch: usize,
    ) {
        let mut sum_loss = 0.0;
        for i in progress_bar(
            0..training_data.len(),
            training_data.len() as u64,
            &format!("Epoch {}:", epoch),
        ) {
            // for i in 0..training_data.len() {
            // println!("{}", i);
            let input = &training_data[i];
            let output = &correct_output[i];

            let result = self.evaluate(input);
            let base_loss = (self.loss)(&result, output);
            sum_loss += base_loss.abs();

            let mut gradient = self.make_weights_grid();

            for layer_n in 0..self.weights.len() {
                for node_n in 0..self.weights[layer_n].len() {
                    for weight_n in 0..self.weights[layer_n][node_n].len() {
                        self.weights[layer_n][node_n][weight_n] += 0.01;

                        let new_loss = (self.loss)(&(self.evaluate(input)), output);
                        let del = (new_loss - base_loss) / 0.01;

                        gradient[layer_n][node_n][weight_n] = del;

                        self.weights[layer_n][node_n][weight_n] -= 0.01;
                    }
                }
            }

            let dw = self.optimizer.optimize(&gradient);

            for layer_n in 0..dw.len() {
                for node_n in 0..dw[layer_n].len() {
                    for weight_n in 0..dw[layer_n][node_n].len() {
                        self.weights[layer_n][node_n][weight_n] += dw[layer_n][node_n][weight_n];
                    }
                }
            }

            // println!("hi {:?} {:?}", result, base_loss);

            // todo: store bias in weights array of neuron
            // todo: store previous delta, use node[weight_index] += (rate * weight_update) + (momentum * prev_delta);
            // todo: halt condition
        }
        println!("Loss: {}", sum_loss / (training_data.len() as f64));
    }

    pub fn train_epochs(
        &mut self,
        training_data: &Vec<Vec<f64>>,
        correct_output: &Vec<Vec<f64>>,
        epochs: usize,
    ) {
        self.set_up_weights();
        self.optimizer.with_shape(&self.shape);

        let transformed_data = progress_bar(
            training_data.iter(),
            training_data.len() as u64,
            "Convolving:",
        )
        .map(|i| self.apply_transforms(i))
        .collect();

        for i in 0..epochs {
            self.train(&transformed_data, correct_output, i + 1);
        }
    }

    fn set_up_weights(&mut self) {
        if !self.weights.is_empty() {
            return;
        }

        let mut rng = rand::thread_rng();

        self.weights = vec![];
        let mut last_length = self.shape[0];
        for l in self.shape.iter().skip(1) {
            // skip first layer
            let mut nodes = vec![];
            for _i in 0..*l {
                nodes.push(
                    (0..last_length + 1) // +1 cause bias
                        .map(|_| rng.gen_range(-1.0..1.0))
                        .collect(),
                );
            }
            self.weights.push(nodes);
            last_length = *l;
        }
    }

    // fn set_loss_function(&mut self, loss: impl Fn(i64) -> i64) {
    //     unimplemented!();
    // }

    pub fn optimizer(&mut self, opt: Box<dyn optimizers::Optimizer>) -> &mut Self {
        self.optimizer = opt;
        self
    }

    // pub fn activation(&mut self, act: &'static dyn Fn(f64) -> f64) -> &mut Self {
    //     self.activation = Box::new(act);
    //     self
    // }

    pub fn loss(&mut self, loss: &'static Loss) -> &mut Self {
        self.loss = loss;
        self
    }

    pub fn make_weights_grid(&self) -> Vec<Vec<Vec<f64>>> {
        let mut gradient = vec![];

        for layer_n in 0..self.weights.len() {
            let mut layer_gradient = vec![];
            for node_n in 0..self.weights[layer_n].len() {
                let mut neuron_gradient = vec![];
                for _weight_n in 0..self.weights[layer_n][node_n].len() {
                    neuron_gradient.push(0.0f64);
                }

                layer_gradient.push(neuron_gradient);
            }

            gradient.push(layer_gradient);
        }

        gradient
    }

    pub fn add_transform(&mut self, transform: convolution::Transform) -> &mut Self {
        self.transforms.push(transform);
        self
    }

    pub fn apply_transforms(&self, input: &[f64]) -> Vec<f64> {
        let mut matrix2d = vec![];
        let mut result_vec = input.to_owned();
        for t in &self.transforms {
            match t {
                convolution::Transform::Convolve2D(kernel, width) => {
                    if matrix2d.is_empty() {
                        let mut row = vec![];
                        for (idx, val) in input.iter().enumerate() {
                            if idx == ((matrix2d.len() + 1) * width) {
                                matrix2d.push(row);
                                row = vec![];
                            }
                            row.push(*val);
                        }
                        matrix2d.push(row);
                    }

                    matrix2d = convolution::convolve(matrix2d, kernel.to_vec());
                }
                convolution::Transform::MaxPool(dim) => {
                    matrix2d = convolution::max_pool(matrix2d, (dim.0, dim.1));
                }
                convolution::Transform::Flatten() => result_vec = convolution::flatten(&matrix2d),
            }
        }

        result_vec
    }

    pub fn save(&self, filename: &str) {
        let mut f = BufWriter::new(File::create(filename).unwrap());
        serialize_into(&mut f, &self.to_config()).unwrap();
        println!("Saved to file {}", filename);
    }

    pub fn to_config(&self) -> Config {
        let mut a = vec![];
        for i in self.activation.iter() {
            a.push(utils::activation_to_str(i).to_string())
        }
        Config {
            weights: self.weights.clone(),
            activation: a,
            loss: utils::loss_to_str(&self.loss).to_string(),
            transforms: self.transforms.to_vec(),
            shape: self.shape.clone(),
        }
    }

    pub fn from_config(c: Config) -> Network {
        let mut a = vec![];
        for i in c.activation {
            a.push(utils::str_to_activation(i).unwrap());
        }
        Network {
            weights: c.weights,
            activation: a,
            loss: utils::str_to_loss(c.loss).unwrap(),
            transforms: c.transforms,
            optimizer: optimizers::GradDescent::new(),
            shape: c.shape,
        }
    }
}

impl fmt::Display for Network {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        // Write strictly the first element into the supplied output
        // stream: `f`. Returns `fmt::Result` which indicates whether the
        // operation succeeded or failed. Note that `write!` uses syntax which
        // is very similar to `println!`.
        write!(f, "hi")
    }
}
