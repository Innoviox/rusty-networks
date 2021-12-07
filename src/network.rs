use crate::convolution;
use crate::optimizers;
use crate::utils;
use crate::utils::{dot, progress_bar};
use rand::Rng;
use std::fmt;

pub struct Network {
    // layers: Vec<Layer>,
    weights: Vec<Vec<Vec<f64>>>, // vector of layers of nodes of weights
    activation: Box<&'static dyn Fn(f64) -> f64>,
    loss: Box<dyn Fn(&Vec<f64>, &Vec<f64>) -> f64>,
    transforms: Vec<convolution::Transform>,
    optimizer: Box<dyn optimizers::Optimizer>,
}

impl Network {
    pub fn new(shape: Vec<u64>) -> Network {
        // shape is number of nodes in each layer
        let mut rng = rand::thread_rng();

        let mut weights = vec![];
        let mut last_length = shape[0];
        for l in shape.iter().skip(1) {
            // skip first layer
            let mut nodes = vec![];
            for _i in 0..*l {
                nodes.push(
                    (0..last_length + 1) // +1 cause bias
                        .map(|_| rng.gen_range(-1.0..1.0))
                        .collect(),
                );
            }
            weights.push(nodes);
            last_length = *l;
        }

        Network {
            weights,
            activation: Box::new(&utils::sigmoid),
            loss: Box::new(utils::mse),
            transforms: vec![],
            //	    optimizer: optimizers::GradDescent::new(),
            optimizer: optimizers::Adam::new(&shape, 0.9, 0.9, 0.99),
        }
    }

    pub fn evaluate(&self, input: &Vec<f64>) -> Vec<f64> {
        let mut values: Vec<f64> = input.to_vec();
        for layer in &self.weights {
            // values = layer.evaluate(&values, &self.activation);
            values = layer
                .iter()
                .map(|node| (self.activation)(dot(&values, node)))
                .collect();
        }

        values
    }

    fn change_weight(&mut self, layer_n: usize, node_n: usize, weight_n: usize, delta: f64) {
        self.weights[layer_n][node_n][weight_n] += delta;
    }

    // fn change_bias(&mut self, layer_n: usize, node_n: usize, delta: f64) {
    //     self.layers[layer_n].nodes[node_n].bias += delta;
    // }

    fn train(
        &mut self,
        training_data: &Vec<Vec<f64>>,
        correct_output: &Vec<Vec<f64>>,
        epoch: usize,
    ) {
        for i in progress_bar(
            0..training_data.len(),
            training_data.len() as u64,
            &format!("Epoch {}:", epoch),
        ) {
            // println!("{}", i);
            let input = &training_data[i];
            let output = &correct_output[i];

            let result = self.evaluate(input);
            let base_loss = (self.loss)(&result, output);

            let mut gradient = self.make_weights_grid();

            for layer_n in 0..self.weights.len() {
                for node_n in 0..self.weights[layer_n].len() {
                    for weight_n in 0..self.weights[layer_n][node_n].len() {
                        self.change_weight(layer_n, node_n, weight_n, 0.01);

                        let new_loss = (self.loss)(&(self.evaluate(input)), output);
                        let del = (new_loss - base_loss) / 0.01;

                        gradient[layer_n][node_n][weight_n] = del;

                        self.change_weight(layer_n, node_n, weight_n, -0.01);
                    }
                }
            }

            let dw = self.optimizer.optimize(&gradient);

            for layer_n in 0..dw.len() {
                for node_n in 0..dw[layer_n].len() {
                    for weight_n in 0..dw[layer_n][node_n].len() {
                        self.change_weight(
                            layer_n,
                            node_n,
                            weight_n,
                            dw[layer_n][node_n][weight_n],
                        );
                    }
                }
            }

            // todo: store bias in weights array of neuron
            // todo: store previous delta, use node[weight_index] += (rate * weight_update) + (momentum * prev_delta);
            // todo: halt condition
        }
    }

    pub fn train_epochs(
        &mut self,
        training_data: &Vec<Vec<f64>>,
        correct_output: &Vec<Vec<f64>>,
        epochs: usize,
    ) {
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

    // fn set_loss_function(&mut self, loss: impl Fn(i64) -> i64) {
    //     unimplemented!();
    // }

    pub fn optimizer(&mut self, opt: Box<dyn optimizers::Optimizer>) -> &mut Self {
        self.optimizer = opt;
        self
    }

    pub fn activation(&mut self, act: &'static dyn Fn(f64) -> f64) -> &mut Self {
        self.activation = Box::new(act);
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

    pub fn apply_transforms(&self, input: &Vec<f64>) -> Vec<f64> {
        let mut matrix2d = vec![];
        let mut result_vec = input.clone();
        for t in &self.transforms {
            match t {
                convolution::Transform::Convolve2D(kernel, width) => {
                    if matrix2d.len() == 0 {
                        let mut row = vec![];
                        for (idx, val) in input.iter().enumerate() {
                            if idx == ((matrix2d.len() + 1) * width) {
                                matrix2d.push(row);
                                row = vec![];
                            }
                            row.push(val.clone());
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
