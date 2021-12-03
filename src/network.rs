use crate::layer::Layer;
use crate::node::Node;
use crate::utils;
use rand::Rng;
use std::fmt;

pub struct Network {
    layers: Vec<Layer>,
    activation: Box<dyn Fn(f64) -> f64>,
    loss: Box<dyn Fn(&Vec<f64>, &Vec<f64>) -> f64>,
}

impl Network {
    pub fn new(shape: Vec<u64>) -> Network {
        // shape is number of nodes in each layer
        let mut rng = rand::thread_rng();

        let mut layers = vec![];
        let mut last_length = shape[0];
        for l in shape.iter().skip(1) {
            // skip first layer
            let mut nodes = vec![];
            for i in 0..*l {
                let n = Node {
                    weights: (0..last_length + 1) // +1 cause bias
                        .map(|_| rng.gen_range(0.0..1.0))
                        .collect(),
                    // bias: rng.gen_range(0.0..1.0),
                    value: 0,
                };
                nodes.push(n);
            }
            layers.push(Layer { nodes });
            last_length = *l;
        }

        Network {
            layers,
            activation: Box::new(utils::sigmoid),
            loss: Box::new(utils::mse),
        }
    }

    pub fn evaluate(&self, input: &Vec<f64>) -> Vec<f64> {
        let mut values: Vec<f64> = input.to_vec();
        for layer in &self.layers {
            values = layer.evaluate(&values, &self.activation);
        }

        values
    }

    fn change_weight(&mut self, layer_n: usize, node_n: usize, weight_n: usize, delta: f64) {
        self.layers[layer_n].nodes[node_n].weights[weight_n] += delta;
    }

    // fn change_bias(&mut self, layer_n: usize, node_n: usize, delta: f64) {
    //     self.layers[layer_n].nodes[node_n].bias += delta;
    // }

    fn train(&mut self, training_data: &Vec<Vec<f64>>, correct_output: &Vec<Vec<f64>>) {
        for i in 0..training_data.len() {
            let input = &training_data[i];
            let output = &correct_output[i];

            let result = self.evaluate(input);
            let base_loss = (self.loss)(&result, output);

            let mut gradient = self.make_weights_grid();

            for layer_n in 0..self.layers.len() {
                for node_n in 0..self.layers[layer_n].nodes.len() {
                    for weight_n in 0..self.layers[layer_n].nodes[node_n].weights.len() {
                        self.change_weight(layer_n, node_n, weight_n, 0.01);

                        let new_loss = (self.loss)(&(self.evaluate(input)), output);
                        let del = (new_loss - base_loss) / 0.01;

                        gradient[layer_n][node_n][weight_n] = del;

                        self.change_weight(layer_n, node_n, weight_n, -0.01);
                    }
                }
            }

            for layer_n in 0..gradient.len() {
                for node_n in 0..gradient[layer_n].len() {
                    for weight_n in 0..gradient[layer_n][node_n].len() {
                        self.change_weight(
                            layer_n,
                            node_n,
                            weight_n,
                            -gradient[layer_n][node_n][weight_n],
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
        for i in 0..epochs {
            self.train(training_data, correct_output);
        }
    }

    fn set_loss_function(&mut self, loss: impl Fn(i64) -> i64) {
        unimplemented!();
    }

    pub fn set_optimizer_function() {
        unimplemented!();
    }

    fn make_weights_grid(&self) -> Vec<Vec<Vec<f64>>> {
        let mut gradient = vec![];

        for layer_n in 0..self.layers.len() {
            let mut layer_gradient = vec![];
            for node_n in 0..self.layers[layer_n].nodes.len() {
                let mut neuron_gradient = vec![];
                for weight_n in 0..self.layers[layer_n].nodes[node_n].weights.len() {
                    neuron_gradient.push(0.0f64);
                }

                layer_gradient.push(neuron_gradient);
            }

            gradient.push(layer_gradient);
        }

        gradient
    }

    pub fn convolve(&self, kernel: Vec<Vec<f64>>, stride: usize, padding: usize) {
        // ignore stride and padding for now
    }

    pub fn max_pool(&self) {}
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
