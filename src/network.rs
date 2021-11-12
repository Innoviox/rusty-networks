use crate::layer::Layer;
use crate::node::Node;
use crate::utils;
use rand::Rng;

struct Network {
    layers: Vec<Layer>,
    activation: Box<dyn Fn(f64) -> f64>,
    loss: Box<dyn Fn(Vec<f64>, Vec<f64>) -> f64>,
}

impl Network {
    pub fn new(shape: Vec<u64>) -> Network {
        // shape is number of nodes in each layer
        let mut rng = rand::thread_rng();

        let layers = vec![];
        let mut last_length = 0;
        for l in shape {
            let mut nodes = vec![];
            for i in 0..l {
                let n = Node {
                    weights: (0..last_length).map(|i| rng.gen_range(0.0..1.0)).collect(),
                    bias: rng.gen_range(0.0..1.0),
                    value: 0,
                };
                nodes.push(n);
            }
            last_length = l;
        }

        Network {
            layers: layers,
            activation: Box::new(utils::sigmoid),
            loss: Box::new(utils::mse),
        }
    }

    pub fn evaluate(&self, input: Vec<f64>) -> Vec<f64> {
        // self.layers[0].set_values(input);

        let mut values: Vec<f64> = input;

        for layer in self.layers.iter().skip(1) {
            values = layer.evaluate(values, &self.activation);
        }

        values
    }

    fn train(&mut self, training_data: Vec<f64>, correct_output: Vec<f64>) {}

    fn train_epochs(&mut self, training_data: Vec<f64>, correct_output: Vec<f64>) {
        unimplemented!();
    }

    fn set_loss_function(&mut self, loss: impl Fn(i64) -> i64) {
        unimplemented!();
    }

    pub fn set_optimizer_function() {
        unimplemented!();
    }
}
