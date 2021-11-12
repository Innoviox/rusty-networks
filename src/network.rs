struct Network {
    layers: Vec<Layer>,
}

impl Network {
    pub fn new(shape: Vec<u64>) -> Network {
        unimplemented!();
    }

    pub fn evaluate(&self, input: Vec<f64>) -> i64 {
        // self.layers[0].set_values(input);

        let mut values: Vec<f64> = input;

        for layer in self.layers.iter().skip(1) {
            values = layer.evaluate(values);
        }

        0
    }

    fn train(&mut self, training_data: Vec<f64>, correct_output: Vec<f64>) {
        unimplemented!();
    }

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
