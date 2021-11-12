use ndarray::Array;

struct Network {
    Layers: Vec<Layer>,
}

impl Network {
    pub fn new(shape: Vec<u64>) -> Network {
        unimplemented!();
    }

    pub fn evaluate(&self, input: Array) -> i64 {
        unimplemented!();
    }

    fn train(&mut self, training_data: Array, correct_output: Array);

    fn train_epochs(&mut self, training_data: Array, correct_output: Array);

    fn set_loss_function(&mut self, loss: impl Fn(i64) -> i64);

    pub fn set_optimizer_function() {
        unimplemented!();
    }
}
