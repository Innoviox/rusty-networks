struct Node {
    weights: <some matrix type>,
    bias: <some matrix type>,
    value: i64,
}

struct Layer {
    nodes: Vec<Node>,
    values: <some matrix type>,
}

struct Network {
    Layers: Vec<Layer>,
}

impl Node {
}

impl Layer {
    fn backprop() {
	unimplemented!();
    }
}


impl Network {
    pub fn new(shape: Vec<u64>) -> Network {
	unimplemented!();
    }

    pub fn evaluate(&self, input: <some matrix type>) -> i64 {
	unimplemented!();
    }

    pub fn train(&mut self, training_data: <some matrix type>, correct_output: <some matrix type>);

    pub fn train_epochs(&mut self, training_data: <some matrix type>, correct_output: <some matrix type>);

    pub fn set_loss_function(&mut self, loss: impl Fn(i64) -> i64);

    pub fn set_optimizer_function() {
	unimplemented!();
    }
}



#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        let result = 2 + 2;
        assert_eq!(result, 4);
    }
}
