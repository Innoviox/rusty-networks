struct Layer {
    nodes: Vec<Node>,
}

impl Layer {
    fn backprop() {
        unimplemented!();
    }

    fn evaluate(&self, input: Vec<f64>) -> Vec<f64> {
        self.nodes.map(|node| node.evaluate(input)).collect()
    }
}
