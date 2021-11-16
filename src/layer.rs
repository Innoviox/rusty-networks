use crate::node::Node;

pub struct Layer {
    pub nodes: Vec<Node>,
}

impl Layer {
    fn backprop() {
        unimplemented!();
    }

    pub fn evaluate(&self, input: &Vec<f64>, activation: &Box<dyn Fn(f64) -> f64>) -> Vec<f64> {
        self.nodes
            .iter()
            .map(|node| node.evaluate(input, activation))
            .collect()
    }
}
