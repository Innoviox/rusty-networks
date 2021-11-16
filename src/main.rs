use rand::Rng;
use rustynetworks::network::Network;

fn main() {
    let mut network = Network::new(vec![3, 5, 3]);

    let mut rng = rand::thread_rng();
    let mut training_input = vec![];
    let mut training_output = vec![];

    for l in vec![
        vec![0, 0, 0],
        vec![0, 0, 1],
        vec![0, 1, 0],
        vec![0, 1, 1],
        vec![1, 0, 0],
        vec![1, 0, 1],
        vec![1, 1, 0],
        vec![1, 1, 1],
    ] {
        let k: Vec<f64> = l.iter().map(|i| *i as f64).collect();
        training_input.push(k.clone());
        training_output.push(vec![k[2], k[1], k[0]]);
    }

    network.train_epochs(&training_input, &training_output, 100);

    // let mut testing_input = vec![];
    // let mut testing_output = vec![];

    for l in vec![
        vec![0, 0, 0],
        vec![0, 0, 1],
        vec![0, 1, 0],
        vec![0, 1, 1],
        vec![1, 0, 0],
        vec![1, 0, 1],
        vec![1, 1, 0],
        vec![1, 1, 1],
    ] {
        let k: Vec<f64> = l.iter().map(|i| *i as f64).collect();
        println!("input: {:?} evaluation: {:?}", l, network.evaluate(&k),);
    }
}
