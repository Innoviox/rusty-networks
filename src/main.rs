use rustynetworks::convolution::Transform::*;
use rustynetworks::network::Network;
use rustynetworks::optimizers::Adam;
use rustynetworks::utils::{argmax, progress_bar, sigmoid};
use std::fs::File;
use std::io::prelude::*;

fn _read_mnist(img_fn: &str, lab_fn: &str, n: usize) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
    let mut images = vec![];
    let mut labels = vec![];

    let mut im_f = File::open(img_fn).ok().unwrap();
    let mut lab_f = File::open(lab_fn).ok().unwrap();

    let mut img_buffer = [0; 784];
    let mut label_buffer = [0; 1];

    im_f.read(&mut [0; 16]).ok();
    lab_f.read(&mut [0; 8]).ok();

    for _ in progress_bar(0..n, n, "MNIST:") {
        im_f.read(&mut img_buffer).ok();
        lab_f.read(&mut label_buffer).ok();

        let mut image = vec![];
        for i in img_buffer {
            image.push(if i > 127 { 1.0 } else { 0.0 });
        }

        let mut label = vec![0.0; 10];
        label[label_buffer[0] as usize] = 1.0;

        images.push(image);
        labels.push(label);
    }

    (images, labels)
}

fn read_mnist() -> (
    (Vec<Vec<f64>>, Vec<Vec<f64>>),
    (Vec<Vec<f64>>, Vec<Vec<f64>>),
) {
    (
        _read_mnist(
            "mnist/train-images-idx3-ubyte",
            "mnist/train-labels-idx1-ubyte",
            60000,
        ),
        _read_mnist(
            "mnist/t10k-images-idx3-ubyte",
            "mnist/t10k-labels-idx1-ubyte",
            10000,
        ),
    )
}

fn main() {
    let kernel = vec![
        vec![1.0, 0.0, 1.0],
        vec![0.0, 1.0, 0.0],
        vec![1.0, 0.0, 1.0],
    ];

    let shape = vec![625, 10];

    let mut network = Network::new(shape.clone());
    // let mut network = Network::from_file("test.rn");

    network
        .activation(&sigmoid)
        .optimizer(Adam::new(&shape.clone(), 0.9, 0.9, 0.99))
        .add_transform(Convolve2D(kernel.clone(), 28))
        .add_transform(MaxPool((2, 2)))
        .add_transform(Flatten());

    let ((train_img, train_label), (test_img, test_labels)) = read_mnist();

    network.train_epochs(&train_img, &train_label, 1);
    network.save("test.rn");

    let mut correct = 0.0;
    for idx in 0..10000 {
        if argmax(&test_labels[idx]) == argmax(&network.evaluate(&test_img[idx])) {
            correct += 1.0;
        }
    }

    println!("Accuracy: {}", correct / 60.0);
}

fn _main() {
    let mut network = Network::new(vec![3, 50, 3]);

    let _rng = rand::thread_rng();
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
