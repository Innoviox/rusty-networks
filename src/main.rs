use rustynetworks::convolution::Transform::*;
use rustynetworks::network::Network;
use rustynetworks::optimizers::{Adam, GradDescent};
use rustynetworks::utils::{
    argmax, categorical_cross_entropy, progress_bar, relu, sigmoid, softmax,
};
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

    for _ in progress_bar(0..n, n as u64, "MNIST:") {
        im_f.read(&mut img_buffer).ok();
        lab_f.read(&mut label_buffer).ok();

        let mut image = vec![];
        for i in img_buffer {
            image.push(i as f64 / 255.0);
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
            10000,
        ),
        _read_mnist(
            "mnist/t10k-images-idx3-ubyte",
            "mnist/t10k-labels-idx1-ubyte",
            10000,
        ),
    )
}

fn main() {
    // let v1 = vec![0.01, 0.19, 0.03, 0.08, 0.14, 0.27, 0.03, 0.11, 0.12, 0.01];
    // println!("{:?}", softmax(v1));
    // println!("{:?}", softmax(vec![8.0, 5.0, 0.0]));

    test_mnist();
}

fn test_mnist() {
    let n = 10000;
    // let kernel = vec![
    //     vec![-1.0, -1.0, -1.0],
    //     vec![-1.0, 8.0, -1.0],
    //     vec![-1.0, -1.0, -1.0],
    // ];
    let kernel = vec![
        vec![1.0, 0.0, 1.0],
        vec![0.0, 1.0, 0.0],
        vec![1.0, 0.0, 1.0],
    ];
    // let kernel = vec![
    //     vec![0.0, -1.0, 0.0],
    //     vec![-1.0, 5.0, -1.0],
    //     vec![0.0, -1.0, 0.0],
    // ];

    let mut network = Network::default();
    // let mut network = Network::from_file("50acc.rn");

    network
        .add_layer(625, &sigmoid)
        .add_layer(30, &sigmoid)
        .add_layer(10, &softmax)
        .loss(&categorical_cross_entropy)
        // .optimizer(Adam::new(&vec![625, 50, 10], 0.01, 0.9, 0.99))
        .optimizer(GradDescent::new())
        .add_transform(Convolve2D(kernel.clone(), 28))
        .add_transform(MaxPool((2, 2)))
        // .add_transform(Convolve2D(kernel.clone(), 25))
        // .add_transform(MaxPool((2, 2)))
        // .add_transform(Convolve2D(kernel.clone(), 22))
        // .add_transform(MaxPool((2, 2)))
        .add_transform(Flatten());

    let ((train_img, train_label), (test_img, test_labels)) = read_mnist();

    network.train_epochs(&train_img, &train_label, 10);
    network.save("625-30-10-grad-1conv-xkern.rn");

    let mut correct = 0.0;
    for idx in 0..n {
        let eval = &network.predict(&test_img[idx]);
        if argmax(&test_labels[idx]) == argmax(eval) {
            correct += 1.0;

            // for j in 0..28 {
            //     for k in 0..28 {
            //         print!(
            //             "{}",
            //             if test_img[idx][j * 28 + k] > 0.5 {
            //                 1
            //             } else {
            //                 0
            //             }
            //         );
            //     }
            //     if j < 10 {
            //         let n = (100.0 * eval[j]) as usize;

            //         print!(" {:?} {}", j, n);
            //     }
            //     println!();
            // }
        }

        // println!(
        //     "{:?} {:?}",
        //     &network.evaluate(&network.apply_transforms(&test_img[idx])),
        //     argmax(&network.evaluate(&network.apply_transforms(&test_img[idx])))
        // );
    }

    println!("Accuracy: {}", correct / n as f64);
}

// fn _main() {
//     let mut network = Network::new(vec![3, 50, 3]);

//     let _rng = rand::thread_rng();
//     let mut training_input = vec![];
//     let mut training_output = vec![];

//     for l in vec![
//         vec![0, 0, 0],
//         vec![0, 0, 1],
//         vec![0, 1, 0],
//         vec![0, 1, 1],
//         vec![1, 0, 0],
//         vec![1, 0, 1],
//         vec![1, 1, 0],
//         vec![1, 1, 1],
//     ] {
//         let k: Vec<f64> = l.iter().map(|i| *i as f64).collect();
//         training_input.push(k.clone());
//         training_output.push(vec![k[2], k[1], k[0]]);
//     }

//     network.train_epochs(&training_input, &training_output, 100);

//     // let mut testing_input = vec![];
//     // let mut testing_output = vec![];

//     for l in vec![
//         vec![0, 0, 0],
//         vec![0, 0, 1],
//         vec![0, 1, 0],
//         vec![0, 1, 1],
//         vec![1, 0, 0],
//         vec![1, 0, 1],
//         vec![1, 1, 0],
//         vec![1, 1, 1],
//     ] {
//         let k: Vec<f64> = l.iter().map(|i| *i as f64).collect();
//         println!("input: {:?} evaluation: {:?}", l, network.evaluate(&k),);
//     }
// }
//
