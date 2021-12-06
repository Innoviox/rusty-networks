use rustynetworks::convolution::Transform;
use rustynetworks::network::Network;
use std::fs::File;
use std::io::prelude::*;

fn _read_mnist(img_fn: &str, lab_fn: &str) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
    let mut images = vec![];
    let mut labels = vec![];

    let mut im_f = File::open(img_fn).ok().unwrap();
    let mut lab_f = File::open(lab_fn).ok().unwrap();

    let mut img_buffer = [0; 784];
    let mut label_buffer = [0; 1];

    im_f.read(&mut [0; 16]).ok();
    lab_f.read(&mut [0; 8]).ok();

    for _ in 0..60000 {
        im_f.read(&mut img_buffer).ok();
        lab_f.read(&mut label_buffer).ok();

        let mut image = vec![];
        for i in img_buffer {
            image.push(if i > 127 { 0.0 } else { 1.0 });
        }

        let mut label = vec![0.0; 10];
        label[label_buffer[0] as usize] = 1.0;

        images.push(image);
        labels.push(label);
    }

    // let i = &images[49235];
    // for j in 0..28 {
    //     for k in 0..28 {
    //         print!("{}", i[j * 28 + k]);
    //     }
    //     println!();
    // }

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
        ),
        _read_mnist(
            "mnist/t10k-images-idx3-ubyte",
            "mnist/t10k-labels-idx1-ubyte",
        ),
    )
}

fn main() {
    let kernel = vec![
        vec![1.0, 0.0, 1.0],
        vec![0.0, 1.0, 0.0],
        vec![1.0, 0.0, 1.0],
    ];

    let shape = vec![625, 100, 10];

    let mut network = Network::new(shape);

    network
        .add_transform(Transform::Convolve2D(kernel, 28))
        .add_transform(Transform::MaxPool((2, 2)))
        .add_transform(Transform::Flatten());

    let ((train_img, train_label), (_test_img, _test_labels)) = read_mnist();

    network.train_epochs(&train_img, &train_label, 5);
}
