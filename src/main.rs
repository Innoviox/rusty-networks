use rand::Rng;
use rustynetworks::convolution::Transform;
use rustynetworks::network::Network;

fn main() {
    let kernel = vec![
        vec![1.0, 0.0, 1.0],
        vec![0.0, 1.0, 0.0],
        vec![1.0, 0.0, 1.0],
    ];

    let mut network = Network::new(vec![784, 100, 10])
        .add_transform(Transform::Convolve2D(kernel, 28))
        .add_transform(Transform::MaxPool((2, 2)))
        .add_transform(Transform::Flatten());
}
