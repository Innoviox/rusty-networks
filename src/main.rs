use rustynetworks::convolution::Transform::*;
use rustynetworks::network::Network;
use rustynetworks::optimizers::{Optimizer, Adam, GradDescent};
use rustynetworks::utils::{
    argmax, categorical_cross_entropy, progress_bar, relu, sigmoid, softmax, mse,
};
use std::fs::File;
use std::io::prelude::*;
use clap::{arg, App, Arg, ArgMatches, ArgGroup};

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
            5000,
        ),
        _read_mnist(
            "mnist/t10k-images-idx3-ubyte",
            "mnist/t10k-labels-idx1-ubyte",
            1000,
        ),
    )
}

fn _main() {
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
    // ]; // edges

    let kernel = vec![
        vec![1.0, 0.0, 1.0],
        vec![0.0, 1.0, 0.0],
        vec![1.0, 0.0, 1.0],
    ]; // xkern
       // let kernel = vec![
       //     vec![0.0, -1.0, 0.0],
       //     vec![-1.0, 5.0, -1.0],
       //     vec![0.0, -1.0, 0.0],
       // ]; // blur

    let mut network = Network::default();
    // let mut network = Network::from_file("neurals/86acc.rn");

    network
        .add_layer(625, &sigmoid)
        .add_layer(40, &sigmoid)
        .add_layer(10, &softmax)
        .loss(&categorical_cross_entropy)
        // .optimizer(Adam::new(0.01, 0.9, 0.99))
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
    network.save("neurals/40-5000-10.rn");

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

fn read_data(dir: String) -> (
    (Vec<Vec<f64>>, Vec<Vec<f64>>),
    (Vec<Vec<f64>>, Vec<Vec<f64>>),
) {
    unimplemented!();
}

fn handle_matches(args: ArgMatches) -> Result<f64, String> {

    let mut network;
    let mut shape = vec![];
    if args.is_present("network") {
	network = Network::from_file(args.value_of("network").unwrap());
    } else {
	network = Network::default();
	let mut layers = args.values_of("layers").unwrap();
	loop {
	    let nodes = match layers.next() {
		Some(n) => n.parse::<u64>().expect("Layer size must be a positive integer"),
		None => break,
	    };
	    shape.push(nodes);
	    match layers.next().unwrap() {
		"softmax" => network.add_layer(nodes, &softmax),
		"sigmoid" => network.add_layer(nodes, &sigmoid),
		"relu" => network.add_layer(nodes + 1, &relu),
		_ => return Err("Unknown activation function. Use 'sigmoid', 'relu', or 'softmax'.".to_string()),
	    };
	};
    };

    if args.is_present("optimizer-adam") {
	let mut opt: Box<dyn Optimizer> = Adam::new(0.02, 0.9, 0.99);
	opt.with_shape(&shape.clone());
	network.optimizer(opt);
    };

    if args.is_present("loss-mse") {
	network.loss(&mse);
    };
    
    let epochs = match args.value_of("train") {
	Some(e) => match e.parse::<usize>() {
	    Ok(e2) => e2,
	    Err(_) => return Err("Provide a positive number for the number of epochs.".to_string())
	},
	None => return Err("Specify the train-epochs parameter.".to_string())
    };

    let ((train_data, train_labels), (test_data, test_labels)) = read_mnist();

    let kernel = vec![
        vec![1.0, 0.0, 1.0],
        vec![0.0, 1.0, 0.0],
        vec![1.0, 0.0, 1.0],
    ];

    network.add_transform(Convolve2D(kernel.clone(), 28))
	.add_transform(MaxPool((2, 2)))
	.add_transform(Convolve2D(kernel.clone(), 25))
	.add_transform(MaxPool((2, 2)))
	.add_transform(Flatten());
    
    network.train_epochs(&train_data, &train_labels, epochs);

    if args.is_present("save") {
	network.save(args.value_of("save").unwrap());
    };

    let mut correct = 0.0;
    for idx in 0..test_data.len() {
	if argmax(&test_labels[idx]) == argmax(&network.predict(&test_data[idx])) {
	    correct += 1.0;
	}
    }
    Ok(correct / (test_data.len() as f64))
}

    

fn main() {
    let matches = App::new("Rusty Networks")
	.version("0.1.0")
	.author("Evan Guenterberg <eguenter@umd.edu>")
	.author("Simon Chervenak <simonlcherv@gmail.com")
	.about("Fully connected neural networks")
	.arg(
	    arg!(-l --layers "Layers of the network to create.")
		.takes_value(true)
		.required(true)
		.multiple_values(true),
	)
	.arg(
	    arg!(-n --network "File of a previously trained network.")
		.takes_value(true)
		.required(false)
		.multiple_values(false)
	)
	.group(
	    ArgGroup::new("creation")
		.arg("layers")
		.arg("network")
		.required(true)
	)
	.arg(
	    arg!(--"optimizer-adam" "Use the Adam optimizer")
		.name("optimizer-adam")
		.takes_value(false)
		.required(false)
	)
	.arg(
	    arg!(--"optimizer-gd" "Use the gradient descent optimizer")
		.name("optimizer-gd")
		.takes_value(false)
		.required(false)
	)
	.group(
	    ArgGroup::new("optimizer")
		.arg("optimizer-adam")
		.arg("optimizer-gd")
		.required(true)
		.multiple(false)

	)
	.arg(
	    arg!(--"loss-mse" "Use the Mean Squared Error loss function")
		.name("loss-mse")
		.takes_value(false)
	)
	.arg(
	    arg!(--"loss-cce" "Use the Categorical Cross Entropy loss function")
		.name("loss-cce")
		.takes_value(false)
	)
	.group(
	    ArgGroup::new("loss")
		.arg("loss-mse")
		.arg("loss-cce")
		.required(true)
	)
	.arg(
	    arg!(--save "File to save the network in after training.")
		.name("save")
		.takes_value(true)
		.required(false)
		.multiple_values(false)
	)
	.arg(
	    arg!(-t --"train-epochs" "How many times to train the network on the data.")
		.name("train")
		.takes_value(true)
		.required(true)
		.multiple_values(false)
	)

	.get_matches();

    match handle_matches(matches) {
	Err(err) => panic!("{}", err),
	Ok(acc) => println!("Accuracy: {}", acc)
    };
}
