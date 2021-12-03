pub fn convolve(matrix: Vec<Vec<f64>>, kernel: Vec<Vec<f64>>) -> Vec<Vec<f64>> {
    let stride = 1;

    let mut result = vec![];

    let a = kernel.len();
    let b = kernel[0].len();
    let w = matrix.len() - (a - 1);
    let h = matrix[0].len() - (b - 1);

    for i in 0..w {
        let mut row = vec![];
        for j in 0..h {
            let mut n = 0.0;
            for c in 0..a {
                for d in 0..b {
                    n += matrix[i + c][j + d] * kernel[c][d];
                }
            }

            row.push(n);
        }
        result.push(row);
    }

    result
}
