fn _filter_fn<F>(matrix: &Vec<Vec<f64>>, dim: (usize, usize), f: F) -> Vec<Vec<f64>>
where
    F: Fn(usize, usize, usize, usize, f64) -> f64,
{
    let stride = 1; // todo

    let mut result = vec![];

    let w = matrix.len() - (dim.0 - 1);
    let h = matrix[0].len() - (dim.1 - 1);

    for i in 0..w {
        let mut row = vec![];
        for j in 0..h {
            let mut n = 0.0;
            for c in 0..dim.0 {
                for d in 0..dim.1 {
                    n = f(i, j, c, d, n);
                }
            }
            row.push(n);
        }
        result.push(row);
    }

    result
}

pub fn convolve(matrix: Vec<Vec<f64>>, kernel: Vec<Vec<f64>>) -> Vec<Vec<f64>> {
    _filter_fn(&matrix, (kernel.len(), kernel[0].len()), |i, j, c, d, n| {
        n + matrix[i + c][j + d] * kernel[c][d]
    })
}

pub fn max_pool(matrix: Vec<Vec<f64>>, filter_size: (usize, usize)) -> Vec<Vec<f64>> {
    _filter_fn(&matrix, filter_size, |i, j, c, d, n| {
        f64::max(n, matrix[i + c][j + d])
    })
}
