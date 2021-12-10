pub trait Optimizer {
    fn optimize(&mut self, gradient: &[Vec<Vec<f64>>]) -> Vec<Vec<Vec<f64>>>;
    fn with_shape(&mut self, shape: &[u64]);
}

#[derive(Clone)]
pub struct GradDescent {}

impl GradDescent {
    pub fn new() -> Box<GradDescent> {
        Box::new(GradDescent {})
    }
}

impl Optimizer for GradDescent {
    fn optimize(&mut self, gradient: &[Vec<Vec<f64>>]) -> Vec<Vec<Vec<f64>>> {
        let mut dg = Vec::new();
        for i in 0..gradient.len() {
            dg.push(Vec::new());
            for j in 0..gradient[i].len() {
                dg[i].push(Vec::new());
                for k in 0..gradient[i][j].len() {
                    dg[i][j].push(-gradient[i][j][k]);
                }
            }
        }
        dg
    }

    fn with_shape(&mut self, _shape: &[u64]) {}
}

#[derive(Debug, Clone)]
pub struct Adam {
    alpha: f64,
    beta1: f64,
    beta2: f64,

    start_beta1: f64,
    start_beta2: f64,

    old_m: Vec<Vec<Vec<f64>>>,
    old_v: Vec<Vec<Vec<f64>>>,
}

impl Adam {
    fn beta1(&mut self) -> f64 {
        self.beta1 *= self.start_beta1;
        self.beta1
    }

    fn beta2(&mut self) -> f64 {
        self.beta2 *= self.start_beta2;
        self.beta2
    }

    fn alpha(&self) -> f64 {
        self.alpha * (1f64 - self.beta2).sqrt() / (1f64 - self.beta1)
    }

    pub fn new(alpha: f64, start_beta1: f64, start_beta2: f64) -> Box<Adam> {
        let old_m = Vec::new();
        let old_v = Vec::new();

        let a = Adam {
            alpha,
            beta1: 0.0,
            beta2: 0.0,
            start_beta1,
            start_beta2,
            old_m,
            old_v,
        };

        Box::new(a)
    }
}

impl Optimizer for Adam {
    fn with_shape(&mut self, shape: &[u64]) {
        self.old_m = Vec::new();
        self.old_v = Vec::new();

        for (i, layer_n) in shape.iter().skip(1).enumerate() {
            self.old_m.push(Vec::new());
            self.old_v.push(Vec::new());
            for node_n in 0..*layer_n as usize {
                self.old_m[i].push(Vec::new());
                self.old_v[i].push(Vec::new());
                for _weight_n in 0..shape[i] + 1 {
                    self.old_m[i][node_n as usize].push(0.0);
                    self.old_v[i][node_n as usize].push(0.0);
                }
            }
        }

        for layer_n in 0..self.old_m.len() {
            for node_n in 0..self.old_m[layer_n].len() {
                for weight_n in 0..self.old_m[layer_n][node_n].len() {
                    self.old_m[layer_n][node_n][weight_n] = 0.0;
                    self.old_v[layer_n][node_n][weight_n] = 0.0;
                }
            }
        }
    }

    fn optimize(&mut self, gradient: &[Vec<Vec<f64>>]) -> Vec<Vec<Vec<f64>>> {
        let b1 = self.beta1();
        let b2 = self.beta2();
        let a = self.alpha();

        let mut m = Vec::new();
        let mut v = Vec::new();
        let mut w = Vec::new();

        let mut temp_m: f64;
        let mut temp_v: f64;
        let mut mhat: f64;
        let mut vhat: f64;

        for layer_n in 0..gradient.len() {
            m.push(Vec::new());
            v.push(Vec::new());
            w.push(Vec::new());
            for node_n in 0..gradient[layer_n].len() {
                m[layer_n].push(Vec::new());
                v[layer_n].push(Vec::new());
                w[layer_n].push(Vec::new());
                for weight_n in 0..gradient[layer_n][node_n].len() {
                    temp_m = b1 * self.old_m[layer_n][node_n][weight_n]
                        + (1f64 - b1) * gradient[layer_n][node_n][weight_n];
                    temp_v = b2 * self.old_v[layer_n][node_n][weight_n]
                        + (1f64 - b2) * gradient[layer_n][node_n][weight_n].powi(2);
                    mhat = temp_m / (1f64 - b1);
                    vhat = temp_v / (1f64 - b2);
                    m[layer_n][node_n].push(temp_m);
                    v[layer_n][node_n].push(temp_v);
                    w[layer_n][node_n].push(-a * mhat / (vhat.sqrt() + 0.00000001));
                }
            }
        }

        self.old_m = m;
        self.old_v = v;

        w
    }
}
