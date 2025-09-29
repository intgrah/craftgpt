use crate::matmul::MatMul;
use crate::{Fixed24, EMBED_SIZE};
use std::fs::File;
use std::io::Read;

const MLP_SCALE: usize = 4;
const HIDDEN_SIZE: usize = EMBED_SIZE * MLP_SCALE;

pub struct MLP {
    matmul_up: MatMul<EMBED_SIZE, HIDDEN_SIZE>,
    matmul_down: MatMul<HIDDEN_SIZE, EMBED_SIZE>,
}

impl MLP {
    pub fn new(block_num: usize) -> Self {
        let mut weights_up = vec![vec![0u8; EMBED_SIZE]; HIDDEN_SIZE];
        let mut weights_down = vec![vec![0u8; HIDDEN_SIZE]; EMBED_SIZE];

        for i in 0..24 {
            let path = format!(
                "weights/weight_files/mlp/mlp_{}.bin",
                25 + 48 * block_num + i
            );
            let mut file = File::open(path).expect("couldn't read file");
            let mut cur_weights = vec![0u8; 9600];
            file.read_exact(&mut cur_weights)
                .expect("couldn't read into buffer");

            for j in 0..5 {
                if i % 2 == 0 {
                    for k in 0..HIDDEN_SIZE {
                        weights_down[48 * j + 2 * i][k] = cur_weights[HIDDEN_SIZE * j + k];
                    }
                    for k in 0..HIDDEN_SIZE {
                        weights_down[48 * j + 2 * i + 1][k] =
                            cur_weights[HIDDEN_SIZE * (j + 5) + k];
                    }
                } else {
                    for k in 0..HIDDEN_SIZE {
                        weights_down[48 * j + 2 * i + 1][k] = cur_weights[HIDDEN_SIZE * j + k];
                    }
                    for k in 0..HIDDEN_SIZE {
                        weights_down[48 * j + 2 * i][k] = cur_weights[HIDDEN_SIZE * (j + 5) + k];
                    }
                }
            }
        }

        for i in 0..24 {
            let path = format!(
                "weights/weight_files/mlp/mlp_{}.bin",
                1 + 48 * block_num + i
            );
            let mut file = File::open(path).expect("couldn't read file");
            let mut cur_weights = vec![0u8; 9600];
            file.read_exact(&mut cur_weights)
                .expect("couldn't read into buffer");

            for j in 0..20 {
                if i % 2 == 0 {
                    for k in 0..EMBED_SIZE {
                        weights_up[48 * j + 2 * i][k] = cur_weights[EMBED_SIZE * j + k];
                    }
                    for k in 0..EMBED_SIZE {
                        weights_up[48 * j + 2 * i + 1][k] = cur_weights[EMBED_SIZE * (j + 20) + k];
                    }
                } else {
                    for k in 0..EMBED_SIZE {
                        weights_up[48 * j + 2 * i + 1][k] = cur_weights[EMBED_SIZE * j + k];
                    }
                    for k in 0..EMBED_SIZE {
                        weights_up[48 * j + 2 * i][k] = cur_weights[EMBED_SIZE * (j + 20) + k];
                    }
                }
            }
        }

        let mut weights_up_array: [[u8; EMBED_SIZE]; HIDDEN_SIZE] = [[0; EMBED_SIZE]; HIDDEN_SIZE];
        let mut weights_down_array: [[u8; HIDDEN_SIZE]; EMBED_SIZE] =
            [[0; HIDDEN_SIZE]; EMBED_SIZE];

        for i in 0..HIDDEN_SIZE {
            for j in 0..EMBED_SIZE {
                weights_up_array[i][j] = weights_up[i][j];
            }
        }

        for i in 0..EMBED_SIZE {
            for j in 0..HIDDEN_SIZE {
                weights_down_array[i][j] = weights_down[i][j];
            }
        }

        let matmul_up = MatMul::new(&weights_up_array, true);
        let matmul_down = MatMul::new(&weights_down_array, false);

        MLP {
            matmul_up,
            matmul_down,
        }
    }

    pub fn forward(&self, input: &[Fixed24; EMBED_SIZE]) -> [Fixed24; EMBED_SIZE] {
        let res = self.matmul_up.forward(input);
        self.matmul_down.forward(&res)
    }
}
