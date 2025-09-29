use std::fs::File;
use std::io::Read;

use crate::matmul::MatMul;
use crate::{EMBED_SIZE, FIXED_POINT_MASK, FIXED_POINT_SIZE, Fixed24};

const OUTPUT_SIZE: usize = 8;
pub const VOCAB_SIZE: usize = 1920;

pub struct Unembedding {
    lm_head: MatMul<EMBED_SIZE, VOCAB_SIZE>,
    softmax_exp: [u32; 1024],
}

impl Unembedding {
    pub fn new() -> Self {
        let mut weights = vec![vec![]; VOCAB_SIZE];

        for i in 0..48 {
            let path = format!("weights/weight_files/unembedding/lm_head_{}.bin", i + 1);
            let mut file = File::open(path).expect("couldn't read file");
            let mut cur_weights = vec![0u8; 9600];
            file.read_exact(&mut cur_weights)
                .expect("couldn't read into buffer");

            for j in 0..20 {
                if i % 2 == 0 {
                    weights[48 * j + 2 * (i % 24) + 960 * (i / 24)] =
                        cur_weights[EMBED_SIZE * j..EMBED_SIZE * (j + 1)].to_vec();
                    weights[48 * j + 2 * (i % 24) + 960 * (i / 24) + 1] =
                        cur_weights[EMBED_SIZE * (j + 20)..EMBED_SIZE * (j + 21)].to_vec();
                } else {
                    weights[48 * j + 2 * (i % 24) + 960 * (i / 24) + 1] =
                        cur_weights[EMBED_SIZE * j..EMBED_SIZE * (j + 1)].to_vec();
                    weights[48 * j + 2 * (i % 24) + 960 * (i / 24)] =
                        cur_weights[EMBED_SIZE * (j + 20)..EMBED_SIZE * (j + 21)].to_vec();
                }
            }
        }

        let mut weights_array: [[u8; EMBED_SIZE]; VOCAB_SIZE] = [[0; EMBED_SIZE]; VOCAB_SIZE];
        for i in 0..VOCAB_SIZE {
            for j in 0..EMBED_SIZE {
                weights_array[i][j] = weights[i][j];
            }
        }

        let lm_head = MatMul::new(&weights_array, false);

        let mut softmax_exp = [0u32; 1024];
        let mut file =
            File::open("weights/weight_files/softmax_2.bin").expect("couldn't read file");
        for i in 0..1024 {
            let mut buf = [0u8; 3];
            file.read_exact(&mut buf)
                .expect("couldn't read into buffer");
            softmax_exp[i] = u32::from_le_bytes([buf[0], buf[1], buf[2], 0]);
        }

        Unembedding {
            lm_head,
            softmax_exp,
        }
    }

    pub fn forward(&self, input: &[Fixed24; EMBED_SIZE]) -> Vec<u64> {
        let mut logits = self.lm_head.forward(input);

        let mut biggest = 0u32;
        for i in 0..VOCAB_SIZE {
            logits[i] ^= 1 << (FIXED_POINT_SIZE - 1);
            biggest = biggest.max(logits[i]);
        }

        let mut softmax_sum: u32 = 0;
        for i in 0..VOCAB_SIZE {
            let power = (biggest - logits[i]) >> 12;
            let res = if power >= 1024 {
                0
            } else {
                self.softmax_exp[power as usize]
            };
            softmax_sum = softmax_sum.wrapping_add(res);
        }
        let softmax_sum = (1u64 << 46) / softmax_sum as u64;

        let mut output = vec![0u64; OUTPUT_SIZE];
        for i in 0..VOCAB_SIZE {
            let power = (biggest - logits[i]) >> 12;
            let res = if power >= 1024 {
                0
            } else {
                self.softmax_exp[power as usize]
            };
            let mut res = ((softmax_sum * res as u64) >> 23) & FIXED_POINT_MASK as u64;
            res = (1 << 11) * res + i as u64;

            for j in 0..OUTPUT_SIZE {
                if res > output[j] {
                    let tmp = output[j];
                    output[j] = res;
                    res = tmp;
                }
            }
        }

        output
    }
}
