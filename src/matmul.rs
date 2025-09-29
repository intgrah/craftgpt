use crate::{FIXED_POINT_MASK, FIXED_POINT_SIZE, Fixed24};

pub const MATMUL_FIXED_POINT: u32 = 18;
const MATMUL_EXTRA_PRECISION: u32 = 4;
const MATMUL_BIG_MASK: u32 = (1 << (FIXED_POINT_SIZE + MATMUL_EXTRA_PRECISION)) - 1;

pub struct MatMul<const INPUT_SIZE: usize, const OUTPUT_SIZE: usize> {
    weights: Box<[[(bool, u32, u32, u32); INPUT_SIZE]; OUTPUT_SIZE]>,
    relu: bool,
}

impl<const INPUT_SIZE: usize, const OUTPUT_SIZE: usize> MatMul<INPUT_SIZE, OUTPUT_SIZE> {
    pub fn new(weights: &[[u8; INPUT_SIZE]; OUTPUT_SIZE], relu: bool) -> Self {
        let mut parsed_weights: [[(bool, u32, u32, u32); INPUT_SIZE]; OUTPUT_SIZE] =
            [[(false, 0, 0, 0); INPUT_SIZE]; OUTPUT_SIZE];

        for (i, &row) in weights.iter().enumerate() {
            for (j, &w) in row.iter().enumerate() {
                let neg = w >= 128;
                let w = w % 128;

                let parsed = if w < 64 {
                    (neg, 8u32, (w / 8) as u32, (w % 8) as u32)
                } else if w < 96 {
                    let w = w - 64;
                    (neg, 7u32, (4 + (w / 8)) as u32, (w % 8) as u32)
                } else if w < 112 {
                    let w = w - 96;
                    (neg, 5u32, (2 + (w / 8)) as u32, (w % 8) as u32)
                } else if w < 120 {
                    let w = w - 112;
                    (neg, 3u32, (1 + (w / 8)) as u32, (w % 8) as u32)
                } else {
                    let w = w - 120;
                    (neg, 2u32, (1 + (w / 8)) as u32, (w % 8) as u32)
                };
                parsed_weights[i][j] = parsed;
            }
        }

        Self {
            weights: Box::new(parsed_weights),
            relu,
        }
    }

    pub fn forward(&self, input: &[Fixed24; INPUT_SIZE]) -> [Fixed24; OUTPUT_SIZE] {
        let mut output = [0u32; OUTPUT_SIZE];
        let mut normed: [u32; INPUT_SIZE] = [0; INPUT_SIZE];

        for (i, &x) in input.iter().enumerate() {
            let mut val = x & FIXED_POINT_MASK;
            if val > FIXED_POINT_MASK / 2 {
                val = val.wrapping_add(((1 << MATMUL_EXTRA_PRECISION) - 1) << FIXED_POINT_SIZE);
            }
            normed[i] = val;
        }

        for i in 0..OUTPUT_SIZE {
            let mut cur: u32 = 0;

            for j in 0..INPUT_SIZE {
                let w = &self.weights[i][j];

                let mut big = (normed[j] as u64 * w.2 as u64) & MATMUL_BIG_MASK as u64;
                if big > (MATMUL_BIG_MASK / 2) as u64 {
                    big = big + ((255u64) << (MATMUL_EXTRA_PRECISION + FIXED_POINT_SIZE));
                }

                let mut small = (normed[j] as u64 * w.3 as u64) & MATMUL_BIG_MASK as u64;
                if small > (MATMUL_BIG_MASK / 2) as u64 {
                    small = small + ((255u64) << (MATMUL_EXTRA_PRECISION + FIXED_POINT_SIZE));
                }

                let mut cont = ((big >> w.1) + (small >> (w.1 + 3))) as u32;
                cont &= FIXED_POINT_MASK;

                if w.0 {
                    cont = cont.wrapping_neg() & FIXED_POINT_MASK;
                }

                cur = cur.wrapping_add(cont) & FIXED_POINT_MASK;
            }

            if self.relu && cur > FIXED_POINT_MASK / 2 {
                output[i] = 0;
            } else {
                output[i] = cur;
            }
        }

        output
    }
}
