use crate::matmul::MATMUL_FIXED_POINT;
use crate::{EMBED_SIZE, FIXED_POINT_MASK, FIXED_POINT_SIZE, Fixed24};
use std::fs::File;
use std::io::Read;

const LAYERNORM_CONST: u64 = (1u64 << 32) / (EMBED_SIZE as u64);
const LAYERNORM_CONST_2: u64 = 8663717; // int((1 << 27) / sqrt(EMBED_SIZE))

const EPS: u64 = 164926744; // int(1e-5 * EMBED_SIZE * (1 << (2 * MATMUL_FIXED_POINT)))

pub struct LayerNorm {
    weights: Box<[u32; EMBED_SIZE]>,
}

impl LayerNorm {
    pub fn new(index: usize) -> Self {
        let path = format!("weights/weight_files/layernorm/ln_{}.bin", index);
        let mut file = File::open(path).expect("couldn't read file");
        let mut weights = [0u32; EMBED_SIZE];

        for i in 0..EMBED_SIZE {
            let mut buf = [0u8; 3];
            file.read_exact(&mut buf)
                .expect("couldn't read into buffer");
            let cur = u32::from_le_bytes([buf[0], buf[1], buf[2], 0]);
            weights[i] = cur / 2;
        }

        LayerNorm {
            weights: Box::new(weights),
        }
    }

    pub fn forward(&self, input: &[Fixed24; EMBED_SIZE]) -> [Fixed24; EMBED_SIZE] {
        let mut mean: u32 = 0;
        for &v in input {
            let adjusted = if v > FIXED_POINT_MASK / 2 {
                v.wrapping_add(0xff << FIXED_POINT_SIZE)
            } else {
                v
            };
            mean = mean.wrapping_add(adjusted);
        }
        mean &= u32::MAX;

        let neg = mean >= (1 << (FIXED_POINT_SIZE + 7));
        if neg {
            mean = mean.wrapping_neg() & ((1 << (FIXED_POINT_SIZE + 7)) - 1);
        }
        mean = ((mean as u64 * LAYERNORM_CONST as u64) >> 32) as u32;
        if neg {
            mean = mean.wrapping_neg() & FIXED_POINT_MASK;
        }

        let mut sigma2: u64 = EPS;
        for &v in input {
            let diff_i32 = v as i32 - mean as i32;
            let mut diff = if diff_i32 < 0 {
                (diff_i32 + (1i32 << FIXED_POINT_SIZE)) as u32
            } else {
                diff_i32 as u32
            };
            diff %= 1 << FIXED_POINT_SIZE;
            if diff > FIXED_POINT_MASK / 2 {
                diff = diff.wrapping_neg() & (FIXED_POINT_MASK / 2);
            }
            sigma2 = sigma2.wrapping_add((diff as u64) * (diff as u64));
            sigma2 &= (1u64 << 48) - 1;
        }

        let sigma2_sqrt = (sigma2 as f64).sqrt() as u64;
        let mut sigma2_final =
            ((LAYERNORM_CONST_2 as u64 * sigma2_sqrt) >> 27) & FIXED_POINT_MASK as u64;
        sigma2_final =
            ((1u64 << (2 * MATMUL_FIXED_POINT)) / sigma2_final) & FIXED_POINT_MASK as u64;
        let sigma2_final = sigma2_final as u32;

        let mut result = [0u32; EMBED_SIZE];
        for (i, &v) in input.iter().enumerate() {
            let diff_i32 = v as i32 - mean as i32;
            let mut diff = if diff_i32 < 0 {
                (diff_i32 + (1i32 << FIXED_POINT_SIZE)) as u32
            } else {
                diff_i32 as u32
            };
            diff %= 1 << FIXED_POINT_SIZE;
            let neg = diff >= FIXED_POINT_MASK / 2;
            if neg {
                diff = diff.wrapping_neg() & (FIXED_POINT_MASK / 2);
            }
            let mut res = ((diff as u64 * sigma2_final as u64) >> MATMUL_FIXED_POINT)
                & (FIXED_POINT_MASK / 2) as u64;
            res = ((res * self.weights[i] as u64) >> (MATMUL_FIXED_POINT + 3))
                & (FIXED_POINT_MASK / 2) as u64;
            let mut res = res as u32;
            if neg {
                res = res.wrapping_neg() & FIXED_POINT_MASK;
            }
            result[i] = res;
        }

        result
    }
}
