use std::{fs::File, io::Read};

use crate::{matmul::MatMul, Fixed24, EMBED_SIZE, FIXED_POINT_MASK, FIXED_POINT_SIZE};

const HEADS: usize = 5;
const HEAD_SIZE: usize = EMBED_SIZE / HEADS;

const ATT_CONST: u64 = 4331858; // int((1 << 26) / sqrt(EMBED_SIZE))

pub struct Attention {
    matmul_key: [MatMul<EMBED_SIZE, HEAD_SIZE>; HEADS],
    matmul_value: [MatMul<EMBED_SIZE, HEAD_SIZE>; HEADS],
    matmul_query: [MatMul<EMBED_SIZE, HEAD_SIZE>; HEADS],
    matmul_proj: MatMul<EMBED_SIZE, EMBED_SIZE>,
    softmax_exp: [u32; 1024],
    k_cache: [Vec<[u16; HEAD_SIZE]>; HEADS],
    v_cache: [Vec<[u16; HEAD_SIZE]>; HEADS],
}

impl Attention {
    pub fn new(block_num: usize) -> Self {
        let mut key: [[[u8; EMBED_SIZE]; HEAD_SIZE]; HEADS] = [[[0; EMBED_SIZE]; HEAD_SIZE]; HEADS];
        let mut value: [[[u8; EMBED_SIZE]; HEAD_SIZE]; HEADS] =
            [[[0; EMBED_SIZE]; HEAD_SIZE]; HEADS];
        let mut query: [[[u8; EMBED_SIZE]; HEAD_SIZE]; HEADS] =
            [[[0; EMBED_SIZE]; HEAD_SIZE]; HEADS];
        let mut proj: [[u8; EMBED_SIZE]; EMBED_SIZE] = [[0; EMBED_SIZE]; EMBED_SIZE];

        for i in 0..24 {
            let path = format!(
                "weights/weight_files/attention/att_{}.bin",
                1 + 24 * block_num + i
            );
            let mut file = File::open(path).expect("could't read file");
            let mut cur_weights: [u8; 9600] = [0; 9600];
            file.read_exact(&mut cur_weights)
                .expect("couldn't read into buffer");

            for j in 0..HEADS {
                if i % 2 == 0 {
                    for k in 0..EMBED_SIZE {
                        key[j][2 * i][k] = cur_weights[EMBED_SIZE * (3 * j) + k];
                        value[j][2 * i][k] = cur_weights[EMBED_SIZE * (3 * j + 1) + k];
                        query[j][2 * i][k] = cur_weights[EMBED_SIZE * (3 * j + 2) + k];
                        key[j][2 * i + 1][k] = cur_weights[EMBED_SIZE * (3 * j + 20) + k];
                        value[j][2 * i + 1][k] = cur_weights[EMBED_SIZE * (3 * j + 21) + k];
                        query[j][2 * i + 1][k] = cur_weights[EMBED_SIZE * (3 * j + 22) + k];
                    }
                } else {
                    for k in 0..EMBED_SIZE {
                        key[j][2 * i + 1][k] = cur_weights[EMBED_SIZE * (3 * j) + k];
                        value[j][2 * i + 1][k] = cur_weights[EMBED_SIZE * (3 * j + 1) + k];
                        query[j][2 * i + 1][k] = cur_weights[EMBED_SIZE * (3 * j + 2) + k];
                        key[j][2 * i][k] = cur_weights[EMBED_SIZE * (3 * j + 20) + k];
                        value[j][2 * i][k] = cur_weights[EMBED_SIZE * (3 * j + 21) + k];
                        query[j][2 * i][k] = cur_weights[EMBED_SIZE * (3 * j + 22) + k];
                    }
                }
            }

            for j in 0..HEADS {
                if i % 2 == 0 {
                    for k in 0..EMBED_SIZE {
                        proj[HEAD_SIZE * j + 2 * i][k] = cur_weights[EMBED_SIZE * (j + 15) + k];
                        proj[HEAD_SIZE * j + 2 * i + 1][k] = cur_weights[EMBED_SIZE * (j + 35) + k];
                    }
                } else {
                    for k in 0..EMBED_SIZE {
                        proj[HEAD_SIZE * j + 2 * i + 1][k] = cur_weights[EMBED_SIZE * (j + 15) + k];
                        proj[HEAD_SIZE * j + 2 * i][k] = cur_weights[EMBED_SIZE * (j + 35) + k];
                    }
                }
            }
        }

        let matmul_key: [MatMul<EMBED_SIZE, HEAD_SIZE>; HEADS] =
            key.map(|k| MatMul::new(&k, false));
        let matmul_value: [MatMul<EMBED_SIZE, HEAD_SIZE>; HEADS] =
            value.map(|v| MatMul::new(&v, false));
        let matmul_query: [MatMul<EMBED_SIZE, HEAD_SIZE>; HEADS] =
            query.map(|q| MatMul::new(&q, false));
        let matmul_proj = MatMul::new(&proj, false);

        let mut softmax_exp = [0u32; 1024];
        let mut file = File::open("weights/weight_files/softmax.bin").expect("couldn't read file");
        for i in 0..1024 {
            let mut buf = [0u8; 3];
            file.read_exact(&mut buf).expect("could't read into buffer");
            softmax_exp[i] = u32::from_le_bytes([buf[0], buf[1], buf[2], 0]);
        }

        Attention {
            matmul_key,
            matmul_value,
            matmul_query,
            matmul_proj,
            softmax_exp,
            k_cache: std::array::from_fn(|_| Vec::new()),
            v_cache: std::array::from_fn(|_| Vec::new()),
        }
    }

    fn to_float16(&self, value: u32, offset: i32) -> u16 {
        let neg = value > FIXED_POINT_MASK / 2;
        let value = if neg {
            value.wrapping_neg() & (FIXED_POINT_MASK / 2)
        } else {
            value
        };

        for i in (0..FIXED_POINT_SIZE as i32).rev() {
            if ((value >> i) & 1) > 0 {
                let res = ((value << (FIXED_POINT_SIZE as i32 - i)) >> 14) & ((1 << 10) - 1);
                let res = res + (((i + 9 - offset) as u32) << 10);
                let res = res + ((neg as u32) << 15);
                return res as u16;
            }
        }
        0
    }

    pub fn undo_last(&mut self) {
        for i in 0..HEADS {
            self.k_cache[i].pop();
            self.v_cache[i].pop();
        }
    }

    pub fn forward(&mut self, input: &[Fixed24; EMBED_SIZE]) -> [Fixed24; EMBED_SIZE] {
        let mut proj_input = [0u32; EMBED_SIZE];
        let mut proj_offset = 0;

        for head in 0..HEADS {
            let keys = self.matmul_key[head].forward(input);
            let mut keys_array = [0u16; HEAD_SIZE];
            for (i, &k) in keys.iter().enumerate() {
                keys_array[i] = self.to_float16(k, 0);
            }
            self.k_cache[head].push(keys_array);

            let values = self.matmul_value[head].forward(input);
            let mut values_array = [0u16; HEAD_SIZE];
            for (i, &v) in values.iter().enumerate() {
                values_array[i] = self.to_float16(v, 0);
            }
            self.v_cache[head].push(values_array);

            let queries = self.matmul_query[head].forward(input);
            let mut queries_array = [0u16; HEAD_SIZE];
            for (i, &q) in queries.iter().enumerate() {
                queries_array[i] = self.to_float16(q, 0);
            }

            let cache_len = self.k_cache[head].len();
            let mut relevance = [0u32; 1024];
            for (i, v) in self.k_cache[head].iter().enumerate() {
                for (j, &q) in queries_array.iter().enumerate() {
                    relevance[i] = relevance[i].wrapping_add(float_mult(v[j], q, 5));
                    relevance[i] &= FIXED_POINT_MASK;
                }
            }

            let mut biggest = 0u32;
            for i in 0..cache_len {
                let neg = relevance[i] > FIXED_POINT_MASK / 2;
                if neg {
                    relevance[i] = relevance[i].wrapping_neg() & (FIXED_POINT_MASK / 2);
                }
                relevance[i] = ((relevance[i] as u64 * ATT_CONST as u64) >> 23) as u32
                    & (FIXED_POINT_MASK / 2);
                if neg {
                    relevance[i] = relevance[i].wrapping_neg() & FIXED_POINT_MASK;
                }
                relevance[i] ^= 1 << (FIXED_POINT_SIZE - 1);
                biggest = biggest.max(relevance[i]);
            }

            let mut output = [0u32; HEAD_SIZE];
            let mut softmax_sum = 0u32;
            for i in 0..cache_len {
                let power = (biggest - relevance[i]) >> 10;
                let res = if power >= 1024 {
                    0
                } else {
                    self.softmax_exp[power as usize]
                };
                softmax_sum = softmax_sum.wrapping_add(res);
            }
            softmax_sum &= FIXED_POINT_MASK;
            let softmax_sum_inv = (1u64 << 39) / softmax_sum as u64;

            for i in 0..cache_len {
                let power = (biggest - relevance[i]) >> 10;
                let res = if power >= 1024 {
                    0
                } else {
                    self.softmax_exp[power as usize]
                };
                let mut res =
                    ((softmax_sum_inv * res as u64) >> 17) as u32 & (FIXED_POINT_MASK / 2);
                res = self.to_float16(res, 4) as u32;

                for (j, &v) in self.v_cache[head][i].iter().enumerate() {
                    output[j] = output[j].wrapping_add(float_mult(res as u16, v, 0));
                    output[j] &= FIXED_POINT_MASK;
                }
            }

            for (i, &val) in output.iter().enumerate() {
                proj_input[proj_offset + i] = val;
            }
            proj_offset += HEAD_SIZE;
        }

        let result = self.matmul_proj.forward(&proj_input);
        result
    }
}

fn float_mult(a: u16, b: u16, shift: u32) -> u32 {
    let mut neg = false;
    let offset = ((a >> 10) & 31) + ((b >> 10) & 31);

    let mut a = a as u32;
    let mut b = b as u32;

    if a >= (1 << 15) {
        neg = !neg;
        a -= 1 << 15;
    }
    if b >= (1 << 15) {
        neg = !neg;
        b -= 1 << 15;
    }

    if a > 0 {
        a = (a & ((1 << 10) - 1)) + (1 << 10);
    }
    if b > 0 {
        b = (b & ((1 << 10) - 1)) + (1 << 10);
    }

    let mut res: u32 = (((a as u128 * b as u128) << offset) >> (56 + shift)) as u32;
    res &= FIXED_POINT_MASK;
    if neg {
        res = res.wrapping_neg() & FIXED_POINT_MASK;
    }
    res
}
