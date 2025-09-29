use std::fs::File;
use std::io::Read;

use crate::{Fixed24, EMBED_SIZE, FIXED_POINT_MASK};

pub struct Embedding {
    wte: Vec<Vec<u32>>,
    wpe: Vec<Vec<u32>>,
}

impl Embedding {
    pub fn new() -> Self {
        let mut wte = Vec::new();

        for i in 0..60 {
            let path = format!("weights/weight_files/embedding/wte_{}.bin", i + 1);
            let mut file = File::open(path).expect("couldn't read file");

            for _ in 0..32 {
                let mut embedding = Vec::with_capacity(EMBED_SIZE);
                for _ in 0..EMBED_SIZE {
                    let mut buf = [0u8; 3];
                    file.read_exact(&mut buf)
                        .expect("couldn't read into buffer");
                    let mut cur = u32::from_le_bytes([buf[0], buf[1], buf[2], 0]);
                    if cur >= (1 << 17) {
                        cur |= ((1 << 18) * ((1 << 6) - 1)) & 0xFFFFFF;
                    }
                    embedding.push(cur);
                }
                wte.push(embedding);
            }
        }

        let mut wpe = Vec::new();

        for i in 0..2 {
            let path = format!("weights/weight_files/embedding/wpe_{}.bin", i + 1);
            let mut file = File::open(path).expect("couldn't read file");

            for _ in 0..32 {
                let mut embedding = Vec::with_capacity(EMBED_SIZE);
                for _ in 0..EMBED_SIZE {
                    let mut buf = [0u8; 3];
                    file.read_exact(&mut buf)
                        .expect("couldn't read into buffer");
                    let mut cur = u32::from_le_bytes([buf[0], buf[1], buf[2], 0]);
                    if cur >= (1 << 17) {
                        cur |= ((1 << 18) * ((1 << 6) - 1)) & 0xFFFFFF;
                    }
                    embedding.push(cur);
                }
                wpe.push(embedding);
            }
        }

        Embedding { wte, wpe }
    }

    pub fn get_weights(&self, token: usize, pos: Option<usize>) -> Vec<Fixed24> {
        let mut weights = self.wte[token].clone();

        if let Some(pos) = pos {
            assert!(pos < 64);
            for i in 0..EMBED_SIZE {
                weights[i] = weights[i].wrapping_add(self.wpe[pos][i]) & FIXED_POINT_MASK;
            }
        }

        weights
    }
}
