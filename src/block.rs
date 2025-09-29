use crate::attention::Attention;
use crate::layernorm::LayerNorm;
use crate::mlp::MLP;
use crate::{EMBED_SIZE, FIXED_POINT_MASK, Fixed24};

pub struct Block {
    ln_1: LayerNorm,
    att: Attention,
    ln_2: LayerNorm,
    mlp: MLP,
}

impl Block {
    pub fn new(block_num: usize) -> Self {
        Block {
            ln_1: LayerNorm::new(2 * block_num + 1),
            att: Attention::new(block_num),
            ln_2: LayerNorm::new(2 * block_num + 2),
            mlp: MLP::new(block_num),
        }
    }

    pub fn forward(&mut self, input: &mut [Fixed24; EMBED_SIZE]) {
        let ln1_out = self.ln_1.forward(input);
        let att_diff = self.att.forward(&ln1_out);

        for i in 0..EMBED_SIZE {
            input[i] = input[i].wrapping_add(att_diff[i]) & FIXED_POINT_MASK;
        }

        let ln2_out = self.ln_2.forward(input);
        let mlp_diff = self.mlp.forward(&ln2_out);

        for i in 0..EMBED_SIZE {
            input[i] = input[i].wrapping_add(mlp_diff[i]) & FIXED_POINT_MASK;
        }
    }

    #[allow(dead_code)]
    pub fn undo_last(&mut self) {
        self.att.undo_last();
    }
}
