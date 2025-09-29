use crate::EMBED_SIZE;
use crate::block::Block;
use crate::embedding::Embedding;
use crate::layernorm::LayerNorm;
use crate::unembedding::Unembedding;

const LAYERS: usize = 6;

pub struct Model {
    tokens: Embedding,
    transformer: [Block; LAYERS],
    ln_f: LayerNorm,
    unembedding: Unembedding,
    index: usize,
}

impl Model {
    pub fn new() -> Self {
        let tokens = Embedding::new();
        let transformer = std::array::from_fn(Block::new);

        let ln_f = LayerNorm::new(13);
        let unembedding = Unembedding::new();

        println!("Model loaded.");

        Model {
            tokens,
            transformer,
            ln_f,
            unembedding,
            index: 0,
        }
    }

    pub fn process(&mut self, token: usize) -> Vec<u64> {
        let pos = if self.index < 64 {
            Some(self.index)
        } else {
            Some(63)
        };
        let weights_vec = self.tokens.get_weights(token, pos);
        let mut value = [0u32; EMBED_SIZE];
        for (i, &w) in weights_vec.iter().enumerate() {
            value[i] = w;
        }

        for block in &mut self.transformer {
            block.forward(&mut value);
        }

        let value = self.ln_f.forward(&value);
        let ans = self.unembedding.forward(&value);
        self.index += 1;
        ans
    }

    #[allow(dead_code)]
    pub fn undo_last(&mut self) {
        self.index -= 1;
        for block in &mut self.transformer {
            block.undo_last();
        }
    }
}
