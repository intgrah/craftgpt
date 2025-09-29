pub struct PRNG {
    seed: u32,
}

impl PRNG {
    pub fn new(seed: u32) -> Self {
        PRNG { seed }
    }

    pub fn next(&mut self) -> u32 {
        for _ in 0..256 {
            let next_bit = ((self.seed >> 22) & 1) ^ ((self.seed >> 17) & 1);
            self.seed <<= 1;
            self.seed &= (1 << 23) - 1;
            self.seed += next_bit;
        }
        self.seed
    }
}
