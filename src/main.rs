use std::fs::File;
use std::io::{self, BufRead, Write};

mod layernorm;
mod matmul;
mod mlp;
mod prng;
use prng::PRNG;
mod attention;
mod block;
mod model;
use model::Model;
mod embedding;
pub use embedding::Embedding;
mod unembedding;
pub use unembedding::Unembedding;

const EMBED_SIZE: usize = 240;
const FIXED_POINT_SIZE: u32 = 24;
const FIXED_POINT_MASK: u32 = (1 << FIXED_POINT_SIZE) - 1;

type Fixed24 = u32;

fn get_prompt(tokens: &[String]) -> io::Result<Vec<usize>> {
    print!("Enter prompt: ");
    io::stdout().flush()?;

    let mut input = String::new();
    io::stdin().read_line(&mut input)?;
    let mut prompt = format!(" {}", input.trim().to_lowercase());

    let mut ans = vec![0];

    while !prompt.is_empty() {
        let mut maxlength = 0;
        let mut best = None;

        for (i, token) in tokens.iter().enumerate() {
            let token_str = token.replace('_', " ");
            if token_str.len() <= prompt.len()
                && token_str == &prompt[..token_str.len()]
                && token_str.len() > maxlength
            {
                maxlength = token_str.len();
                best = Some(i);
            }
        }

        if let Some(best_idx) = best {
            ans.push(best_idx);
            prompt = prompt[maxlength..].to_string();
        } else {
            println!("Could not parse prompt: '{}'", prompt);
            std::process::exit(1);
        }
    }

    Ok(ans)
}

fn main() -> io::Result<()> {
    let tokens: Vec<String> = {
        let file = File::open("tokens.txt")?;
        let reader = io::BufReader::new(file);
        reader.lines().collect::<Result<_, _>>()?
    };

    let mut conversation = Vec::new();
    let mut model = Model::new();

    print!("Enter RNG seed, or -1 to view next token probability distribution: ");
    io::stdout().flush()?;
    let mut seed_input = String::new();
    io::stdin().read_line(&mut seed_input)?;
    let seed: i32 = seed_input.trim().parse().unwrap();

    let mut rng = PRNG::new(seed as u32);

    loop {
        let prompt = get_prompt(&tokens)?;
        conversation.extend_from_slice(&prompt);
        conversation.push(1);

        for &token in &prompt {
            println!("Processing token '{}'", tokens[token]);
            assert!(token < unembedding::VOCAB_SIZE);
            model.process(token);
        }

        if seed == -1 {
            let mut nxt = 1;
            loop {
                println!("Processing token '{}'", tokens[nxt]);
                let ans = model.process(nxt);
                for i in 0..8 {
                    let token = (ans[i] & 2047) as usize;
                    let prob = (ans[i] >> 11) as f64;
                    let prob_normalized =
                        (prob / ((1u64 << 23) as f64) * 100000.0).round() / 100000.0;
                    println!(
                        "{}: {:>4}, probability {:.5}, {}",
                        i + 1,
                        token,
                        prob_normalized,
                        tokens[token]
                    );
                }

                print!("Enter next token ID: ");
                io::stdout().flush()?;
                let mut nxt_input = String::new();
                io::stdin().read_line(&mut nxt_input)?;
                nxt = nxt_input.trim().parse().unwrap();

                if nxt == 0 || nxt == 1 {
                    break;
                }
                conversation.push(nxt);
            }
        } else {
            let mut nxt = 1;
            loop {
                println!("Processing token '{}'", tokens[nxt]);
                let act = model.process(nxt);
                let mut cur = rng.next() as i32;
                let mut here = -1i32;

                for j in (1..=7).rev() {
                    if (act[j] >> 11) < (1 << 20) {
                        continue;
                    }
                    cur -= (act[j] >> 11) as i32;
                    if cur < 0 {
                        here = (act[j] & 2047) as i32;
                        break;
                    }
                }

                if here == -1 {
                    here = (act[0] & 2047) as i32;
                }

                if here == 0 || here == 1 {
                    break;
                }

                conversation.push(here as usize);
                nxt = here as usize;
            }
        }

        let output: String = conversation
            .iter()
            .map(|&i| tokens[i].as_str())
            .collect::<String>()
            .replace('_', " ");
        println!("{}", output);
    }
}
