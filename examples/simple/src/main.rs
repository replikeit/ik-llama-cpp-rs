//! Minimal general-generation example for ik-llama-cpp-2.
//!
//! Usage:
//!   cargo run -p simple -- --model <merged.gguf> --prompt "The capital of France is" -n 32
//!
//! (Set IK_LLAMA_CPP_LIB_DIR + LD_LIBRARY_PATH to the ik build for the fast path.)

use anyhow::{Context, Result};
use clap::Parser;

use ik_llama_cpp_2::{
    LlamaBackend, LlamaBatch, LlamaContext, LlamaContextParams, LlamaModel, LlamaModelParams,
    LlamaSampler,
};

#[derive(Parser, Debug)]
#[command(about = "ik_llama.cpp general-generation smoke example")]
struct Args {
    /// Path to a single (merged) GGUF model file.
    #[arg(long)]
    model: String,
    /// Prompt text.
    #[arg(long, default_value = "The capital of France is")]
    prompt: String,
    /// Number of tokens to generate.
    #[arg(short = 'n', long, default_value_t = 32)]
    n_len: usize,
    /// Context size.
    #[arg(long, default_value_t = 2048)]
    n_ctx: u32,
    /// Threads.
    #[arg(short = 't', long, default_value_t = 8)]
    threads: u32,
    /// GPU layers to offload (0 = CPU).
    #[arg(long, default_value_t = 0)]
    n_gpu_layers: u32,
    /// Sampling temperature (<= 0 = greedy).
    #[arg(long, default_value_t = 0.0)]
    temp: f32,
}

fn main() -> Result<()> {
    tracing_subscriber::fmt().with_env_filter("warn").init();
    let args = Args::parse();

    let backend = LlamaBackend::init().context("backend init")?;
    let mparams = LlamaModelParams::default().with_n_gpu_layers(args.n_gpu_layers);
    let model =
        LlamaModel::load_from_file(&backend, &args.model, &mparams).context("load model")?;

    let cparams = LlamaContextParams::default()
        .with_n_ctx(std::num::NonZeroU32::new(args.n_ctx))
        .with_n_threads(args.threads)
        .with_seed(42);
    let mut ctx = LlamaContext::new(&model, &cparams).context("create context")?;

    let prompt_tokens = model.tokenize(&args.prompt, true).context("tokenize")?;
    let cap = prompt_tokens.len().max(args.n_len) + 8;
    let mut batch = LlamaBatch::new(cap, 1);
    batch.add_sequence(&prompt_tokens, 0, false)?;
    ctx.decode(&mut batch).context("decode prompt")?;

    let mut sampler = if args.temp > 0.0 {
        LlamaSampler::chain_simple([
            LlamaSampler::top_k(40),
            LlamaSampler::top_p(0.95, 1),
            LlamaSampler::temp(args.temp),
            LlamaSampler::dist(0),
        ])
    } else {
        LlamaSampler::greedy()
    };

    print!("{}", args.prompt);
    let mut logits_idx = batch.n_tokens() - 1;
    let mut n_past = batch.n_tokens();
    for _ in 0..args.n_len {
        let tok = sampler.sample(&ctx, logits_idx);
        sampler.accept(tok);
        if model.is_eog(tok) {
            break;
        }
        print!("{}", model.token_to_piece_lossy(tok)?);
        use std::io::Write;
        std::io::stdout().flush().ok();

        batch.clear();
        batch.add(tok, n_past, &[0], true)?;
        n_past += 1;
        ctx.decode(&mut batch).context("decode token")?;
        logits_idx = 0;
    }
    println!();
    Ok(())
}
