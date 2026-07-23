//! MTP (Multi-Token Prediction / NextN) example — the full speculative driver.
//!
//! Runs ik's `common_speculative` MTP flow via the safe `MtpSpeculative`
//! (new → begin(prompt) → step()*), the Rust equivalent of
//! `llama-cli --spec-type mtp:n_max=1,p_min=0.0`. Requires the `common` feature
//! (enabled by this example's dependency) and a combined NextN GGUF (see
//! scripts/prepare_models.sh).
//!
//! Usage:
//!   cargo run -p mtp -- --model .models/mtp-combined.gguf --prompt "..." -n 16

use anyhow::{bail, Context, Result};
use clap::Parser;

use ik_llama_cpp_2::{
    LlamaBackend, LlamaContext, LlamaContextParams, LlamaModel, LlamaModelParams, MtpSpeculative,
    MtpSpeculativeParams,
};

#[derive(Parser, Debug)]
#[command(about = "ik_llama.cpp MTP speculative-decoding example")]
struct Args {
    /// Path to a combined NextN GGUF (general + mtp halves).
    #[arg(long)]
    model: String,
    /// Prompt text.
    #[arg(long, default_value = "The capital of France is")]
    prompt: String,
    /// Tokens to generate.
    #[arg(short = 'n', long, default_value_t = 32)]
    n_len: usize,
    /// GPU layers (0 = CPU).
    #[arg(long, default_value_t = 0)]
    n_gpu_layers: u32,
    /// Threads.
    #[arg(short = 't', long, default_value_t = 8)]
    threads: u32,
}

fn main() -> Result<()> {
    tracing_subscriber::fmt().with_env_filter("warn").init();
    let args = Args::parse();

    let backend = LlamaBackend::init().context("backend init")?;

    // NextN model must be loaded with .with_mtp(true).
    let mparams = LlamaModelParams::default()
        .with_mtp(true)
        .with_n_gpu_layers(args.n_gpu_layers);
    let model =
        LlamaModel::load_from_file(&backend, &args.model, &mparams).context("load model")?;

    let n_nextn = model.n_nextn_layer();
    if n_nextn <= 0 {
        bail!("model reports {n_nextn} NextN layers — not an MTP model (or missing .with_mtp).");
    }
    println!("MTP model: n_nextn_layer = {n_nextn}");

    // Context must also be created with .with_mtp(true).
    let cparams = LlamaContextParams::default()
        .with_n_ctx(std::num::NonZeroU32::new(2048))
        .with_n_threads(args.threads)
        .with_mtp(true)
        .with_seed(42);
    let mut ctx = LlamaContext::new(&model, &cparams).context("create MTP context")?;

    let prompt = model.tokenize(&args.prompt, true).context("tokenize")?;

    let mut spec = MtpSpeculative::new(&model, &mut ctx, MtpSpeculativeParams::default())
        .context("init MTP driver")?;
    spec.begin(&prompt).context("MTP begin (prompt warmup)")?;

    print!("{}", args.prompt);
    use std::io::Write;
    let mut generated = 0usize;
    let mut steps = 0usize;
    let mut drafts_accepted = 0i64;
    'gen: while generated < args.n_len {
        let step = spec.step().context("MTP step")?;
        steps += 1;
        drafts_accepted += i64::from(step.n_accepted);
        for tok in step.tokens {
            if model.is_eog(tok) {
                break 'gen;
            }
            print!("{}", model.token_to_piece_lossy(tok)?);
            std::io::stdout().flush().ok();
            generated += 1;
            if generated >= args.n_len {
                break 'gen;
            }
        }
    }
    println!();
    let rate = if steps > 0 {
        drafts_accepted as f64 / steps as f64
    } else {
        0.0
    };
    println!(
        "MTP EXAMPLE OK: n_nextn_layer={n_nextn}, steps={steps}, generated={generated}, \
         drafts_accepted={drafts_accepted}, accept_rate={rate:.2}/step"
    );
    Ok(())
}
