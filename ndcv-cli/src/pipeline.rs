use anyhow::{Context, Result, bail};
use clap::Args;

use crate::io::NdImage;
use crate::ops;

/// A single operation in a pipeline, parsed from comma-separated segments.
#[derive(Debug, Clone)]
pub enum PipelineOp {
    Blur(ops::blur::BlurArgs),
    Sobel(ops::sobel::SobelArgsCli),
    Resize(ops::resize::ResizeArgs),
    Color(ops::color::ColorArgs),
    Orient(ops::orient::OrientArgs),
    Xdog(ops::xdog::XDoGArgs),
}

#[derive(Debug, Clone, Args)]
pub struct PipelineArgs {
    /// Pipeline operations, separated by commas.
    ///
    /// Example: blur --sigma 1.5 , resize --width 800 , color --to gray
    ///
    /// Each segment is: <operation-name> [args...]
    /// Supported operations: blur, sobel, resize, color, orient, xdog
    #[arg(trailing_var_arg = true, allow_hyphen_values = true, num_args = 1..)]
    pub ops: Vec<String>,
}

impl PipelineOp {
    /// Apply this operation to the image
    pub fn apply(&self, image: &NdImage) -> Result<NdImage> {
        match self {
            PipelineOp::Blur(args) => ops::blur::run(image, args),
            PipelineOp::Sobel(args) => ops::sobel::run(image, args),
            PipelineOp::Resize(args) => ops::resize::run(image, args),
            PipelineOp::Color(args) => ops::color::run(image, args),
            PipelineOp::Orient(args) => ops::orient::run(image, args),
            PipelineOp::Xdog(args) => ops::xdog::run(image, args),
        }
    }

    pub fn name(&self) -> &'static str {
        match self {
            PipelineOp::Blur(_) => "blur",
            PipelineOp::Sobel(_) => "sobel",
            PipelineOp::Resize(_) => "resize",
            PipelineOp::Color(_) => "color",
            PipelineOp::Orient(_) => "orient",
            PipelineOp::Xdog(_) => "xdog",
        }
    }
}

/// Parse the raw pipeline args into a sequence of operations.
///
/// The args are split on "," tokens to form segments, and each segment
/// is parsed as a subcommand using clap.
pub fn parse_pipeline(args: &PipelineArgs) -> Result<Vec<PipelineOp>> {
    let segments = split_on_comma(&args.ops);

    if segments.is_empty() {
        bail!("pipeline requires at least one operation");
    }

    let mut ops = Vec::new();
    for (i, segment) in segments.iter().enumerate() {
        if segment.is_empty() {
            bail!("empty operation in pipeline at position {}", i + 1);
        }

        let op_name = &segment[0];
        let op_args = &segment[1..];

        let op = parse_single_op(op_name, op_args).with_context(|| {
            format!(
                "failed to parse pipeline operation #{} '{}'",
                i + 1,
                op_name
            )
        })?;
        ops.push(op);
    }

    Ok(ops)
}

/// Split a flat list of strings on "," tokens.
fn split_on_comma(args: &[String]) -> Vec<Vec<String>> {
    let mut segments: Vec<Vec<String>> = Vec::new();
    let mut current: Vec<String> = Vec::new();

    for arg in args {
        if arg == "," {
            if !current.is_empty() {
                segments.push(std::mem::take(&mut current));
            }
        } else {
            current.push(arg.clone());
        }
    }

    if !current.is_empty() {
        segments.push(current);
    }

    segments
}

/// Parse a single operation from its name and arguments.
fn parse_single_op(name: &str, args: &[String]) -> Result<PipelineOp> {
    use clap::Parser;

    // We build a fake argv: ["ndcv-pipeline", arg1, arg2, ...]
    // and use clap's `try_parse_from` to parse it into the appropriate Args struct.

    let argv: Vec<&str> = std::iter::once("ndcv-pipeline")
        .chain(args.iter().map(|s| s.as_str()))
        .collect();

    match name {
        "blur" => {
            #[derive(Parser)]
            struct Wrapper {
                #[command(flatten)]
                inner: ops::blur::BlurArgs,
            }
            let w = Wrapper::try_parse_from(&argv)?;
            Ok(PipelineOp::Blur(w.inner))
        }
        "sobel" => {
            #[derive(Parser)]
            struct Wrapper {
                #[command(flatten)]
                inner: ops::sobel::SobelArgsCli,
            }
            let w = Wrapper::try_parse_from(&argv)?;
            Ok(PipelineOp::Sobel(w.inner))
        }
        "resize" => {
            #[derive(Parser)]
            struct Wrapper {
                #[command(flatten)]
                inner: ops::resize::ResizeArgs,
            }
            let w = Wrapper::try_parse_from(&argv)?;
            Ok(PipelineOp::Resize(w.inner))
        }
        "color" => {
            #[derive(Parser)]
            struct Wrapper {
                #[command(flatten)]
                inner: ops::color::ColorArgs,
            }
            let w = Wrapper::try_parse_from(&argv)?;
            Ok(PipelineOp::Color(w.inner))
        }
        "orient" => {
            #[derive(Parser)]
            struct Wrapper {
                #[command(flatten)]
                inner: ops::orient::OrientArgs,
            }
            let w = Wrapper::try_parse_from(&argv)?;
            Ok(PipelineOp::Orient(w.inner))
        }
        "xdog" => {
            #[derive(Parser)]
            struct Wrapper {
                #[command(flatten)]
                inner: ops::xdog::XDoGArgs,
            }
            let w = Wrapper::try_parse_from(&argv)?;
            Ok(PipelineOp::Xdog(w.inner))
        }
        _ => bail!(
            "unknown operation '{}' (available: blur, sobel, resize, color, orient, xdog)",
            name
        ),
    }
}

pub fn run(image: &NdImage, args: &PipelineArgs) -> Result<NdImage> {
    let ops = parse_pipeline(args)?;

    let mut current = image.clone();
    for (i, op) in ops.iter().enumerate() {
        eprintln!("[{}/{}] applying: {}", i + 1, ops.len(), op.name());
        current = op
            .apply(&current)
            .with_context(|| format!("pipeline step #{} '{}' failed", i + 1, op.name()))?;
    }

    Ok(current)
}
