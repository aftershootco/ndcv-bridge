mod io;
mod ops;
mod pipeline;

use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};

use io::NdImage;

#[derive(Parser)]
#[command(
    name = "ndcv",
    about = "CLI for experimenting with images using ndcv-bridge filters",
    long_about = "Load images, apply filters (blur, sobel, resize, color conversion, \
                  orient/flip, blend), and save results. Supports chaining operations \
                  via the pipeline subcommand.",
    version
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Apply Gaussian blur to an image
    Blur {
        /// Input image path
        input: PathBuf,
        /// Output image path
        #[arg(short, long)]
        output: PathBuf,
        #[command(flatten)]
        args: ops::blur::BlurArgs,
    },

    /// Apply Sobel edge detection
    Sobel {
        /// Input image path
        input: PathBuf,
        /// Output image path
        #[arg(short, long)]
        output: PathBuf,
        #[command(flatten)]
        args: ops::sobel::SobelArgsCli,
    },

    /// Resize an image
    Resize {
        /// Input image path
        input: PathBuf,
        /// Output image path
        #[arg(short, long)]
        output: PathBuf,
        #[command(flatten)]
        args: ops::resize::ResizeArgs,
    },

    /// Convert image color space
    Color {
        /// Input image path
        input: PathBuf,
        /// Output image path
        #[arg(short, long)]
        output: PathBuf,
        #[command(flatten)]
        args: ops::color::ColorArgs,
    },

    /// Rotate, flip, or orient an image
    Orient {
        /// Input image path
        input: PathBuf,
        /// Output image path
        #[arg(short, long)]
        output: PathBuf,
        #[command(flatten)]
        args: ops::orient::OrientArgs,
    },

    /// Blend two images using a mask
    Blend {
        /// Base (input) image path
        input: PathBuf,
        /// Output image path
        #[arg(short, long)]
        output: PathBuf,
        #[command(flatten)]
        args: ops::blend::BlendArgs,
    },

    /// Apply Extended Difference of Gaussians (XDoG) edge detection
    Xdog {
        /// Input image path
        input: PathBuf,
        /// Output image path
        #[arg(short, long)]
        output: PathBuf,
        #[command(flatten)]
        args: ops::xdog::XDoGArgs,
    },

    /// Chain multiple operations in sequence
    ///
    /// Operations are separated by commas. Example:
    ///   ndcv pipeline input.jpg -o out.jpg -- blur --sigma 1.5 , resize --width 800 , color --to gray
    Pipeline {
        /// Input image path
        input: PathBuf,
        /// Output image path
        #[arg(short, long)]
        output: PathBuf,
        /// Operations to apply (separated by commas)
        #[command(flatten)]
        args: pipeline::PipelineArgs,
    },

    /// Print image information (dimensions, channels, format)
    Info {
        /// Input image path
        input: PathBuf,
    },

    Completions {
        /// Shell type for which to generate completions
        #[arg(value_enum)]
        shell: clap_complete::Shell,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Command::Blur {
            input,
            output,
            args,
        } => {
            let image = NdImage::load(&input)?;
            eprintln!(
                "loaded: {}x{}, {} channels",
                image.width(),
                image.height(),
                image.channels()
            );
            let result = ops::blur::run(&image, &args).context("blur failed")?;
            result.save(&output)?;
            eprintln!("saved: {}", output.display());
        }

        Command::Sobel {
            input,
            output,
            args,
        } => {
            let image = NdImage::load(&input)?;
            eprintln!(
                "loaded: {}x{}, {} channels",
                image.width(),
                image.height(),
                image.channels()
            );
            let result = ops::sobel::run(&image, &args).context("sobel failed")?;
            result.save(&output)?;
            eprintln!("saved: {}", output.display());
        }

        Command::Resize {
            input,
            output,
            args,
        } => {
            let image = NdImage::load(&input)?;
            eprintln!(
                "loaded: {}x{}, {} channels",
                image.width(),
                image.height(),
                image.channels()
            );
            let result = ops::resize::run(&image, &args).context("resize failed")?;
            eprintln!(
                "resized: {}x{}, {} channels",
                result.width(),
                result.height(),
                result.channels()
            );
            result.save(&output)?;
            eprintln!("saved: {}", output.display());
        }

        Command::Color {
            input,
            output,
            args,
        } => {
            let image = NdImage::load(&input)?;
            eprintln!(
                "loaded: {}x{}, {} channels",
                image.width(),
                image.height(),
                image.channels()
            );
            let result = ops::color::run(&image, &args).context("color conversion failed")?;
            eprintln!(
                "converted: {}x{}, {} channels",
                result.width(),
                result.height(),
                result.channels()
            );
            result.save(&output)?;
            eprintln!("saved: {}", output.display());
        }

        Command::Orient {
            input,
            output,
            args,
        } => {
            let image = NdImage::load(&input)?;
            eprintln!(
                "loaded: {}x{}, {} channels",
                image.width(),
                image.height(),
                image.channels()
            );
            let result = ops::orient::run(&image, &args).context("orient failed")?;
            eprintln!(
                "oriented: {}x{}, {} channels",
                result.width(),
                result.height(),
                result.channels()
            );
            result.save(&output)?;
            eprintln!("saved: {}", output.display());
        }

        Command::Blend {
            input,
            output,
            args,
        } => {
            let image = NdImage::load(&input)?;
            eprintln!(
                "loaded base: {}x{}, {} channels",
                image.width(),
                image.height(),
                image.channels()
            );
            let result = ops::blend::run(&image, &args).context("blend failed")?;
            result.save(&output)?;
            eprintln!("saved: {}", output.display());
        }

        Command::Xdog {
            input,
            output,
            args,
        } => {
            let image = NdImage::load(&input)?;
            eprintln!(
                "loaded: {}x{}, {} channels",
                image.width(),
                image.height(),
                image.channels()
            );
            let result = ops::xdog::run(&image, &args).context("xdog failed")?;
            result.save(&output)?;
            eprintln!("saved: {}", output.display());
        }

        Command::Pipeline {
            input,
            output,
            args,
        } => {
            let image = NdImage::load(&input)?;
            eprintln!(
                "loaded: {}x{}, {} channels",
                image.width(),
                image.height(),
                image.channels()
            );
            let result = pipeline::run(&image, &args).context("pipeline failed")?;
            eprintln!(
                "result: {}x{}, {} channels",
                result.width(),
                result.height(),
                result.channels()
            );
            result.save(&output)?;
            eprintln!("saved: {}", output.display());
        }

        Command::Info { input } => {
            let image = NdImage::load(&input)?;
            println!("file:     {}", input.display());
            println!("width:    {} px", image.width());
            println!("height:   {} px", image.height());
            println!("channels: {}", image.channels());
            let color_type = match image.channels() {
                1 => "grayscale",
                3 => "RGB",
                4 => "RGBA",
                n => &format!("{}-channel", n),
            };
            println!("type:     {}", color_type);
        }

        Command::Completions { shell } => {
            let mut command = <Cli as clap::CommandFactory>::command();
            clap_complete::generate(
                shell,
                &mut command,
                env!("CARGO_BIN_NAME"),
                &mut std::io::stdout(),
            );
        }
    }

    Ok(())
}
