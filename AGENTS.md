# AGENTS.md

## Project Overview

Rust workspace providing bridges between `ndarray`, OpenCV (`opencv` crate), and related
image processing utilities. Organization: `aftershootco`. Edition 2024, resolver v2.

### Workspace Crates

| Crate | Purpose |
|---|---|
| `ndcv-bridge` | Core library: ndarray <-> OpenCV Mat conversions, gaussian blur, sobel, resize, color space, contours, blending, orientation, ROI |
| `ndcv-cli` | CLI binary (`ndcv`) with subcommands: blur, sobel, resize, color, orient, blend, pipeline, info |
| `ndarray-image` | Conversions between `image` crate types and ndarray |
| `ndarray-resize` | Standalone `fast_image_resize` <-> ndarray bridge |
| `ndarray-safetensors` | Serialize/deserialize ndarray to SafeTensors format |
| `bounding-box` | Generic N-dimensional axis-aligned bounding box with NMS, ROI, drawing |

## Build Commands

The project uses Nix for reproducible builds. Enter the dev shell with `nix develop` or
`direnv allow`. All tools below are available in the Nix dev shell.

```bash
# Build the entire workspace
cargo build

# Build a specific crate
cargo build -p ndcv-bridge
cargo build -p ndcv-cli

# Build without OpenCV (feature-gated modules excluded)
cargo build -p ndcv-bridge --no-default-features
```

## Test Commands

Tests use `cargo-nextest` (available in Nix dev shell). Standard `cargo test` also works.

```bash
# Run all tests
cargo nextest run

# Run tests for a specific crate
cargo nextest run -p ndcv-bridge
cargo nextest run -p bounding-box

# Run a single test by name (substring match)
cargo nextest run -p ndcv-bridge test_gaussian_basic
cargo nextest run -p ndcv-bridge sobel_2d_preserves_shape

# Run tests matching a pattern
cargo nextest run -p ndcv-bridge sobel_

# Standard cargo test (also works)
cargo test -p ndcv-bridge -- test_gaussian_basic
```

## Lint and Format Commands

```bash
# Clippy (CI runs with --deny warnings)
cargo clippy --all-targets -- --deny warnings

# Format check
cargo fmt --check

# Format fix
cargo fmt

# TOML formatting (taplo, available in Nix shell)
taplo fmt

# License/advisory checks
cargo deny check
cargo audit
```

## Benchmarks

Benchmarks use `divan` with `harness = false`.

```bash
cargo bench -p ndcv-bridge --bench gaussian
cargo bench -p ndcv-bridge --bench conversions
```

## Code Coverage

```bash
cargo llvm-cov          # requires cargo-llvm-cov from Nix shell
```

## Code Style Guidelines

### Formatting

- Default `rustfmt` settings (no `rustfmt.toml`). Do not add one.
- Default clippy settings. CI enforces `--deny warnings` -- all clippy warnings must be fixed.
- TOML files are formatted with `taplo`.

### Imports

- Glob imports are used for closely-related crates: `use ndarray::*;`, `use crate::conversions::*;`.
- For external crates with potential name collisions, use qualified paths or explicit imports.
- No enforced grouping or blank-line separation between std/external/crate imports.
- Prefer `use crate::` for intra-crate imports.

### Naming Conventions

- **Traits wrapping OpenCV functions**: prefix with `NdCv` (e.g., `NdCvGaussianBlur`, `NdCvSobel`, `NdCvResize`, `NdCvFindContours`).
- **Traits for pure-Rust operations**: prefix with `Nd` (e.g., `NdBlend`, `NdFir`, `NdRoiZeroPadded`) or use bare PascalCase for domain concepts (`Orient`, `CvType`, `Roi`).
- **Error types**: per-module enum with `Error` suffix (e.g., `GaussianBlurError`, `NdCvSobelError`, `ConversionError`).
- **Functions**: snake_case. Use `_def` suffix for variants with default parameters (e.g., `gaussian_blur_def`). Use `_inplace` suffix for in-place mutations.
- **Enums**: PascalCase variants. Use `#[repr(C)]` or `#[repr(i32)]` when mapping to C/OpenCV constants.
- **Type aliases**: short abbreviations are acceptable (e.g., `Aabb2<T>`, `BBox2`).

### Module Organization

- OpenCV-dependent modules are gated with `#[cfg(feature = "opencv")]`.
- Each module has a `mod seal { pub trait Sealed {} }` pattern to restrict trait implementations.
- Flat re-exports at crate root in `lib.rs` -- public API items are `pub use`d from submodules.
- Internal prelude: `pub(crate) mod prelude_` with common error types and `Result` alias.

### Error Handling

- Primary: `thiserror` derive for per-module error enums. Each module defines its own error type.
- Error enums typically have variants for `ConversionError` and `OpenCvError` wrapping underlying errors with `#[from]`.
- Some modules use `error-stack` with `change_context` (notably `ndarray-resize`).
- CLI uses `anyhow` for top-level error handling.
- Functions return `Result<T, ModuleSpecificError>`. Do not use `Box<dyn Error>`.

### Trait Design

- Extension traits are implemented on `ndarray::ArrayBase<S, D>` with appropriate bounds.
- Typical bound pattern: `T: CvType + seal::Sealed`, `S: RawData + Data<Elem = T>`, `D: Dimension`.
- Traits often have a full-parameter method and a `_def` convenience method with defaults.
- Use `derive_builder` for complex argument structs (see `SobelArgs`).

### Unsafe Code

- Unsafe is used for zero-copy OpenCV Mat <-> ndarray conversions and in-place OpenCV operations.
- Always wrap in minimal `unsafe {}` blocks.
- The `inplace.rs` module provides `op_inplace` helper for safe in-place OpenCV operations using pointer aliasing.
- SIMD operations via `wide` crate (e.g., `f32x4` in blending).

### Tests

- Tests are inline: `#[cfg(test)] mod tests { ... }` within source files.
- No separate test files or `tests/` directory for unit tests.
- Test function naming: `test_` prefix or descriptive snake_case (e.g., `sobel_2d_preserves_shape`, `test_gaussian_basic`).
- Helper functions at the top of test modules (e.g., `vertical_edge_image()`, `horizontal_edge_image()`).
- Use `assert_eq!`, `assert!` with descriptive messages. `pretty_assertions` is available.
- Use `#[should_panic]` for expected failure tests.
- Test patterns: shape preservation, round-trip conversions, edge cases, boundary values, type combinations.

### Dependencies and Features

- `opencv` feature is default-enabled. Building without it excludes all OpenCV-dependent modules.
- `bytemuck` is used for safe type punning and Pod trait bounds.
- `rayon` is used for parallel iteration (ndarray `rayon` feature enabled).
- `glam` vector types are mapped to OpenCV types (Vec2 -> CV_32FC2, etc.).
- `nalgebra` is used for matrix operations alongside ndarray.
- Image dimension convention: `(height, width, channels)` for 3D arrays, `(height, width)` for 2D.

### CI Checks (all must pass)

1. `cargo clippy --all-targets -- --deny warnings`
2. `cargo fmt --check`
3. `taplo fmt --check`
4. `cargo nextest run`
5. `cargo doc` (must build without warnings)
6. `cargo audit`
7. `cargo deny check`
8. `cargo llvm-cov` (Linux only)
