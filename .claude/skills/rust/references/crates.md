# Crate preferences

Default picks per problem area. Deviating is allowed, but say why.

## Error handling

- library: `thiserror`
- binary: `error-stack`

## Async

- runtime: `tokio`, with `smol` as fallback
- utilities: `futures`

## Logging

- `tracing` for logging and tracing
- `tracing-subscriber` for formatting

## Configuration

- `config` for config loading
- `serde` for serialization

## HTTP

- `reqwest` for making HTTP calls
- `axum` for HTTP servers

## Parsing

- `nom` for parsers for arbitrary formats that don't already have serde support

## Image Processing

- `image` for reading and writing generic images (jpg / png etc)
- `fast_image_resize` for resizing images

## Array Processing

- `ndarray` When working with large arrays like images.
- `nalgebra` when working with complex maths and geometry
