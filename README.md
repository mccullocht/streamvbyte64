# Fast Byte-Aligned Integer Coding
This crate is a Rust port of [Daniel Lemire's `streamvbyte` library](https://github.com/lemire/streamvbyte).
It contains multiple implementations of this format aimed at different integer sizes and value distributions.

Each `Coder` implementation produces different formats that are incompatible with one another. Names
provide the length of each of the 4 possible tags for each value so `Coder1234` encodes each entry
as 1, 2, 3, or 4 bytes. A scalar implementation is always available at a large speed penalty but
the implementation will automatically use an accelerated implementation for the target if available.

# Performance

A scalar implementation is available for all `Coder`s but this is typically pretty slow. All
implementations also include acceleration when building for little-endian `aarch64` with `NEON`
instruction support, although it should be easy to extend to `x86_64` in the future (contributions
are welcome).)

`Coder1234` will typically be fastest, but other tag length distributions (including `Coder1248` for
64-bit values) are available.)

Benchmark numbers were collected on M1 Macbook Air, to generate your own run
```
cargo bench --bench=streamvbyte
````

Each benchmark is run with 1024 elements of up to a certain length using a zipf-like distribution.

`encode` and `decode` benchmarks measure value coding throughput, `deltas` variants measure the
same assuming monotonically increasing inputs and compressing deltas between values. `data_len`
measures throughput for determining data length based on tag values, `skip_deltas` does the same
for delta coded streams and includes the sum of all values skipped.

## `Coder1234`

| Benchmark         | Throughput    |
| ----------------- | ------------: |
| `encode/1       ` | ` 4.8Gelem/s` |
| `encode_deltas/1` | ` 4.1Gelem/s` |
| `decode/1       ` | `12.7Gelem/s` |
| `decode_deltas/1` | ` 5.6Gelem/s` |
| `skip_deltas/1  ` | `19.9Gelem/s` |
| `data_len/1     ` | `54.2Gelem/s` |
| `encode/2       ` | ` 4.8Gelem/s` |
| `encode_deltas/2` | ` 3.8Gelem/s` |
| `decode/2       ` | ` 8.2Gelem/s` |
| `decode_deltas/2` | ` 4.2Gelem/s` |
| `skip_deltas/2  ` | ` 8.5Gelem/s` |
| `encode/4       ` | ` 4.8Gelem/s` |
| `encode_deltas/4` | ` 4.1Gelem/s` |
| `decode/4       ` | ` 8.2Gelem/s` |
| `decode_deltas/4` | ` 4.2Gelem/s` |
| `skip_deltas/4  ` | ` 8.4Gelem/s` |

## `Coder0124`

| Benchmark         | Throughput    |
| ----------------- | ------------: |
| `encode/1       ` | ` 4.2Gelem/s` |
| `encode_deltas/1` | ` 3.6Gelem/s` |
| `decode/1       ` | ` 7.2Gelem/s` |
| `decode_deltas/1` | ` 4.2Gelem/s` |
| `skip_deltas/1  ` | ` 7.1Gelem/s` |
| `data_len/1     ` | `53.8Gelem/s` |
| `encode/2       ` | ` 4.2Gelem/s` |
| `encode_deltas/2` | ` 3.6Gelem/s` |
| `decode/2       ` | ` 7.2Gelem/s` |
| `decode_deltas/2` | ` 4.2Gelem/s` |
| `skip_deltas/2  ` | ` 7.3Gelem/s` |
| `encode/4       ` | ` 4.2Gelem/s` |
| `encode_deltas/4` | ` 3.6Gelem/s` |
| `decode/4       ` | ` 7.2Gelem/s` |
| `decode_deltas/4` | ` 4.2Gelem/s` |
| `skip_deltas/4  ` | ` 7.5Gelem/s` |

## `Coder1248`

| Benchmark         | Throughput    |
| ----------------- | ------------: |
| `encode/1       ` | ` 3.1Gelem/s` |
| `encode_deltas/1` | ` 2.5Gelem/s` |
| `decode/1       ` | ` 4.6Gelem/s` |
| `decode_deltas/1` | ` 3.6Gelem/s` |
| `skip_deltas/1  ` | ` 6.0Gelem/s` |
| `data_len/1     ` | `53.8Gelem/s` |
| `encode/4       ` | ` 3.1Gelem/s` |
| `encode_deltas/4` | ` 2.5Gelem/s` |
| `decode/4       ` | ` 4.8Gelem/s` |
| `decode_deltas/4` | ` 3.6Gelem/s` |
| `skip_deltas/4  ` | ` 5.7Gelem/s` |
| `encode/8       ` | ` 3.0Gelem/s` |
| `encode_deltas/8` | ` 2.5Gelem/s` |
| `decode/8       ` | ` 4.8Gelem/s` |
| `decode_deltas/8` | ` 3.0Gelem/s` |
| `skip_deltas/8  ` | ` 5.4Gelem/s` |