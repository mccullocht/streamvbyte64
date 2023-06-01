/*! # Fast Byte-Aligned Integer Coding
This crate is a Rust port of [Daniel Lemire's `streamvbyte` library](https://github.com/lemire/streamvbyte). It contains
multiple implementations of this format aimed at different value distributions and integer sizes.

Each `Group32` or `Group64` implementation produces different formats that are incompatible with one another. Implementations
will be SIMD accelerate if:
* an implementation has been written for your architecture target
* the CPU being used supports the necessary instructions.

At the moment group implementations only have acceleration on little-endian `aarch64` targets with `NEON` instruction support.

## Example without delta-coding

```
use streamvbyte64::{Group32, Group1234};

let coder = Group1234::new();
let values = vec![
    0u32, 128, 256, 1024, 70, 36, 1000000,
    378, 45, 888, 26, 262144, 88, 89, 90, 16777216
];
let (tag_len, data_len) = Group1234::max_compressed_bytes(values.len());
let mut encoded = vec![0u8; tag_len + data_len];
let (tags, data) = encoded.split_at_mut(tag_len);
// This is the number of bytes in data actually used. If you're writing the encoded data
// to an output you would write tags and &data[..encoded_data_len].
let encoded_data_len = coder.encode(&values, tags, data);

let mut decoded = vec![0u32; values.len()];
coder.decode(tags, &data[..encoded_data_len], &mut decoded);
assert_eq!(values, decoded);

// You can also use this a bit like an array using data_len() to skip whole groups
// at the cost of reading all the tags before the target group.
let mut group = [0u32; 4];
coder.decode(&tags[2..3], &data[coder.data_len(&tags[..2])..], &mut group);
assert_eq!(values[11], group[3]);
```

## Example with delta coding

```
use streamvbyte64::{Group32, Group1234};

let coder = Group1234::new();
let mut sum = 0u32;
let values = [
    0u32, 128, 256, 1024, 70, 36, 1000000,
    378, 45, 888, 26, 262144, 88, 89, 90, 16777216
].iter().map(|x| {
    sum += x;
    sum
}).collect::<Vec<_>>();
let (tag_len, data_len) = Group1234::max_compressed_bytes(values.len());
let mut encoded = vec![0u8; tag_len + data_len];
let (tags, data) = encoded.split_at_mut(tag_len);
let nodelta_data_len = coder.encode(&values, tags, data);
// encode_deltas() and decode_deltas() both accept an initial value that is subtracted from
// or added to all the value in the stream. At the beginning of the stream this is usually
// zero but may be non-zero if you are encoding/decoding in the middle of a stream.
let encoded_data_len = coder.encode_deltas(0, &values, tags, data);
// Fewer bytes are written with the delta encoding as we only record the distance between
// each value and not the value itself.
assert!(encoded_data_len < nodelta_data_len);

let mut decoded = vec![0u32; values.len()];
coder.decode_deltas(0, tags, &data[..encoded_data_len], &mut decoded);
assert_eq!(values, decoded);

// You can also use this a bit like an array using skip_deltas() to skip whole groups
// at a cost that is less than straight decoding.
let (skip_data_len, initial) = coder.skip_deltas(&tags[..2], data);
let mut group = [0u32; 4];
coder.decode_deltas(initial, &tags[2..3], &data[skip_data_len..], &mut group);
assert_eq!(values[11], group[3]);
```
*/

mod arch;
mod group_impl;
mod raw_group;
mod tag_utils;

// TODO: add a GroupTag trait that is shared with RawGroup, it would be useful for testing.

/// `Group32` compresses and decompresses groups of 4 32-bit integers together in a byte-aligned format.
/// Different `Group32` implementations may support different byte length distributions.
///
/// Groups are coded into two separate streams: a tag stream where each byte describes the contents of a group, and a
/// data stream containing values as described by the tag. Use `max_compressed_bytes()` to compute the number of tag
/// and data bytes that must be allocated to safely encode a slice of input values.
pub trait Group32: Sized + Copy + Clone {
    /// Create a new Group32 coder, selecting the fastest implementation available (ideally SIMD accelerated).
    ///
    /// These objects should be relatively cheap to create and require no heap allocation.
    fn new() -> Self;

    /// Returns the number of `(tag_bytes, data_bytes)` required to compress a slice of length `len`.
    fn max_compressed_bytes(len: usize) -> (usize, usize) {
        let num_groups = (len + 3) / 4;
        (num_groups, num_groups * 4 * std::mem::size_of::<u32>())
    }

    /// Encodes a slice of values, writing tags and data to separate streams.
    ///
    /// For every 4 input values, one tag byte and up to 16 data bytes may be written to the output streams.
    ///
    /// Returns the number of bytes written to the data stream.
    ///
    /// # Panics
    ///
    /// - If `values.len() % 4 != 0`
    /// - If `tags` or `data` are too small to fit all of the output data.
    fn encode(&self, values: &[u32], tags: &mut [u8], data: &mut [u8]) -> usize;

    /// Encodes a slice of values, writing tags and data to separate streams. Values are interpreted as deltas starting from `initial`
    /// which produces a more compact output that is also more expensive to encode and decode.
    ///
    /// For every 4 input values, one tag byte and up to 16 data bytes may be written to the output streams.
    /// Returns the number of bytes written to the data stream.
    ///
    /// # Panics
    ///
    /// - If `values.len() % 4 != 0`
    /// - If `tags` or `data` are too small to fit all of the output data.
    fn encode_deltas(
        &self,
        initial: u32,
        values: &[u32],
        tags: &mut [u8],
        data: &mut [u8],
    ) -> usize;

    /// Decodes input tag and data streams to an output slice.
    ///
    /// Returns the number of bytes consumed from the data stream, and one tag byte is consumed for every 4 values decoded.
    ///
    /// # Panics
    ///
    /// - If `values.len() % 4 != 0`.
    /// - If `tags.len() < values.len() / 4`.
    /// - If decoding would consume bytes past the end of `data`.
    fn decode(&self, tags: &[u8], data: &[u8], values: &mut [u32]) -> usize;

    /// Decodes input tag and data streams to an output slice. Values are interepreted as deltas starting from `initial`.
    ///
    /// Returns the number of bytes consumed from the data stream, and one tag byte is consumed for every 4 values decoded.
    ///
    /// # Panics
    ///
    /// - If `values.len() % 4 != 0`.
    /// - If `tags.len() < values.len() / 4`.
    /// - If decoding would consume bytes past the end of `data`.
    fn decode_deltas(&self, initial: u32, tags: &[u8], data: &[u8], values: &mut [u32]) -> usize;

    /// Returns the data length of all the groups encoded by `tags`.
    fn data_len(&self, tags: &[u8]) -> usize;

    /// Skip `tags.len() * 4` deltas read from input tag and data streams.
    ///
    /// Returns the number of bytes consume from the data stream and the sum of all the deltas that were skipped.
    ///
    /// # Panics
    ///
    ///  - If decoding would consume bytes past the end of `data`.
    fn skip_deltas(&self, tags: &[u8], data: &[u8]) -> (usize, u32);
}

mod group1234;
pub use group1234::Group1234;

mod group0124;
pub use group0124::Group0124;

/// `Group64` compresses and decompresses groups of 4 64-bit integers together in a byte-aligned format.
/// Different `Group64` implementations may support different byte length distributions.
///
/// Groups are coded into two separate streams: a tag stream where each byte describes the contents of a group, and a
/// data stream containing values as described by the tag. Use `max_compressed_bytes()` to compute the number of tag
/// and data bytes that must be allocated to safely encode a slice of input values.
pub trait Group64: Sized + Copy + Clone {
    /// Create a new Group64 coder, selecting the fastest implementation available (ideally SIMD accelerated).
    ///
    /// These objects should be relatively cheap to create and require no heap allocation.
    fn new() -> Self;

    /// Returns the number of `(tag_bytes, data_bytes)` required to compress a slice of length `len`.
    fn max_compressed_bytes(len: usize) -> (usize, usize) {
        let num_groups = (len + 3) / 4;
        (num_groups, num_groups * 4 * std::mem::size_of::<u64>())
    }

    /// Encodes a slice of values, writing tags and data to separate streams.
    ///
    /// For every 4 input values, one tag byte and up to 32 data bytes may be written to the output streams.
    ///
    /// Returns the number of bytes written to the data stream.
    ///
    /// # Panics
    ///
    /// - If `values.len() % 4 != 0`
    /// - If `tags` or `data` are too small to fit all of the output data.
    fn encode(&self, values: &[u64], tags: &mut [u8], data: &mut [u8]) -> usize;

    /// Encodes a slice of values, writing tags and data to separate streams.
    /// Values are interpreted as deltas starting from `initial` which produces a more compact output that is also more
    /// expensive to encode and decode.
    ///
    /// For every 4 input values, one tag byte and up to 32 data bytes may be written to the output streams.
    ///
    /// Returns the number of bytes written to the data stream.
    ///
    /// # Panics
    ///
    /// - If `values.len() % 4 != 0`
    /// - If `tags` or `data` are too small to fit all of the output data.
    fn encode_deltas(
        &self,
        initial: u64,
        values: &[u64],
        tags: &mut [u8],
        data: &mut [u8],
    ) -> usize;

    /// Decodes input tag and data streams to an output slice.
    ///
    /// Returns the number of bytes consumed from the data stream, and one tag byte is consumed for every 4 values decoded.
    ///
    /// # Panics
    ///
    /// - If `values.len() % 4 != 0`.
    /// - If `tags.len() < values.len() / 4`.
    /// - If decoding would consume bytes past the end of `data`.
    fn decode(&self, tags: &[u8], data: &[u8], values: &mut [u64]) -> usize;

    /// Decodes input tag and data streams to an output slice. Values are interepreted as deltas starting from `initial`.
    ///
    /// Returns the number of bytes consumed from the data stream, and one tag byte is consumed for every 4 values decoded.
    ///
    /// # Panics
    ///
    /// - If `values.len() % 4 != 0`.
    /// - If `tags.len() < values.len() / 4`.
    /// - If decoding would consume bytes past the end of `data`.
    fn decode_deltas(&self, initial: u64, tags: &[u8], data: &[u8], values: &mut [u64]) -> usize;

    /// Returns the data length of all the groups encoded by `tags`.
    fn data_len(&self, tags: &[u8]) -> usize;

    /// Skip `tags.len() * 4` deltas read from input tag and data streams.
    ///
    /// Returns the number of bytes consume from the data stream and the sum of all the deltas that were skipped.
    ///
    /// # Panics
    ///
    ///  - If decoding would consume bytes past the end of `data`.
    fn skip_deltas(&self, tags: &[u8], data: &[u8]) -> (usize, u64);
}

mod group1248;
pub use group1248::Group1248;

#[cfg(test)]
pub(crate) mod tests;
