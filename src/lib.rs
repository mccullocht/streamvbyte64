/*! # Fast Byte-Aligned Integer Coding
This crate is a Rust port of [Daniel Lemire's `streamvbyte` library](https://github.com/lemire/streamvbyte).
It contains multiple implementations of this format aimed at different integer sizes and value distributions.

Each `Coder` implementation produces different formats that are incompatible with one another. Names
provide the length of each of the 4 possible tags for each value so `Coder1234` encodes each entry
as 1, 2, 3, or 4 bytes. A scalar implementation is always available at a large speed penalty but
the implementation will automatically use an accelerated implementation for the target if available.

At the moment group implementations only have acceleration on little-endian `aarch64` targets with
`NEON` instruction support.

## Example without delta-coding

```
use streamvbyte64::{Coder, Coder1234};

let coder = Coder1234::new();
let values = vec![
    0u32, 128, 256, 1024, 70, 36, 1000000,
    378, 45, 888, 26, 262144, 88, 89, 90, 16777216
];
let (tag_len, data_len) = Coder1234::max_compressed_bytes(values.len());
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
use streamvbyte64::{Coder, Coder1234};

let coder = Coder1234::new();
let mut sum = 0u32;
let values = [
    0u32, 128, 256, 1024, 70, 36, 1000000,
    378, 45, 888, 26, 262144, 88, 89, 90, 16777216
].iter().map(|x| {
    sum += x;
    sum
}).collect::<Vec<_>>();
let (tag_len, data_len) = Coder1234::max_compressed_bytes(values.len());
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
mod coder_impl;
mod coding_descriptor;
mod raw_group;
mod tag_utils;

mod coder0124;
mod coder1234;
mod coder1248;

pub use num_traits::{ops::wrapping::WrappingAdd, ops::wrapping::WrappingSub, PrimInt};

/// `Coder` compresses and decompresses integers in a byte-aligned format compose of two streams.
///
/// Groups of 4 integers are coded into two separate streams: a tag stream where each byte describes
/// a group, and a data stream containing values as described by the tag. The coder _does not_
/// record the number of entries in the stream, instead assuming a multiple of 4.
///
/// Different coder implementations support different integer widths (32 or 64 bit) as well as
/// different byte length distributions to better compress some data sets.
///
/// Use `max_compressed_bytes()` to compute the number of tag and data bytes that must be allocated
/// to safely encode a slice of input values.
pub trait Coder: Sized + Copy + Clone {
    /// The input/output element type for this coder, typically `u32` or `u64`.
    type Elem: PrimInt + WrappingAdd + WrappingSub + std::fmt::Debug + Sized + Copy + Clone;

    /// Create a new `Coder`, selecting the fastest implementation available.
    ///
    /// These objects should be relatively cheap to create and require no heap allocation.
    fn new() -> Self;

    /// Returns the number of `(tag_bytes, data_bytes)` required to compress a slice of length `len`.
    fn max_compressed_bytes(len: usize) -> (usize, usize) {
        let num_groups = (len + 3) / 4;
        (
            num_groups,
            num_groups * 4 * std::mem::size_of::<Self::Elem>(),
        )
    }

    /// Encodes a slice of values, writing tags and data to separate streams.
    ///
    /// For every 4 input values one tag byte and up to `std::mem::size_of::<Elem>() * 4` data bytes
    /// may be written to output.
    ///
    /// Returns the number of bytes written to the data stream.
    ///
    /// # Panics
    ///
    /// - If `values.len() % 4 != 0`
    /// - If `tags` or `data` are too small to fit all of the output data.
    fn encode(&self, values: &[Self::Elem], tags: &mut [u8], data: &mut [u8]) -> usize;

    /// Encodes a slice of values, writing tags and data to separate streams.
    ///
    /// Values are interpreted as deltas starting from `initial` which produces a more compact
    /// output that is also more expensive to encode and decode.
    ///
    /// For every 4 input values one tag byte and up to `std::mem::size_of::<Elem>() * 4` data bytes
    /// may be written to output.
    ///
    /// Returns the number of bytes written to the data stream.
    ///
    /// # Panics
    ///
    /// - If `values.len() % 4 != 0`
    /// - If `tags` or `data` are too small to fit all of the output data.
    fn encode_deltas(
        &self,
        initial: Self::Elem,
        values: &[Self::Elem],
        tags: &mut [u8],
        data: &mut [u8],
    ) -> usize;

    /// Decodes input tags and data streams to an output slice.
    ///
    /// Consumes all tag values in the input stream to produce `tags.len() * 4` values. May consume
    /// up to `std::mem::size_of<Elem>() * tags.len() * 4` bytes from data.
    ///
    /// Returns the number of bytes consumed from the data stream.
    ///
    /// # Panics
    ///
    /// - If `values.len() % 4 != 0`.
    /// - If `tags.len() < values.len() / 4`.
    /// - If decoding would consume bytes past the end of `data`.
    fn decode(&self, tags: &[u8], data: &[u8], values: &mut [Self::Elem]) -> usize;

    /// Decodes input tags and data streams to an output slice.
    ///
    /// Values are interepreted as deltas starting from `initial`.
    ///
    /// Consumes all tag values in the input stream to produce `tags.len() * 4` values. May consume
    /// up to `std::mem::size_of<Elem>() * tags.len() * 4` bytes from data.
    ///
    /// Returns the number of bytes consumed from the data stream.
    ///
    /// # Panics
    ///
    /// - If `values.len() % 4 != 0`.
    /// - If `tags.len() < values.len() / 4`.
    /// - If decoding would consume bytes past the end of `data`.
    fn decode_deltas(
        &self,
        initial: Self::Elem,
        tags: &[u8],
        data: &[u8],
        values: &mut [Self::Elem],
    ) -> usize;

    /// Returns the data length of all the groups encoded by `tags`.
    fn data_len(&self, tags: &[u8]) -> usize;

    /// Skip `tags.len() * 4` deltas read from input tag and data streams.
    ///
    /// Returns the number of bytes consumed from the data stream and the sum of all the deltas that
    /// were skipped.
    ///
    /// # Panics
    ///
    ///  - If decoding would consume bytes past the end of `data`.
    fn skip_deltas(&self, tags: &[u8], data: &[u8]) -> (usize, Self::Elem);
}

pub use coder0124::Coder0124;
pub use coder1234::Coder1234;
pub use coder1248::Coder1248;

#[cfg(test)]
pub(crate) mod tests;
