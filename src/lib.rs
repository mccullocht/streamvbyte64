mod dist1234;
mod dist1248;
pub(crate) mod group_impl;
mod tag_utils;
mod unsafe_group;

use std::io::{Result, Write};

// TODO: add a GroupTag trait that is shared with UnsafeGroup, it would be useful for testing.

/// Group32 encode and decodes a compact, byte-aligned format where groups of 4 32-bit integers are encoded together.
/// Different Group32 implementations may support different byte length distributions.
///
/// Groups are coded into two separate streams: a tag stream where each byte describes the contents of a group, and a
/// data stream containing values as described by the tag.
pub trait Group32: Sized + Copy + Clone {
    /// Create a new Group32 coder, selecting the fastest implementation available (ideally SIMD accelerated).
    ///
    /// These objects should be relatively cheap to create and require no heap allocation.
    fn new() -> Self;

    /// Encodes a slice of values, writing tags and data to separate streams.
    ///
    /// For every 4 input values, one tag byte and up to 16 data bytes may be written to the output streams.
    /// Returns the number of bytes written to the data stream.
    ///
    /// # Panics
    ///
    /// - If `values.len() % 4 != 0`
    /// - If `tags` or `data` are too small to fit all of the output data.
    fn encode(&self, values: &[u32], tags: &mut [u8], data: &mut [u8]) -> usize;

    /// Encodes a slice of values, writing tags and data to separate streams.
    /// Values are interpreted as deltas starting from `initial` which produces a more compact output that is also more
    /// expensive to encode and decode.
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

    /// Encodes a slice of values, writing tags and data to separate streams.
    ///
    /// For every 4 input values, one tag byte and up to 16 data bytes may be written to the output streams.
    /// Returns the number of bytes written to the data stream.
    ///
    /// # Panics
    ///
    /// - If `values.len() % 4 != 0`
    /// - If `tags` or `data` are too small to fit all of the output data.
    fn encode_to_writer<W: Write>(
        &self,
        values: &[u32],
        tags: &mut [u8],
        data: &mut W,
    ) -> Result<usize> {
        let mut buf = [0u8; 256];
        let mut written = 0usize;
        for (chunk_values, chunk_tags) in values.chunks(32).zip(tags.chunks_mut(8)) {
            let nbuf = self.encode(chunk_values, chunk_tags, &mut buf);
            data.write(&buf[..nbuf])?;
            written += nbuf;
        }
        Ok(written)
    }

    /// Encodes a slice of values, writing tags and data to separate streams.
    /// Values are interpreted as deltas starting from `initial` which produces a more compact output that is also more
    /// expensive to encode and decode.
    ///
    /// For every 4 input values, one tag byte and up to 16 data bytes may be written to the output streams.
    /// Returns the number of bytes written to the data stream.
    ///
    /// # Panics
    ///
    /// - If `values.len() % 4 != 0`
    /// - If `tags` or `data` are too small to fit all of the output data.
    fn encode_deltas_to_writer<W: Write>(
        &self,
        initial: u32,
        values: &[u32],
        tags: &mut [u8],
        data: &mut W,
    ) -> Result<usize> {
        let mut buf = [0u8; 256];
        let mut written = 0usize;
        let mut base = initial;
        for (chunk_values, chunk_tags) in values.chunks(32).zip(tags.chunks_mut(8)) {
            let nbuf = self.encode_deltas(base, chunk_values, chunk_tags, &mut buf);
            data.write(&buf[..nbuf])?;
            written += nbuf;
            base = *chunk_values.last().unwrap(); // chunks() guarantees there will be at least one value.
        }
        Ok(written)
    }

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

    /// Decodes input tag and data streams to an output slice.
    /// Values are interepreted as deltas starting from `initial`.
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

#[cfg(test)]
pub(crate) mod tests;
