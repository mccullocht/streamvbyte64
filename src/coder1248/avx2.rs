use crunchy::unroll;

use super::{scalar, CodingDescriptor1248};
use crate::arch::shuffle::{decode_shuffle_entry, encode_shuffle_entry};
use crate::coding_descriptor::CodingDescriptor;
use crate::raw_group::RawGroup;
use std::arch::x86_64::{
    __m128i, __m256i, _mm256_add_epi32, _mm256_loadu2_m128i, _mm256_loadu_si256, _mm256_min_epu16,
    _mm256_min_epu8, _mm256_movemask_epi8, _mm256_packus_epi16, _mm256_packus_epi32,
    _mm256_permute4x64_epi64, _mm256_set1_epi32, _mm256_set1_epi64x, _mm256_shuffle_epi8,
    _mm256_storeu2_m128i, _mm256_storeu_si256,
};

/// Build a 16 entry encode table processing half of the input group (2 entries).
/// This is necessary because pshufb can only address 16 bytes of input at a time.
const ENCODE_TABLE: [[u8; 16]; 16] = {
    let mut table = [[0u8; 16]; 16];
    let mut tag = 0usize;
    while tag < 16 {
        table[tag] = encode_shuffle_entry::<{ std::mem::size_of::<u64>() }, 16>(
            tag as u8,
            CodingDescriptor1248::TAG_LEN,
            128,
        );
        tag += 1;
    }
    table
};

/// Build a 16 entry decode table processing half of the input group (2 entries).
/// This is necessary because pshufb can only address 16 bytes of input at a time.
const DECODE_TABLE: [[u8; 16]; 16] = {
    let mut table = [[0u8; 16]; 16];
    let mut tag = 0usize;
    while tag < 16 {
        table[tag] = decode_shuffle_entry::<{ std::mem::size_of::<u64>() }, 16>(
            tag as u8,
            CodingDescriptor1248::TAG_LEN,
            128,
        );
        tag += 1;
    }
    table
};

#[inline(always)]
unsafe fn load_shuffle(table: &[[u8; 16]; 16], nibble_tags: (usize, usize)) -> __m256i {
    _mm256_loadu2_m128i(
        table[nibble_tags.1].as_ptr() as *const __m128i,
        table[nibble_tags.0].as_ptr() as *const __m128i,
    )
}

// TODO: this could be shared with generate_nibble_tag_len_table().
const NIBBLE_LEN: [usize; 16] = [2, 3, 5, 9, 3, 4, 6, 10, 5, 6, 8, 12, 9, 10, 12, 16];

#[derive(Clone, Copy, Debug)]
pub(crate) struct RawGroupImpl(__m256i);

impl RawGroupImpl {
    /// Use a similar approach to 1234 ssse3 impl in that we movemask to generate the tag value.
    /// Do 16-bit saturating narrow, a bit of manipulation, followed by 32-bit saturating narrow.
    /// The bit manipulation ensures that we handle cases with bytes 2-4 set correctly, the 32-bit
    /// narrow will ensure that if bytes 5-8 are set we will set all bits in the output tag.
    #[inline(always)]
    unsafe fn compute_tag(&self) -> usize {
        let m = _mm256_min_epu8(self.0, _mm256_set1_epi64x(0x0101010101010100));
        // NB: this interleaves output from the arguments such that the the first two lanes are
        // identical. Undo this with the permute below.
        let n1 = _mm256_packus_epi16(m, m);
        let m1 = _mm256_min_epu16(n1, _mm256_set1_epi32(0x1_0100));
        // map 0x0100 => 0x8000 for movemask
        let m2 = _mm256_add_epi32(m1, _mm256_set1_epi32(0x7f00));
        let m3 = _mm256_permute4x64_epi64(m2, 0b00001000);
        let n2 = _mm256_packus_epi32(m3, m3);
        _mm256_movemask_epi8(n2) as usize & 0xff
    }

    /// Splits input 8-bit `tag` into two nibble-length tags covering two entries instead of 4.
    #[inline(always)]
    fn nibble_tags(tag: usize) -> (usize, usize) {
        debug_assert!(tag < 256);
        (tag & 0xf, tag >> 4)
    }

    /// Return the data length for each nibble tag.
    #[inline(always)]
    fn nibble_data_len(nibble_tags: (usize, usize)) -> (usize, usize) {
        (NIBBLE_LEN[nibble_tags.0], NIBBLE_LEN[nibble_tags.1])
    }
}

impl RawGroup for RawGroupImpl {
    type Elem = u64;
    const TAG_LEN: [usize; 4] = CodingDescriptor1248::TAG_LEN;

    #[inline]
    fn set1(value: Self::Elem) -> Self {
        unsafe { RawGroupImpl(_mm256_set1_epi64x(std::mem::transmute(value))) }
    }

    #[inline]
    unsafe fn load_unaligned(ptr: *const Self::Elem) -> Self {
        RawGroupImpl(_mm256_loadu_si256(ptr as *const __m256i))
    }

    #[inline]
    unsafe fn store_unaligned(ptr: *mut Self::Elem, group: Self) {
        _mm256_storeu_si256(ptr as *mut __m256i, group.0)
    }

    #[inline]
    unsafe fn encode(output: *mut u8, group: Self) -> (u8, usize) {
        let tag = group.compute_tag();
        let nibble_tags = Self::nibble_tags(tag);
        let nibble_data_len = Self::nibble_data_len(nibble_tags);
        let shuf = load_shuffle(&ENCODE_TABLE, nibble_tags);
        _mm256_storeu2_m128i(
            output.add(nibble_data_len.0) as *mut __m128i,
            output as *mut __m128i,
            _mm256_shuffle_epi8(group.0, shuf),
        );
        (tag as u8, nibble_data_len.0 + nibble_data_len.1)
    }

    #[inline]
    unsafe fn encode_deltas(output: *mut u8, base: Self, group: Self) -> (u8, usize) {
        todo!()
    }

    #[inline]
    unsafe fn decode(input: *const u8, tag: u8) -> (usize, Self) {
        let nibble_tags = Self::nibble_tags(tag as usize);
        let nibble_data_len = Self::nibble_data_len(nibble_tags);
        let input = _mm256_loadu2_m128i(
            input.add(nibble_data_len.0) as *const __m128i,
            input as *const __m128i,
        );
        let shuf = load_shuffle(&DECODE_TABLE, nibble_tags);
        (
            nibble_data_len.0 + nibble_data_len.1,
            RawGroupImpl(_mm256_shuffle_epi8(input, shuf)),
        )
    }

    #[inline]
    unsafe fn decode_deltas(input: *const u8, tag: u8, base: Self) -> (usize, Self) {
        todo!()
    }

    #[inline]
    fn data_len(tag: u8) -> usize {
        scalar::RawGroupImpl::data_len(tag)
    }

    #[inline]
    unsafe fn skip_deltas(input: *const u8, tag: u8) -> (usize, Self::Elem) {
        todo!()
    }
}

#[cfg(test)]
crate::tests::raw_group_test_suite!();

#[cfg(test)]
crate::tests::compat_test_suite!();
