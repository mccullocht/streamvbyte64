use super::{scalar, CodingDescriptor1248};
use crate::coding_descriptor::CodingDescriptor;
use crate::raw_group::RawGroup;
use std::arch::x86_64::{
    __m128i, _mm_add_epi32, _mm_loadu_si128, _mm_min_epi16, _mm_min_epi8, _mm_min_epu16,
    _mm_movemask_epi8, _mm_packus_epi16, _mm_packus_epi32, _mm_set1_epi32, _mm_set1_epi64x,
    _mm_shuffle_epi8, _mm_storeu_si128,
};

const fn generate_encode_shuffle_table() -> [[u8; 16]; 16] {
    let mut table = [[128u8; 16]; 16];
    let mut tag = 0usize;
    while tag < 16 {
        let mut shuf_idx = 0;
        let mut i = 0;
        while i < 2 {
            let vtag = (tag >> (i * 2)) & 0x3;
            let mut j = 0;
            while j < (1 << vtag) {
                table[tag][shuf_idx] = (i * 8 + j) as u8;
                shuf_idx += 1;
                j += 1;
            }
            i += 1;
        }
        tag += 1;
    }
    table
}

const ENCODE_TABLE: [[u8; 16]; 16] = generate_encode_shuffle_table();

/// Generate a decode shuffle table reading up to 16 bytes of input and 2 output u64s at a time.
/// This works different from aarch64/NEON as the shuffle instruction does not accept more than
/// 16 bytes of data as input.
/// XXX we could probably allow the function to control the number of input tags.
const fn generate_decode_shuffle_table() -> [[u8; 16]; 16] {
    let mut table = [[128u8; 16]; 16];
    let mut tag = 0usize;
    while tag < 16 {
        let mut shuf_idx = 0;
        let mut i = 0;
        while i < 2 {
            let vtag = (tag >> (i * 2)) & 0x3;
            let mut j = 0;
            while j < (1 << vtag) {
                table[tag][i * 8 + j] = shuf_idx;
                shuf_idx += 1;
                j += 1;
            }
            i += 1;
        }
        tag += 1;
    }
    table
}

const DECODE_TABLE: [[u8; 16]; 16] = generate_decode_shuffle_table();

// XXX this is a duplicate of another generated table.
const NIBBLE_LEN: [usize; 16] = [2, 3, 5, 9, 3, 4, 6, 10, 5, 6, 8, 12, 9, 10, 12, 16];

#[derive(Clone, Copy, Debug)]
pub(crate) struct RawGroupImpl(__m128i, __m128i);

impl RawGroupImpl {
    /// Use a similar approach to 1234 ssse3 impl in that we movemask to generate the tag value.
    /// Do 16-bit saturating narrow, a bit of manipulation, followed by 32-bit saturating narrow.
    /// The bit manipulation ensures that we handle cases with bytes 2-4 set correctly, the 32-bit
    /// narrow will ensure that if bytes 5-8 are set we will set all bits in the output tag.
    #[inline(always)]
    unsafe fn compute_tag(&self) -> u8 {
        let mmask = _mm_set1_epi64x(0x0101010101010100);
        let m = (_mm_min_epi8(self.0, mmask), _mm_min_epi8(self.1, mmask));
        let n1 = _mm_packus_epi16(m.0, m.1);
        let m1 = _mm_min_epu16(n1, _mm_set1_epi32(0x1_0100));
        // map 0x0100 => 0x8000 for movemask
        let m2 = _mm_add_epi32(m1, _mm_set1_epi32(0x7f00));
        let n2 = _mm_packus_epi32(m2, m2);
        _mm_movemask_epi8(n2) as u8
    }

    /// Produce data length for lo and hi nibbles of the input tag.
    #[inline(always)]
    fn nibble_tag_lens(tag: usize) -> (usize, usize) {
        (NIBBLE_LEN[tag & 0xf], NIBBLE_LEN[tag >> 4])
    }
}

impl RawGroup for RawGroupImpl {
    type Elem = u64;
    const TAG_LEN: [usize; 4] = CodingDescriptor1248::TAG_LEN;

    #[inline]
    fn set1(value: Self::Elem) -> Self {
        unsafe {
            let h = _mm_set1_epi64x(std::mem::transmute(value));
            RawGroupImpl(h, h)
        }
    }

    #[inline]
    unsafe fn load_unaligned(ptr: *const Self::Elem) -> Self {
        RawGroupImpl(
            _mm_loadu_si128(ptr as *const __m128i),
            _mm_loadu_si128(ptr.add(2) as *const __m128i),
        )
    }

    #[inline]
    unsafe fn store_unaligned(ptr: *mut Self::Elem, group: Self) {
        _mm_storeu_si128(ptr as *mut __m128i, group.0);
        _mm_storeu_si128(ptr.add(2) as *mut __m128i, group.1);
    }

    #[inline]
    unsafe fn encode(output: *mut u8, group: Self) -> (u8, usize) {
        let tag = group.compute_tag();
        let utag = tag as usize;
        let half_lens = Self::nibble_tag_lens(utag);
        let shuf = (
            _mm_loadu_si128(ENCODE_TABLE[tag as usize & 0xf].as_ptr() as *const __m128i),
            _mm_loadu_si128(ENCODE_TABLE[tag as usize >> 4].as_ptr() as *const __m128i),
        );
        _mm_storeu_si128(output as *mut __m128i, _mm_shuffle_epi8(group.0, shuf.0));
        _mm_storeu_si128(
            output.add(half_lens.0) as *mut __m128i,
            _mm_shuffle_epi8(group.1, shuf.1),
        );
        (tag, half_lens.0 + half_lens.1)
    }

    #[inline]
    unsafe fn encode_deltas(output: *mut u8, base: Self, group: Self) -> (u8, usize) {
        todo!()
    }

    #[inline]
    unsafe fn decode(input: *const u8, tag: u8) -> (usize, Self) {
        let half_lens = Self::nibble_tag_lens(tag as usize);
        let inputs = (
            _mm_loadu_si128(input as *const __m128i),
            _mm_loadu_si128(input.add(half_lens.0) as *const __m128i),
        );
        let shuf = (
            _mm_loadu_si128(DECODE_TABLE[tag as usize & 0xf].as_ptr() as *const __m128i),
            _mm_loadu_si128(DECODE_TABLE[tag as usize >> 4].as_ptr() as *const __m128i),
        );
        (
            half_lens.0 + half_lens.1,
            RawGroupImpl(
                _mm_shuffle_epi8(inputs.0, shuf.0),
                _mm_shuffle_epi8(inputs.1, shuf.1),
            ),
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
