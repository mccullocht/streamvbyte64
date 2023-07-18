use crunchy::unroll;

use super::{scalar, CodingDescriptor1248};
use crate::arch::shuffle::{decode_shuffle_entry, encode_shuffle_entry};
use crate::coding_descriptor::CodingDescriptor;
use crate::raw_group::RawGroup;
use std::arch::x86_64::{
    __m128i, _mm_add_epi32, _mm_add_epi64, _mm_alignr_epi8, _mm_bslli_si128, _mm_bsrli_si128,
    _mm_cvtsi128_si64x, _mm_loadu_si128, _mm_min_epu16, _mm_min_epu8, _mm_movemask_epi8,
    _mm_packus_epi16, _mm_packus_epi32, _mm_set1_epi32, _mm_set1_epi64x, _mm_shuffle_epi32,
    _mm_shuffle_epi8, _mm_storeu_si128, _mm_sub_epi64,
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
unsafe fn load_shuffle(table: &[[u8; 16]; 16], nibble_tags: (usize, usize)) -> (__m128i, __m128i) {
    (
        _mm_loadu_si128(table[nibble_tags.0].as_ptr() as *const __m128i),
        _mm_loadu_si128(table[nibble_tags.1].as_ptr() as *const __m128i),
    )
}

// TODO: this could be shared with generate_nibble_tag_len_table().
const NIBBLE_LEN: [usize; 16] = [2, 3, 5, 9, 3, 4, 6, 10, 5, 6, 8, 12, 9, 10, 12, 16];

#[derive(Clone, Copy, Debug)]
pub(crate) struct RawGroupImpl(__m128i, __m128i);

impl RawGroupImpl {
    /// Use a similar approach to 1234 ssse3 impl in that we movemask to generate the tag value.
    /// Do 16-bit saturating narrow, a bit of manipulation, followed by 32-bit saturating narrow.
    /// The bit manipulation ensures that we handle cases with bytes 2-4 set correctly, the 32-bit
    /// narrow will ensure that if bytes 5-8 are set we will set all bits in the output tag.
    #[inline(always)]
    unsafe fn compute_tag(&self) -> usize {
        let mmask = _mm_set1_epi64x(0x0101010101010100);
        let m = (_mm_min_epu8(self.0, mmask), _mm_min_epu8(self.1, mmask));
        let n1 = _mm_packus_epi16(m.0, m.1);
        let m1 = _mm_min_epu16(n1, _mm_set1_epi32(0x1_0100));
        // map 0x0100 => 0x8000 for movemask
        let m2 = _mm_add_epi32(m1, _mm_set1_epi32(0x7f00));
        let n2 = _mm_packus_epi32(m2, m2);
        _mm_movemask_epi8(n2) as usize & 0xff
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
        let nibble_tags = Self::nibble_tags(tag);
        let nibble_data_len = Self::nibble_data_len(nibble_tags);
        let shuf = load_shuffle(&ENCODE_TABLE, nibble_tags);
        _mm_storeu_si128(output as *mut __m128i, _mm_shuffle_epi8(group.0, shuf.0));
        _mm_storeu_si128(
            output.add(nibble_data_len.0) as *mut __m128i,
            _mm_shuffle_epi8(group.1, shuf.1),
        );
        (tag as u8, nibble_data_len.0 + nibble_data_len.1)
    }

    #[inline]
    unsafe fn encode_deltas(output: *mut u8, base: Self, group: Self) -> (u8, usize) {
        let delta_base = (
            _mm_alignr_epi8::<8>(group.0, base.1),
            _mm_alignr_epi8::<8>(group.1, group.0),
        );
        let delta_group = RawGroupImpl(
            _mm_sub_epi64(group.0, delta_base.0),
            _mm_sub_epi64(group.1, delta_base.1),
        );
        Self::encode(output, delta_group)
    }

    #[inline]
    unsafe fn decode(input: *const u8, tag: u8) -> (usize, Self) {
        let nibble_tags = Self::nibble_tags(tag as usize);
        let nibble_data_len = Self::nibble_data_len(nibble_tags);
        let inputs = (
            _mm_loadu_si128(input as *const __m128i),
            _mm_loadu_si128(input.add(nibble_data_len.0) as *const __m128i),
        );
        let shuf = load_shuffle(&DECODE_TABLE, nibble_tags);
        (
            nibble_data_len.0 + nibble_data_len.1,
            RawGroupImpl(
                _mm_shuffle_epi8(inputs.0, shuf.0),
                _mm_shuffle_epi8(inputs.1, shuf.1),
            ),
        )
    }

    #[inline]
    unsafe fn decode_deltas(input: *const u8, tag: u8, base: Self) -> (usize, Self) {
        let delta_base = _mm_shuffle_epi32(base.1, 0b11101110);
        let (len, Self(a_b, c_d)) = Self::decode(input, tag);
        let z_a = _mm_bslli_si128(a_b, 8);
        let b_c = _mm_alignr_epi8(c_d, a_b, 8);
        let a_ab = _mm_add_epi64(z_a, a_b);
        let pa_pab = _mm_add_epi64(delta_base, a_ab);
        let bc_cd = _mm_add_epi64(b_c, c_d);
        (len, RawGroupImpl(pa_pab, _mm_add_epi64(pa_pab, bc_cd)))
    }

    #[inline]
    fn data_len(tag: u8) -> usize {
        scalar::RawGroupImpl::data_len(tag)
    }

    #[inline]
    unsafe fn skip_deltas(input: *const u8, tag: u8) -> (usize, Self::Elem) {
        let (len, Self(a_b, c_d)) = Self::decode(input, tag);
        let ac_bd = _mm_add_epi64(a_b, c_d);
        let abcd_bd = _mm_add_epi64(_mm_bsrli_si128::<8>(ac_bd), ac_bd);
        (len, std::mem::transmute(_mm_cvtsi128_si64x(abcd_bd)))
    }

    #[inline]
    unsafe fn skip_deltas8(input: *const u8, tag8: u64) -> (usize, Self::Elem) {
        let tags = tag8.to_le_bytes();
        let (mut offset, group) = Self::decode(input, tags[0]);
        let mut sum_delta = _mm_add_epi64(group.0, group.1);
        unroll! {
            for i in 1..8 {
                let (len, group) = Self::decode(input.add(offset), tags[i]);
                offset += len;
                sum_delta = _mm_add_epi64(sum_delta, group.0);
                sum_delta = _mm_add_epi64(sum_delta, group.1);
            }
        }
        sum_delta = _mm_add_epi64(sum_delta, _mm_bsrli_si128(sum_delta, 8));
        (offset, std::mem::transmute(_mm_cvtsi128_si64x(sum_delta)))
    }
}

#[cfg(test)]
crate::tests::raw_group_test_suite!();

#[cfg(test)]
crate::tests::compat_test_suite!();
