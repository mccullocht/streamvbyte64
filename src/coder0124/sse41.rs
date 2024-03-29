use crunchy::unroll;

use super::{scalar, CodingDescriptor0124};
use crate::arch::shuffle::decode_shuffle_entry;
use crate::raw_group::RawGroup;
use crate::{arch::shuffle::encode_shuffle_entry, coding_descriptor::CodingDescriptor};
use std::arch::x86_64::{
    __m128i, _mm_add_epi32, _mm_alignr_epi8, _mm_bslli_si128, _mm_loadu_si128, _mm_min_epi16,
    _mm_min_epu8, _mm_movemask_epi8, _mm_packus_epi32, _mm_set1_epi16, _mm_set1_epi32,
    _mm_set1_epi8, _mm_shuffle_epi32, _mm_shuffle_epi8, _mm_storeu_si128, _mm_sub_epi32,
};

const ELEM_LEN: usize = std::mem::size_of::<u32>();
const GROUP_LEN: usize = ELEM_LEN * 4;
const ENCODE_TABLE: [[u8; 16]; 256] = {
    let mut table = [[0u8; 16]; 256];
    let mut tag = 0;
    while tag < 256 {
        table[tag] = encode_shuffle_entry::<ELEM_LEN, GROUP_LEN>(
            tag as u8,
            CodingDescriptor0124::TAG_LEN,
            128,
        );
        tag += 1;
    }
    table
};
const DECODE_TABLE: [[u8; 16]; 256] = {
    let mut table = [[0u8; 16]; 256];
    let mut tag = 0;
    while tag < 256 {
        table[tag] = decode_shuffle_entry::<ELEM_LEN, GROUP_LEN>(
            tag as u8,
            CodingDescriptor0124::TAG_LEN,
            128,
        );
        tag += 1;
    }
    table
};

#[derive(Clone, Copy, Debug)]
pub(crate) struct RawGroupImpl(__m128i);

impl RawGroupImpl {
    unsafe fn compute_tag(&self) -> u8 {
        // fill each set byte with 0x01.
        let a = _mm_min_epu8(self.0, _mm_set1_epi8(1));
        // transform 0x0101 => 0x0100. This is important in the low half word to handle tag=2.
        let b = _mm_min_epi16(a, _mm_set1_epi16(0x0100));
        // add to low half work to set hi bit in each byte if lowest bit is set.
        let c = _mm_add_epi32(b, _mm_set1_epi32(0x7f7f));
        // saturating pack to produce a value that can movemask to produce the tag.
        let d = _mm_packus_epi32(c, c);
        _mm_movemask_epi8(d) as u8
    }
}

impl RawGroup for RawGroupImpl {
    type Elem = u32;
    const TAG_LEN: [usize; 4] = CodingDescriptor0124::TAG_LEN;

    #[inline]
    fn set1(value: Self::Elem) -> Self {
        RawGroupImpl(unsafe { _mm_set1_epi32(std::mem::transmute(value)) })
    }

    #[inline]
    unsafe fn load_unaligned(ptr: *const Self::Elem) -> Self {
        RawGroupImpl(_mm_loadu_si128(ptr as *const __m128i))
    }

    #[inline]
    unsafe fn store_unaligned(ptr: *mut Self::Elem, group: Self) {
        _mm_storeu_si128(ptr as *mut __m128i, group.0)
    }

    #[inline]
    unsafe fn encode(output: *mut u8, group: Self) -> (u8, usize) {
        let tag = group.compute_tag();

        _mm_storeu_si128(
            output as *mut __m128i,
            _mm_shuffle_epi8(
                group.0,
                _mm_loadu_si128(ENCODE_TABLE[tag as usize].as_ptr() as *mut __m128i),
            ),
        );

        (tag, Self::data_len(tag))
    }

    #[inline]
    unsafe fn encode_deltas(output: *mut u8, base: Self, group: Self) -> (u8, usize) {
        Self::encode(
            output,
            Self(_mm_sub_epi32(group.0, _mm_alignr_epi8(group.0, base.0, 12))),
        )
    }

    #[inline]
    unsafe fn decode(input: *const u8, tag: u8) -> (usize, Self) {
        let group = _mm_shuffle_epi8(
            _mm_loadu_si128(input as *const __m128i),
            _mm_loadu_si128(DECODE_TABLE[tag as usize].as_ptr() as *const __m128i),
        );
        (Self::data_len(tag), Self(group))
    }

    #[inline]
    unsafe fn decode_deltas(input: *const u8, tag: u8, base: Self) -> (usize, Self) {
        let (len, Self(a_b_c_d)) = Self::decode(input, tag);
        let a_ab_bc_cd = _mm_add_epi32(a_b_c_d, _mm_bslli_si128(a_b_c_d, 4));
        let a_ab_abc_abcd = _mm_add_epi32(a_ab_bc_cd, _mm_bslli_si128(a_ab_bc_cd, 8));
        (
            len,
            Self(_mm_add_epi32(
                a_ab_abc_abcd,
                _mm_shuffle_epi32(base.0, 0xff),
            )),
        )
    }

    #[inline]
    fn data_len(tag: u8) -> usize {
        scalar::RawGroupImpl::data_len(tag)
    }

    #[inline]
    unsafe fn skip_deltas(input: *const u8, tag: u8) -> (usize, Self::Elem) {
        let (len, deltas) = Self::decode(input, tag);
        let mut d = [0u32; 4];
        _mm_storeu_si128(d.as_mut_ptr() as *mut __m128i, deltas.0);
        (len, d.into_iter().fold(0, |s, d| s.wrapping_add(d)))
    }

    #[inline]
    unsafe fn skip_deltas8(input: *const u8, tag8: u64) -> (usize, Self::Elem) {
        let tags = tag8.to_le_bytes();
        let (mut offset, mut delta_sum) = Self::decode(input, tags[0]);
        unroll! {
            for i in 1..8 {
                let (len, deltas) = Self::decode(input.add(offset), tags[i]);
                offset += len;
                delta_sum.0 = _mm_add_epi32(delta_sum.0, deltas.0);
            }
        }
        let mut d = [0u32; 4];
        _mm_storeu_si128(d.as_mut_ptr() as *mut __m128i, delta_sum.0);
        (offset, d.into_iter().fold(0, |s, d| s.wrapping_add(d)))
    }
}

#[cfg(test)]
crate::tests::raw_group_test_suite!();

#[cfg(test)]
crate::tests::compat_test_suite!();
