use crunchy::unroll;

use super::{scalar, CodingDescriptor1234};
use crate::arch::shuffle::decode_shuffle_entry;
use crate::raw_group::RawGroup;
use crate::{arch::shuffle::encode_shuffle_entry, coding_descriptor::CodingDescriptor};
use std::arch::x86_64::{
    __m128i, _mm_add_epi32, _mm_adds_epu16, _mm_alignr_epi8, _mm_bslli_si128, _mm_loadu_si128,
    _mm_min_epi16, _mm_min_epu8, _mm_movemask_epi8, _mm_packus_epi16, _mm_set1_epi16,
    _mm_set1_epi32, _mm_set1_epi8, _mm_shuffle_epi32, _mm_shuffle_epi8, _mm_storeu_si128,
    _mm_sub_epi32,
};

const ENCODE_TABLE: [[u8; 16]; 256] = {
    let mut table = [[0u8; 16]; 256];
    let mut tag = 0;
    while tag < 256 {
        table[tag] = encode_shuffle_entry::<{ std::mem::size_of::<u32>() }, 16>(
            tag as u8,
            CodingDescriptor1234::TAG_LEN,
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
        table[tag] = decode_shuffle_entry::<{ std::mem::size_of::<u32>() }, 16>(
            tag as u8,
            CodingDescriptor1234::TAG_LEN,
            128,
        );
        tag += 1;
    }
    table
};

#[derive(Clone, Copy, Debug)]
pub(crate) struct RawGroupImpl(__m128i);

impl RawGroup for RawGroupImpl {
    type Elem = u32;
    const TAG_LEN: [usize; 4] = CodingDescriptor1234::TAG_LEN;

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
        // This implementation for generating the tag byte came from https://github.com/lemire/streamvbyte/blob/08c60644dc6956182c68c1b453ba5f2d42367823/src/streamvbytedelta_x64_encode.c
        let mask_01 = _mm_set1_epi8(0x1);
        let mask_7f00 = _mm_set1_epi16(0x7f00);

        let a = _mm_min_epu8(mask_01, group.0);
        let b = _mm_packus_epi16(a, a);
        let c = _mm_min_epi16(b, mask_01);
        let d = _mm_adds_epu16(c, mask_7f00);
        let tag = _mm_movemask_epi8(d) as u8;

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

    #[inline]
    fn data_len8(tag8: u64) -> usize {
        let sum4 = ((tag8 >> 2) & 0x3333333333333333) + (tag8 & 0x3333333333333333);
        let sum8 = ((sum4 >> 4) & 0x0f0f0f0f0f0f0f0f) + (sum4 & 0x0f0f0f0f0f0f0f0f);
        let sum16 = ((sum8 >> 8) & 0x00ff00ff00ff00ff) + (sum8 & 0x00ff00ff00ff00ff);
        let sum32 = ((sum16 >> 16) & 0x0000ffff0000ffff) + (sum16 & 0x0000ffff0000ffff);
        let sum64 = ((sum32 >> 32) & 0xffffffff) + (sum32 & 0xffffffff);
        sum64 as usize + 32
    }
}

#[cfg(test)]
crate::tests::raw_group_test_suite!();

#[cfg(test)]
crate::tests::compat_test_suite!();
