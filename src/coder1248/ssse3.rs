use super::{scalar, CodingDescriptor1248};
use crate::coding_descriptor::CodingDescriptor;
use crate::raw_group::RawGroup;
use std::arch::x86_64::{__m128i, _mm_loadu_si128, _mm_set1_epi64x, _mm_storeu_si128};

//const ENCODE_TABLE: [[u8; 32]; 256] =
//    crate::arch::tag_encode_shuffle_table64(RawGroupImpl::TAG_LEN);
//const DECODE_TABLE: [[u8; 32]; 256] =
//    crate::arch::tag_decode_shuffle_table64(RawGroupImpl::TAG_LEN);

#[derive(Clone, Copy, Debug)]
pub(crate) struct RawGroupImpl(__m128i, __m128i);

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
        todo!()
    }

    #[inline]
    unsafe fn encode_deltas(output: *mut u8, base: Self, group: Self) -> (u8, usize) {
        todo!()
    }

    #[inline]
    unsafe fn decode(input: *const u8, tag: u8) -> (usize, Self) {
        todo!()
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
