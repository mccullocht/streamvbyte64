use super::{scalar, CodingDescriptor1234};
use crate::coding_descriptor::CodingDescriptor;
use crate::raw_group::RawGroup;
use std::arch::x86_64::{
    __m128i, _mm_adds_epu16, _mm_alignr_epi8, _mm_loadu_si128, _mm_min_epi16, _mm_min_epu8,
    _mm_movemask_epi8, _mm_packus_epi16, _mm_set1_epi16, _mm_set1_epi32, _mm_set1_epi8,
    _mm_shuffle_epi8, _mm_storeu_si128, _mm_sub_epi32,
};

// XXX share this with the neon iplementation.
pub(crate) const fn generate_encode_table() -> [[u8; 16]; 256] {
    let tag_len = [1usize, 2, 3, 4];
    // Default fill with 128 because pshufb will zero fill anything with the hi bit set.
    let mut table = [[128u8; 16]; 256];
    let mut tag = 0usize;
    while tag < 256 {
        let mut shuf_idx = 0;
        let mut i = 0;
        while i < 4 {
            let vtag = (tag >> (i * 2)) & 0x3;
            let mut j = 0;
            while j < tag_len[vtag as usize] {
                table[tag][shuf_idx] = (i * 4 + j) as u8;
                shuf_idx += 1;
                j += 1;
            }
            i += 1;
        }
        tag += 1;
    }
    table
}

const ENCODE_TABLE: [[u8; 16]; 256] = generate_encode_table();

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
        // XXX This came from the implementation in lemire's repository.
        // TODO: add encode2() to interface as we could easily compute two tags at once.
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

    #[inline]
    unsafe fn decode8(input: *const u8, tag8: u64, output: *mut Self::Elem) -> usize {
        todo!()
    }

    #[inline]
    unsafe fn decode_deltas8(
        input: *const u8,
        tag8: u64,
        base: Self,
        output: *mut Self::Elem,
    ) -> (usize, Self) {
        todo!()
    }

    #[inline]
    unsafe fn skip_deltas8(input: *const u8, tag8: u64) -> (usize, Self::Elem) {
        todo!()
    }
}

#[cfg(test)]
crate::tests::raw_group_test_suite!();

#[cfg(test)]
crate::tests::compat_test_suite!();
