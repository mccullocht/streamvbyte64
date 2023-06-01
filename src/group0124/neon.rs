use super::scalar;
use crate::arch::neon::{data_len8, tag_decode_shuffle_table32, tag_encode_shuffle_table32};
use crate::raw_group::RawGroup;
use std::arch::aarch64::{
    uint32x4_t, vaddq_u32, vaddvq_u32, vaddvq_u8, vclzq_u32, vdupq_laneq_u32, vdupq_n_u32,
    vextq_u32, vld1q_s32, vld1q_u32, vld1q_u8, vqtbl1q_u8, vreinterpretq_u32_u8,
    vreinterpretq_u8_u32, vshlq_u32, vshrq_n_u32, vst1q_u32, vst1q_u8, vsubq_u32,
};

const ENCODE_TABLE: [[u8; 16]; 256] = tag_encode_shuffle_table32(super::TAG_LEN);
const DECODE_TABLE: [[u8; 16]; 256] = tag_decode_shuffle_table32(super::TAG_LEN);

#[derive(Clone, Copy, Debug)]
pub(crate) struct RawGroupImpl(uint32x4_t);

impl RawGroup for RawGroupImpl {
    type Elem = u32;
    const TAG_LEN: [usize; 4] = super::TAG_LEN;

    #[inline]
    fn set1(value: u32) -> Self {
        RawGroupImpl(unsafe { vdupq_n_u32(value) })
    }

    #[inline]
    unsafe fn load_unaligned(ptr: *const u32) -> Self {
        RawGroupImpl(vld1q_u32(ptr))
    }

    #[inline]
    unsafe fn store_unaligned(ptr: *mut u32, group: Self) {
        vst1q_u32(ptr, group.0)
    }

    #[inline]
    unsafe fn encode(output: *mut u8, group: Self) -> (u8, usize) {
        // Value tags are computed using the same algorithm as scalar but vector parallel.
        let clz_bytes = vsubq_u32(vdupq_n_u32(4), vshrq_n_u32(vclzq_u32(group.0), 3));
        let value_tags = vqtbl1q_u8(
            vld1q_u8([0, 1, 2, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0].as_ptr()),
            vreinterpretq_u8_u32(clz_bytes),
        );
        // Shift value tags so that they do not overlap and sum them to get the tag.
        let tag = vaddvq_u32(vshlq_u32(
            vreinterpretq_u32_u8(value_tags),
            vld1q_s32([0, 2, 4, 6].as_ptr()),
        )) as u8;
        let written = vaddvq_u8(vqtbl1q_u8(
            vld1q_u8([0, 1, 2, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0].as_ptr()),
            value_tags,
        )) as usize;

        // Use a precomputed table that shuffles all the bytes ientified by the tag as close together as possible.
        // This will write 16 bytes but everything beyond written will be 0.
        vst1q_u8(
            output,
            vqtbl1q_u8(
                vreinterpretq_u8_u32(group.0),
                vld1q_u8(ENCODE_TABLE[tag as usize].as_ptr()),
            ),
        );

        (tag, written)
    }

    #[inline]
    unsafe fn encode_deltas(output: *mut u8, base: Self, group: Self) -> (u8, usize) {
        let deltas = RawGroupImpl(vsubq_u32(group.0, vextq_u32(base.0, group.0, 3)));
        Self::encode(output, deltas)
    }

    #[inline]
    unsafe fn decode(input: *const u8, tag: u8) -> (usize, Self) {
        // Use a precomputed table that shuffles the minimally packed encoded bytes into the right place.
        let v = vreinterpretq_u32_u8(vqtbl1q_u8(
            vld1q_u8(input),
            vld1q_u8(DECODE_TABLE[tag as usize].as_ptr()),
        ));
        (Self::data_len(tag), RawGroupImpl(v))
    }

    #[inline]
    unsafe fn decode_deltas(input: *const u8, tag: u8, base: Self) -> (usize, Self) {
        let (read, group) = Self::decode(input, tag);
        let a_b_c_d = group.0;
        let z_z_z_z = vdupq_n_u32(0);
        let p_p_p_p = vdupq_laneq_u32(base.0, 3);
        let z_a_b_c = vextq_u32(z_z_z_z, a_b_c_d, 3);
        let a_ab_bc_cd = vaddq_u32(a_b_c_d, z_a_b_c);
        let z_z_a_ab = vextq_u32(z_z_z_z, a_ab_bc_cd, 2);
        let pa_pab_pbc_pbd = vaddq_u32(p_p_p_p, a_ab_bc_cd);
        (read, RawGroupImpl(vaddq_u32(pa_pab_pbc_pbd, z_z_a_ab)))
    }

    #[inline]
    fn data_len(tag: u8) -> usize {
        scalar::RawGroupImpl::data_len(tag)
    }

    #[inline]
    unsafe fn skip_deltas(input: *const u8, tag: u8) -> (usize, u32) {
        let (read, group) = Self::decode(input, tag);
        (read, vaddvq_u32(group.0))
    }

    #[inline]
    fn data_len8(tag8: u64) -> usize {
        data_len8(Self::TAG_LEN, tag8)
    }
}

#[cfg(test)]
crate::tests::raw_group_test_suite!();

#[cfg(test)]
crate::tests::compat_test_suite!();
