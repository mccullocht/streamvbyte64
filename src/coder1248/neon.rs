use crunchy::unroll;

use super::{scalar, CodingDescriptor1248};
use crate::arch::neon::{data_len8, tag_decode_shuffle_table64, tag_encode_shuffle_table64};
use crate::coding_descriptor::CodingDescriptor;
use crate::raw_group::RawGroup;
use std::arch::aarch64::{
    uint32x4_t, uint64x2_t, uint8x16_t, uint8x16x2_t, vaddl_high_u32, vaddl_u32, vaddlvq_u32,
    vaddq_u32, vaddq_u64, vaddvq_u32, vclzq_u32, vdupq_laneq_u64, vdupq_n_u32, vdupq_n_u64,
    vextq_u32, vextq_u64, vget_low_u32, vgetq_lane_u64, vld1q_s32, vld1q_u64, vld1q_u8,
    vmovn_high_u64, vmovn_u64, vpaddd_u64, vqmovn_high_u64, vqmovn_u64, vqtbl1q_u8, vqtbl2q_u8,
    vreinterpretq_s32_u8, vreinterpretq_u32_u64, vreinterpretq_u32_u8, vreinterpretq_u64_u8,
    vreinterpretq_u8_u32, vreinterpretq_u8_u64, vshlq_u32, vshrq_n_u32, vst1q_u64, vst1q_u8,
    vsubq_u64, vuzp2q_u32,
};

const ENCODE_TABLE: [[u8; 32]; 256] = tag_encode_shuffle_table64(CodingDescriptor1248::TAG_LEN);
const DECODE_TABLE: [[u8; 32]; 256] = tag_decode_shuffle_table64(CodingDescriptor1248::TAG_LEN);

/// Load a single 32-byte entry from table based on tag.
#[inline(always)]
unsafe fn load_shuffle(table: &[[u8; 32]; 256], tag: u8) -> (uint8x16_t, uint8x16_t) {
    let ptr = table[tag as usize].as_ptr();
    (vld1q_u8(ptr), vld1q_u8(ptr.add(16)))
}

/// Load a single 32-byte decode shuffle table entry based on tag, then narrow it to a 16-byte value.
/// Only use this if tag is set such that there are no 8-byte entries -- in this case it is valid
/// to shuffle into 4 32-bit entries in 1 register rather than 4 64-bit entries in 2 registers.
#[inline(always)]
unsafe fn load_decode_shuffle_narrow(tag: u8) -> uint8x16_t {
    let wshuf = load_shuffle(&DECODE_TABLE, tag);
    vreinterpretq_u8_u32(vmovn_high_u64(
        vmovn_u64(vreinterpretq_u64_u8(wshuf.0)),
        vreinterpretq_u64_u8(wshuf.1),
    ))
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct RawGroupImpl(uint64x2_t, uint64x2_t);

impl RawGroupImpl {
    #[inline(always)]
    unsafe fn compute_tag(&self) -> (u8, usize) {
        // NEON does not provide clz on 64 bit lanes. Split each entry into hi and lo 32 bits values
        // (produce lo by saturating narrow), clz and sum to help compute the values.
        let lo = vqmovn_high_u64(vqmovn_u64(self.0), self.1);
        let hi = vuzp2q_u32(vreinterpretq_u32_u64(self.0), vreinterpretq_u32_u64(self.1));
        let clz_bytes = vshrq_n_u32(vaddq_u32(vclzq_u32(lo), vclzq_u32(hi)), 3);
        let value_tags = vqtbl1q_u8(
            vld1q_u8([3, 3, 3, 3, 2, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0].as_ptr()),
            vreinterpretq_u8_u32(clz_bytes),
        );
        let tag = vaddvq_u32(vshlq_u32(
            vreinterpretq_u32_u8(value_tags),
            vld1q_s32([0, 2, 4, 6].as_ptr()),
        )) as u8;
        let written =
            vaddvq_u32(vshlq_u32(vdupq_n_u32(1), vreinterpretq_s32_u8(value_tags))) as usize;
        (tag, written)
    }

    #[inline(always)]
    fn has_any_tag3(tag8: u64) -> bool {
        (tag8 & 0x5555555555555555 & (tag8 >> 1)) > 0
    }

    #[inline(always)]
    unsafe fn decode32(input: *const u8, tag: u8) -> (usize, uint32x4_t) {
        (
            Self::data_len(tag),
            vreinterpretq_u32_u8(vqtbl1q_u8(vld1q_u8(input), load_decode_shuffle_narrow(tag))),
        )
    }
}

impl RawGroup for RawGroupImpl {
    type Elem = u64;

    const TAG_LEN: [usize; 4] = CodingDescriptor1248::TAG_LEN;

    #[inline]
    fn set1(value: Self::Elem) -> Self {
        unsafe { RawGroupImpl(vdupq_n_u64(value), vdupq_n_u64(value)) }
    }

    #[inline]
    unsafe fn load_unaligned(ptr: *const Self::Elem) -> Self {
        // NB: there are two intrinsic calls but this should be translated into a single ldp instruction.
        RawGroupImpl(vld1q_u64(ptr), vld1q_u64(ptr.add(2)))
    }

    #[inline]
    unsafe fn store_unaligned(ptr: *mut Self::Elem, group: Self) {
        // NB: there are two intrinsic calls but this should be translated into a single stp instruction.
        vst1q_u64(ptr, group.0);
        vst1q_u64(ptr.add(2), group.1);
    }

    #[inline]
    unsafe fn encode(output: *mut u8, group: Self) -> (u8, usize) {
        let (tag, written) = group.compute_tag();
        let shuf = load_shuffle(&ENCODE_TABLE, tag);
        let tbl_bytes = uint8x16x2_t(vreinterpretq_u8_u64(group.0), vreinterpretq_u8_u64(group.1));
        let out = (vqtbl2q_u8(tbl_bytes, shuf.0), vqtbl2q_u8(tbl_bytes, shuf.1));
        vst1q_u8(output, out.0);
        vst1q_u8(output.add(16), out.1);
        (tag, written)
    }

    #[inline]
    unsafe fn encode_deltas(output: *mut u8, base: Self, group: Self) -> (u8, usize) {
        let b0 = vextq_u64(base.1, group.0, 1);
        let b1 = vextq_u64(group.0, group.1, 1);
        Self::encode(
            output,
            RawGroupImpl(vsubq_u64(group.0, b0), vsubq_u64(group.1, b1)),
        )
    }

    #[inline]
    unsafe fn decode(input: *const u8, tag: u8) -> (usize, Self) {
        let shuf = load_shuffle(&DECODE_TABLE, tag);
        let tbl_bytes = uint8x16x2_t(vld1q_u8(input), vld1q_u8(input.add(16)));
        let group_data = (vqtbl2q_u8(tbl_bytes, shuf.0), vqtbl2q_u8(tbl_bytes, shuf.1));
        (
            Self::data_len(tag),
            RawGroupImpl(
                vreinterpretq_u64_u8(group_data.0),
                vreinterpretq_u64_u8(group_data.1),
            ),
        )
    }

    #[inline]
    unsafe fn decode_deltas(input: *const u8, tag: u8, base: Self) -> (usize, Self) {
        let (len, Self(a_b, c_d)) = Self::decode(input, tag);
        let z = vdupq_n_u64(0);
        let z_a = vextq_u64(z, a_b, 1);
        let a_ab = vaddq_u64(z_a, a_b);
        let p = vdupq_laneq_u64(base.1, 1);
        let pa_pab = vaddq_u64(p, a_ab);
        let b_c = vextq_u64(a_b, c_d, 1);
        let bc_cd = vaddq_u64(b_c, c_d);
        let pabc_pabcd = vaddq_u64(pa_pab, bc_cd);
        (len, RawGroupImpl(pa_pab, pabc_pabcd))
    }

    #[inline]
    fn data_len(tag: u8) -> usize {
        scalar::RawGroupImpl::data_len(tag)
    }

    #[inline]
    unsafe fn skip_deltas(input: *const u8, tag: u8) -> (usize, Self::Elem) {
        let (len, group) = Self::decode(input, tag);
        let half = vaddq_u64(group.0, group.1);
        (len, vpaddd_u64(half))
    }

    #[inline]
    fn data_len8(tag8: u64) -> usize {
        data_len8(Self::TAG_LEN, tag8)
    }

    #[inline]
    unsafe fn decode_deltas8(
        input: *const u8,
        tag8: u64,
        base: Self,
        output: *mut Self::Elem,
    ) -> (usize, Self) {
        if Self::has_any_tag3(tag8) {
            return crate::raw_group::default_decode_deltas8(input, tag8, base, output);
        }

        let tags = tag8.to_le_bytes();
        let mut deltas = [vdupq_n_u32(0); 8];
        let mut bases = [0u64; 9];
        bases[0] = vgetq_lane_u64::<1>(base.1);
        let mut offset = 0usize;
        unroll! {
            for i in 0..8 {
                let (len, delta) = Self::decode32(input.add(offset), tags[i]);
                deltas[i] = delta;
                bases[i + 1] = bases[i] + vaddlvq_u32(delta);
                offset += len;
            }
        }
        let z = vdupq_n_u32(0);
        unroll! {
            for i in 0..8 {
                let p = vdupq_n_u64(bases[i]);
                let a_b_c_d = deltas[i];
                let z_a_b_c = vextq_u32(z, a_b_c_d, 3);
                let a_ab = vaddl_u32(vget_low_u32(z_a_b_c), vget_low_u32(a_b_c_d));
                let bc_cd = vaddl_high_u32(z_a_b_c, a_b_c_d);
                let pa_pab = vaddq_u64(p, a_ab);
                let group = Self(pa_pab, vaddq_u64(pa_pab, bc_cd));
                Self::store_unaligned(output.add(i * 4), group);
                if i == 7 {
                    return (offset, group);
                }
            }
        }

        unreachable!()
    }

    #[inline]
    unsafe fn skip_deltas8(input: *const u8, tag8: u64) -> (usize, Self::Elem) {
        if Self::has_any_tag3(tag8) {
            return crate::raw_group::default_skip_deltas8::<Self>(input, tag8);
        }

        let tags = tag8.to_le_bytes();
        let mut offset = 0usize;
        let mut sum = 0u64;
        unroll! {
            for i in 0..8 {
                let (len, deltas) = Self::decode32(input.add(offset), tags[i]);
                offset += len;
                sum = sum.wrapping_add(vaddlvq_u32(deltas));
            }
        }
        (offset, sum)
    }
}

#[cfg(test)]
crate::tests::raw_group_test_suite!();

#[cfg(test)]
crate::tests::compat_test_suite!();
