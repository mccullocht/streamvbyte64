use super::{scalar, CodingDescriptor1234};
use crate::arch::neon::{
    data_len8, sum_deltas32, tag_decode_shuffle_table32, tag_encode_shuffle_table32,
};
use crate::coding_descriptor::CodingDescriptor;
use crate::raw_group::RawGroup;
use crunchy::unroll;
use std::arch::aarch64::{
    uint32x4_t, uint8x16x4_t, vaddlvq_u16, vaddlvq_u8, vaddq_u16, vaddq_u32, vaddvq_u32, vclzq_u32,
    vdupq_laneq_u32, vdupq_n_u16, vdupq_n_u32, vdupq_n_u8, vextq_u16, vextq_u32, vget_low_u16,
    vgetq_lane_u32, vld1_u8, vld1q_s32, vld1q_u32, vld1q_u8, vld2q_u8, vmovl_high_u16, vmovl_u16,
    vmovl_u8, vqsubq_u32, vqtbl1q_u8, vreinterpretq_u32_u8, vreinterpretq_u8_u32, vshlq_u32,
    vshrq_n_u32, vst1q_u32, vst1q_u8, vst4q_u8, vsubq_u32,
};

const ENCODE_TABLE: [[u8; 16]; 256] = tag_encode_shuffle_table32(CodingDescriptor1234::TAG_LEN);
const DECODE_TABLE: [[u8; 16]; 256] = tag_decode_shuffle_table32(CodingDescriptor1234::TAG_LEN);

/// Takes a `u64` containing 8 tag values and produces a value containing the start offset of each
/// group (one per byte) and the sum of all group lengths.
///
/// This uses a SIMD like approach where each byte of the `u64` is treated like a lane. This works
/// because the maximum length of each individual tag value is 16 bytes so the running sum of all
/// lengths will not exceed 128.
fn tag8_offsets(tag8: u64) -> (u64, usize) {
    let mut lengths8 = ((tag8 >> 2) & 0x3333333333333333) + (tag8 & 0x3333333333333333);
    lengths8 = ((lengths8 >> 4) & 0x0f0f0f0f0f0f0f0f) + (lengths8 & 0x0f0f0f0f0f0f0f0f);
    // each tag value is the length of the data value - 1, so add 4 to each byte to account for this
    lengths8 += 0x0404040404040404;

    let mut offsets8 = lengths8 + (lengths8 << 8);
    offsets8 += offsets8 << 16;
    offsets8 += offsets8 << 32;

    // Left shift offsets8 by one byte to get the start offset of each group; extract the high byte
    // of offsets8 to get the length of all groups.
    (offsets8 << 8, (offsets8 >> 56) as usize)
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct RawGroupImpl(uint32x4_t);

impl RawGroup for RawGroupImpl {
    type Elem = u32;
    const TAG_LEN: [usize; 4] = CodingDescriptor1234::TAG_LEN;

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
        let value_tags = vqsubq_u32(vdupq_n_u32(3), vshrq_n_u32(vclzq_u32(group.0), 3));
        // Shift value tags so that they do not overlap and sum them to get the tag.
        let tag = vaddvq_u32(vshlq_u32(value_tags, vld1q_s32([0, 2, 4, 6].as_ptr()))) as u8;
        // Each value tag is one less than the number of encoded bytes, so sum these and add 4.
        let written = vaddvq_u32(value_tags) as usize + 4;

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
        let (read, deltas) = Self::decode(input, tag);
        let group = RawGroupImpl(sum_deltas32(vdupq_laneq_u32::<3>(base.0), deltas.0));
        (read, group)
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
    unsafe fn decode8(input: *const u8, tag8: u64, output: *mut Self::Elem) -> usize {
        if tag8 == 0 {
            // We can decode all 32 entries with 2 loads and 2 stores (plus a broadcast load).
            // vst4q_u8() interleaves 4 input vectors, so we can interleave one 16 byte input with 3 16x0 inputs
            // to produce 16 u32 values where only the low byte is set.
            let zero = vdupq_n_u8(0);
            unroll! {
                for i in 0..2 {
                    let g = vld1q_u8(input.add(i * 16));
                    vst4q_u8(output.add(i * 16) as *mut u8, uint8x16x4_t(g, zero, zero, zero));
                }
            }
            return 32;
        }

        // Compute the offset of each group and total data len up front to allow greater instruction
        // level parallelism.
        let (offsets, data_len) = tag8_offsets(tag8);
        unroll! {
            for i in 0..8 {
                let shift = i * 8;
                let tag = ((tag8 >> shift) & 0xff) as u8;
                let offset = ((offsets >> shift) & 0xff) as usize;
                let (_, group) = Self::decode(input.add(offset), tag);
                Self::store_unaligned(output.add(i * 4), group);
            }
        }
        data_len
    }

    #[inline]
    unsafe fn decode_deltas8(
        input: *const u8,
        tag8: u64,
        base: Self,
        output: *mut Self::Elem,
    ) -> (usize, Self) {
        if tag8 == 0 {
            let z = vdupq_n_u16(0);
            // Load 8 bytes (2 output groups) and widen to 16 bits.
            let deltas = [
                vmovl_u8(vld1_u8(input.add(0))),
                vmovl_u8(vld1_u8(input.add(8))),
                vmovl_u8(vld1_u8(input.add(16))),
                vmovl_u8(vld1_u8(input.add(24))),
            ];
            let mut bases = [0u32; 4];
            bases[0] = vgetq_lane_u32::<3>(base.0);
            bases[1] = bases[0] + vaddlvq_u16(deltas[0]);
            bases[2] = bases[1] + vaddlvq_u16(deltas[1]);
            bases[3] = bases[2] + vaddlvq_u16(deltas[2]);
            unroll! {
                for i in 0..4 {
                    let p = vdupq_n_u32(bases[i]);
                    let a_b_c_d_e_f_g_h = deltas[i];
                    let z_a_b_c_d_e_f_g = vextq_u16(z, a_b_c_d_e_f_g_h, 7);
                    let a_ab_bc_cd_de_ef_fg_gh = vaddq_u16(a_b_c_d_e_f_g_h, z_a_b_c_d_e_f_g);
                    let z_z_a_ab_bc_cd_de_ef = vextq_u16(z, a_ab_bc_cd_de_ef_fg_gh, 6);
                    let a_ab_abc_abcd_bcde_cdef_defg_efgh = vaddq_u16(a_ab_bc_cd_de_ef_fg_gh, z_z_a_ab_bc_cd_de_ef);
                    let a_ab_abc_abcd = vmovl_u16(vget_low_u16(a_ab_abc_abcd_bcde_cdef_defg_efgh));
                    let pa_pab_pabc_pabcd = vaddq_u32(p, a_ab_abc_abcd);
                    vst1q_u32(output.add(i * 8), pa_pab_pabc_pabcd);
                    let pabcde_pabcdef_pabcdefg_pabcdefgh = vaddq_u32(pa_pab_pabc_pabcd, vmovl_high_u16(a_ab_abc_abcd_bcde_cdef_defg_efgh));
                    vst1q_u32(output.add(i * 8 + 4), pabcde_pabcdef_pabcdefg_pabcdefgh);
                    if i == 3 {
                        return (32, RawGroupImpl(pabcde_pabcdef_pabcdefg_pabcdefgh));
                    }
                }
            }
        }

        // Arrange computation to minimize data dependencies between groups to improve instruction
        // level parallelism:
        // * Compute offsets before decoding like in decode8().
        // * Compute base values for each group before computing running sum.
        let (offsets8, data_len) = tag8_offsets(tag8);
        let tags = tag8.to_le_bytes();
        let offsets = offsets8.to_le_bytes();
        let deltas = [
            Self::decode(input, tags[0]).1 .0,
            Self::decode(input.add(offsets[1] as usize), tags[1]).1 .0,
            Self::decode(input.add(offsets[2] as usize), tags[2]).1 .0,
            Self::decode(input.add(offsets[3] as usize), tags[3]).1 .0,
            Self::decode(input.add(offsets[4] as usize), tags[4]).1 .0,
            Self::decode(input.add(offsets[5] as usize), tags[5]).1 .0,
            Self::decode(input.add(offsets[6] as usize), tags[6]).1 .0,
            Self::decode(input.add(offsets[7] as usize), tags[7]).1 .0,
        ];
        let mut bases = [0u32; 8];
        bases[0] = vgetq_lane_u32::<3>(base.0);
        unroll! {
            for i in 0..7 {
                bases[i + 1] = bases[i].wrapping_add(vaddvq_u32(deltas[i]));
            }
        }
        unroll! {
            for i in 0..8 {
                let group_data = sum_deltas32(vdupq_n_u32(bases[i]), deltas[i]);
                vst1q_u32(output.add(i * 4), group_data);
                if i == 7 {
                    return (data_len, RawGroupImpl(group_data));
                }
            }
        }

        unreachable!()
    }

    #[inline]
    fn data_len8(tag8: u64) -> usize {
        data_len8(Self::TAG_LEN, tag8)
    }

    #[inline]
    unsafe fn skip_deltas8(input: *const u8, tag8: u64) -> (usize, Self::Elem) {
        if tag8 == 0 {
            // In this case the values are all single byte in the the 32 bytes from input and we
            // don't care about order so we can sum and widen the output vectors.
            let sg = vld2q_u8(input);
            let sum = vaddlvq_u8(sg.0) + vaddlvq_u8(sg.1);
            return (32, Self::Elem::from(sum));
        }

        let (offsets, data_len) = tag8_offsets(tag8);
        let mut sum = 0u32;
        unroll! {
            for i in 0..8 {
                let shift = i * 8;
                let tag = ((tag8 >> shift) & 0xff) as u8;
                let offset = ((offsets >> shift) & 0xff) as usize;
                let (_, group) = Self::decode(input.add(offset), tag);
                sum = sum.wrapping_add(vaddvq_u32(group.0));
            }
        }
        (data_len, sum)
    }
}

#[cfg(test)]
crate::tests::raw_group_test_suite!();

#[cfg(test)]
crate::tests::compat_test_suite!();

#[cfg(test)]
mod tests {
    use crate::raw_group::RawGroup;

    use super::RawGroupImpl;

    #[test]
    fn decode8_1byte() {
        // Input is 32 bytes (enumerate) with a tag8 of 0.
        let input = std::iter::repeat(0)
            .take(32)
            .enumerate()
            .map(|(i, _)| i as u8)
            .collect::<Vec<_>>();
        let mut output = [0u32; 32];
        let dlen = unsafe { RawGroupImpl::decode8(input.as_ptr(), 0, output.as_mut_ptr()) };

        assert_eq!(dlen, 32);
        assert_eq!(input.iter().map(|v| *v as u32).collect::<Vec<_>>(), output);
    }

    #[test]
    fn decode_deltas8_1byte() {
        let input = std::iter::repeat(1u8).take(32).collect::<Vec<_>>();
        let mut output = [0u32; 32];
        let (dlen, _) = unsafe {
            RawGroupImpl::decode_deltas8(
                input.as_ptr(),
                0,
                RawGroupImpl::set1(1),
                output.as_mut_ptr(),
            )
        };

        assert_eq!(dlen, 32);
        let expected = std::iter::repeat(0)
            .take(32)
            .enumerate()
            .map(|(i, _)| i as u32 + 2)
            .collect::<Vec<_>>();
        assert_eq!(expected, output);
    }

    #[test]
    fn skip_deltas8_1byte() {
        // Input is 32 bytes (enumerate) with a tag8 of 0.
        let input = std::iter::repeat(0)
            .take(32)
            .enumerate()
            .map(|(i, _)| i as u8)
            .collect::<Vec<_>>();
        let (dlen, sum) = unsafe { RawGroupImpl::skip_deltas8(input.as_ptr(), 0) };

        assert_eq!(dlen, 32);
        assert_eq!(input.iter().map(|v| *v as u32).sum::<u32>(), sum);
    }

    #[test]
    fn tag8_offsets() {
        use super::tag8_offsets;
        assert_eq!(
            tag8_offsets(0b00000010_11111111_10011100),
            (0x30_2c_28_24_20_1a_0a_00, 0x34)
        );
        assert_eq!(tag8_offsets(u64::MAX), (0x70_60_50_40_30_20_10_00, 128));
    }
}
