use crunchy::unroll;

use super::{scalar, CodingDescriptor0124};
use crate::arch::neon::{
    data_len8, sum_deltas32, tag_decode_shuffle_table32, tag_encode_shuffle_table32,
};
use crate::coding_descriptor::CodingDescriptor;
use crate::raw_group::RawGroup;
use std::arch::aarch64::{
    uint32x4_t, vaddvq_u32, vaddvq_u8, vclzq_u32, vdupq_laneq_u32, vdupq_n_u32, vextq_u32,
    vgetq_lane_u32, vld1q_s32, vld1q_u32, vld1q_u8, vqtbl1q_u8, vreinterpretq_u32_u8,
    vreinterpretq_u8_u32, vshlq_u32, vshrq_n_u32, vst1q_u32, vst1q_u8, vsubq_u32,
};

const ENCODE_TABLE: [[u8; 16]; 256] = tag_encode_shuffle_table32(CodingDescriptor0124::TAG_LEN);
const DECODE_TABLE: [[u8; 16]; 256] = tag_decode_shuffle_table32(CodingDescriptor0124::TAG_LEN);

/// Takes a `u64` containing 8 tag values and produces a value containing the start offset of each
/// group (one per byte) and the sum of all group lengths.
///
/// This uses a SIMD like approach where each byte of the `u64` is treated like a lane. This works
/// because the maximum length of each individual tag value is 16 bytes so the running sum of all
/// lengths will not exceed 128.
fn tag8_offsets(tag8: u64) -> (u64, usize) {
    // Compute a marker for each individual tag that has a value of 4 (0b11) then sum these to get
    // the number of these values per byte.
    let mut length4s = (tag8 & (tag8 >> 1)) & 0x5555555555555555;
    length4s = ((length4s >> 2) & 0x3333333333333333) + (length4s & 0x3333333333333333);
    length4s = ((length4s >> 4) & 0x0f0f0f0f0f0f0f0f) + (length4s & 0x0f0f0f0f0f0f0f0f);

    let mut lengths8 = ((tag8 >> 2) & 0x3333333333333333) + (tag8 & 0x3333333333333333);
    lengths8 = ((lengths8 >> 4) & 0x0f0f0f0f0f0f0f0f) + (lengths8 & 0x0f0f0f0f0f0f0f0f);
    lengths8 += length4s;

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
    const TAG_LEN: [usize; 4] = CodingDescriptor0124::TAG_LEN;

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
    unsafe fn decode8(input: *const u8, tag8: u64, output: *mut Self::Elem) -> usize {
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
    unsafe fn decode_deltas(input: *const u8, tag: u8, base: Self) -> (usize, Self) {
        let (read, deltas) = Self::decode(input, tag);
        let group = RawGroupImpl(sum_deltas32(vdupq_laneq_u32::<3>(base.0), deltas.0));
        (read, group)
    }

    #[inline]
    unsafe fn decode_deltas8(
        input: *const u8,
        tag8: u64,
        base: Self,
        output: *mut Self::Elem,
    ) -> (usize, Self) {
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

#[cfg(test)]
mod tests {
    #[test]
    fn tag8_offsets() {
        use super::tag8_offsets;
        assert_eq!(
            tag8_offsets(0b00000010_11111111_10011100),
            (0x19_19_19_19_19_17_07_00, 0x19)
        );
        assert_eq!(tag8_offsets(u64::MAX), (0x70_60_50_40_30_20_10_00, 0x80));
    }
}
