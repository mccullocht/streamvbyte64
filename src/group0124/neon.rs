use super::scalar;
use crate::unsafe_group::UnsafeGroup;
use std::arch::aarch64::{
    uint32x4_t, vaddlvq_u8, vaddq_u32, vaddvq_u32, vaddvq_u8, vandq_u8, vclzq_u32, vdupq_laneq_u32,
    vdupq_n_u32, vdupq_n_u8, vextq_u32, vld1q_s32, vld1q_u32, vld1q_u64, vld1q_u8, vqtbl1q_u8,
    vreinterpretq_u32_u8, vreinterpretq_u8_u32, vreinterpretq_u8_u64, vshlq_u32, vshrq_n_u32,
    vst1q_u32, vst1q_u8, vsubq_u32,
};

const fn fill_encode_shuffle(tag: u8, i: u8, mut shuf: [u8; 16], oidx: usize) -> ([u8; 16], usize) {
    let vtag = (tag >> (i * 2)) & 0x3;
    let mut len = 0;
    if vtag >= 1 {
        shuf[oidx] = i * 4;
        len += 1;
    }
    if vtag >= 2 {
        shuf[oidx + 1] = i * 4 + 1;
        len += 1;
    }
    if vtag >= 3 {
        shuf[oidx + 2] = i * 4 + 2;
        shuf[oidx + 3] = i * 4 + 3;
        len += 2;
    }
    (shuf, oidx + len)
}

// Creates a table that maps a tag value to the shuffle that packs 4 input values to an output value.
const fn tag_encode_shuffle_table() -> [[u8; 16]; 256] {
    let mut table = [[64u8; 16]; 256];
    let mut tag = 0usize;
    while tag < 256 {
        let (mut shuf, mut oidx) = fill_encode_shuffle(tag as u8, 0, [255u8; 16], 0);
        (shuf, oidx) = fill_encode_shuffle(tag as u8, 1, shuf, oidx);
        (shuf, oidx) = fill_encode_shuffle(tag as u8, 2, shuf, oidx);
        table[tag] = fill_encode_shuffle(tag as u8, 3, shuf, oidx).0;
        tag += 1;
    }
    table
}
const ENCODE_TABLE: [[u8; 16]; 256] = tag_encode_shuffle_table();

const fn fill_decode_shuffle(tag: u8, i: usize, mut shuf: [u8; 16], iidx: u8) -> ([u8; 16], u8) {
    let vtag = (tag >> (i * 2)) & 0x3;
    let mut len = 0;
    if vtag >= 1 {
        shuf[i * 4] = iidx;
        len += 1;
    }
    if vtag >= 2 {
        shuf[i * 4 + 1] = iidx + 1;
        len += 1;
    }
    if vtag >= 3 {
        shuf[i * 4 + 2] = iidx + 2;
        shuf[i * 4 + 3] = iidx + 3;
        len += 2;
    }
    (shuf, iidx + len)
}

// Creates a table that maps a tag value to the shuffle that unpacks 4 input values to an output value.
const fn tag_decode_shuffle_table() -> [[u8; 16]; 256] {
    let mut table = [[64u8; 16]; 256];
    let mut tag = 0usize;
    while tag < 256 {
        let (mut shuf, mut oidx) = fill_decode_shuffle(tag as u8, 0, [255u8; 16], 0);
        (shuf, oidx) = fill_decode_shuffle(tag as u8, 1, shuf, oidx);
        (shuf, oidx) = fill_decode_shuffle(tag as u8, 2, shuf, oidx);
        table[tag] = fill_decode_shuffle(tag as u8, 3, shuf, oidx).0;
        tag += 1;
    }
    table
}
const DECODE_TABLE: [[u8; 16]; 256] = tag_decode_shuffle_table();

#[derive(Clone, Copy, Debug)]
pub(crate) struct UnsafeGroupImpl(uint32x4_t);

impl UnsafeGroup for UnsafeGroupImpl {
    type Elem = u32;
    const TAG_LEN: [usize; 4] = super::TAG_LEN;

    #[inline]
    fn set1(value: u32) -> Self {
        UnsafeGroupImpl(unsafe { vdupq_n_u32(value) })
    }

    #[inline]
    unsafe fn load_unaligned(ptr: *const u32) -> Self {
        UnsafeGroupImpl(vld1q_u32(ptr))
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
        let deltas = UnsafeGroupImpl(vsubq_u32(group.0, vextq_u32(base.0, group.0, 3)));
        Self::encode(output, deltas)
    }

    #[inline]
    unsafe fn decode(input: *const u8, tag: u8) -> (usize, Self) {
        // Use a precomputed table that shuffles the minimally packed encoded bytes into the right place.
        let v = vreinterpretq_u32_u8(vqtbl1q_u8(
            vld1q_u8(input),
            vld1q_u8(DECODE_TABLE[tag as usize].as_ptr()),
        ));
        (Self::data_len(tag), UnsafeGroupImpl(v))
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
        (read, UnsafeGroupImpl(vaddq_u32(pa_pab_pbc_pbd, z_z_a_ab)))
    }

    #[inline]
    fn data_len(tag: u8) -> usize {
        scalar::UnsafeGroupImpl::data_len(tag)
    }

    #[inline]
    unsafe fn skip_deltas(input: *const u8, tag: u8) -> (usize, u32) {
        let (read, group) = Self::decode(input, tag);
        (read, vaddvq_u32(group.0))
    }

    // TODO: move nibble table computation to tag_utils
    #[inline]
    fn data_len8(tag8: u64) -> usize {
        unsafe {
            // Load tag8 value so that we get a nibble in each of 16 8-bit lanes.
            let mut nibble_tags = vreinterpretq_u8_u64(vld1q_u64([tag8, tag8 >> 4].as_ptr()));
            nibble_tags = vandq_u8(nibble_tags, vdupq_n_u8(0xf));
            let nibble_len = vqtbl1q_u8(
                vld1q_u8([0, 1, 2, 4, 1, 2, 3, 5, 2, 3, 4, 6, 4, 5, 6, 8].as_ptr()),
                nibble_tags,
            );
            vaddlvq_u8(nibble_len) as usize
        }
    }
}
