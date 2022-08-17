use super::scalar;
use crate::unsafe_group::UnsafeGroup;
use std::arch::aarch64::{
    uint64x2_t, uint8x16x2_t, uint8x16x4_t, vaddlvq_u8, vaddq_u64, vaddvq_u32, vandq_u8, vcgtq_u32,
    vcgtq_u64, vdupq_laneq_u64, vdupq_n_u32, vdupq_n_u64, vdupq_n_u8, vextq_u64, vld1q_s32,
    vld1q_u64, vld1q_u8, vminq_u8, vpaddd_u64, vpaddlq_u16, vpaddlq_u8, vqmovn_high_u64,
    vqmovn_u64, vqtbl1q_u8, vqtbl2q_u8, vqtbl4q_u8, vreinterpretq_u64_u8, vreinterpretq_u8_u32,
    vreinterpretq_u8_u64, vshlq_u32, vst1q_u64, vst1q_u8, vsubq_u64,
};

const fn fill_encode_shuffle(tag: u8, i: u8, mut shuf: [u8; 32], oidx: usize) -> ([u8; 32], usize) {
    let vtag = (tag >> (i * 2)) & 0x3;
    shuf[oidx] = i * 8;
    if vtag >= 1 {
        shuf[oidx + 1] = i * 8 + 1;
    }
    if vtag >= 2 {
        shuf[oidx + 2] = i * 8 + 2;
        shuf[oidx + 3] = i * 8 + 3;
    }
    if vtag >= 3 {
        shuf[oidx + 4] = i * 8 + 4;
        shuf[oidx + 5] = i * 8 + 5;
        shuf[oidx + 6] = i * 8 + 6;
        shuf[oidx + 7] = i * 8 + 7;
    }
    (shuf, oidx + (1usize << vtag))
}

// Creates a table that maps a tag value to the shuffle that packs 4 input values to an output value.
const fn tag_encode_shuffle_table() -> [[u8; 32]; 256] {
    let mut table = [[0u8; 32]; 256];
    let mut tag = 0usize;
    while tag < 256 {
        let (mut shuf, mut oidx) = fill_encode_shuffle(tag as u8, 0, [255u8; 32], 0);
        (shuf, oidx) = fill_encode_shuffle(tag as u8, 1, shuf, oidx);
        (shuf, oidx) = fill_encode_shuffle(tag as u8, 2, shuf, oidx);
        table[tag] = fill_encode_shuffle(tag as u8, 3, shuf, oidx).0;
        tag += 1;
    }
    table
}
const ENCODE_TABLE: [[u8; 32]; 256] = tag_encode_shuffle_table();

const fn fill_decode_shuffle(tag: u8, i: usize, mut shuf: [u8; 32], iidx: u8) -> ([u8; 32], u8) {
    let vtag = (tag >> (i * 2)) & 0x3;
    shuf[i * 8] = iidx;
    if vtag >= 1 {
        shuf[i * 8 + 1] = iidx + 1;
    }
    if vtag >= 2 {
        shuf[i * 8 + 2] = iidx + 2;
        shuf[i * 8 + 3] = iidx + 3;
    }
    if vtag >= 3 {
        shuf[i * 8 + 4] = iidx + 4;
        shuf[i * 8 + 5] = iidx + 5;
        shuf[i * 8 + 6] = iidx + 6;
        shuf[i * 8 + 7] = iidx + 7;
    }
    (shuf, iidx + (1u8 << vtag))
}

// Creates a table that maps a tag value to the shuffle that unpacks 4 input values to an output value.
const fn tag_decode_shuffle_table() -> [[u8; 32]; 256] {
    let mut table = [[0u8; 32]; 256];
    let mut tag = 0usize;
    while tag < 256 {
        let (mut shuf, mut oidx) = fill_decode_shuffle(tag as u8, 0, [255u8; 32], 0);
        (shuf, oidx) = fill_decode_shuffle(tag as u8, 1, shuf, oidx);
        (shuf, oidx) = fill_decode_shuffle(tag as u8, 2, shuf, oidx);
        table[tag] = fill_decode_shuffle(tag as u8, 3, shuf, oidx).0;
        tag += 1;
    }
    table
}
const DECODE_TABLE: [[u8; 32]; 256] = tag_decode_shuffle_table();

#[derive(Clone, Copy, Debug)]
pub(crate) struct UnsafeGroupImpl(uint64x2_t, uint64x2_t);

impl UnsafeGroupImpl {
    #[inline]
    unsafe fn compute_tag(group: Self) -> u8 {
        // Perform a saturating narrow of the group values to get a smaller vector for comparison.
        let lo = vqmovn_high_u64(vqmovn_u64(group.0), group.1);
        // Compute comparisons with the maximum value for tags [1,2,3].
        let tag3_v0 = vcgtq_u64(group.0, vdupq_n_u64(0xffffffff));
        let tag3_v1 = vcgtq_u64(group.1, vdupq_n_u64(0xffffffff));
        let tag2 = vcgtq_u32(lo, vdupq_n_u32(0xffff));
        let tag1 = vcgtq_u32(lo, vdupq_n_u32(0xff));
        // Shuffle comparison result bytes together into a single vector where each set value represents
        // the result of a comparison.
        let shuf = vld1q_u8(
            [
                255, 0, 16, 32, 255, 4, 20, 40, 255, 8, 24, 48, 255, 12, 28, 56,
            ]
            .as_ptr(),
        );
        let tag_bytes = vqtbl4q_u8(
            uint8x16x4_t(
                vreinterpretq_u8_u32(tag1),
                vreinterpretq_u8_u32(tag2),
                vreinterpretq_u8_u64(tag3_v0),
                vreinterpretq_u8_u64(tag3_v1),
            ),
            shuf,
        );
        // Reduce each comparison result to 0 or 1, then sum pairwise to compute tags for each value.
        let tag_quads = vminq_u8(tag_bytes, vdupq_n_u8(1));
        let tag_halves = vpaddlq_u8(tag_quads);
        let tags = vpaddlq_u16(tag_halves);
        // Shift the tag values and sum across vector to produce a single u8 value.
        vaddvq_u32(vshlq_u32(tags, vld1q_s32([0, 2, 4, 6].as_ptr()))) as u8
    }
}

impl UnsafeGroup for UnsafeGroupImpl {
    type Elem = u64;

    const TAG_LEN: [u8; 4] = super::TAG_LEN;

    #[inline]
    fn set1(value: Self::Elem) -> Self {
        unsafe { UnsafeGroupImpl(vdupq_n_u64(value), vdupq_n_u64(value)) }
    }

    #[inline]
    unsafe fn load_unaligned(ptr: *const Self::Elem) -> Self {
        // NB: there are two intrinsic calls but this should be translated into a single ldp instruction.
        UnsafeGroupImpl(vld1q_u64(ptr), vld1q_u64(ptr.add(2)))
    }

    #[inline]
    unsafe fn store_unaligned(ptr: *mut Self::Elem, group: Self) {
        // NB: there are two intrinsic calls but this should be translated into a single stp instruction.
        vst1q_u64(ptr, group.0);
        vst1q_u64(ptr.add(2), group.1);
    }

    #[inline]
    unsafe fn encode(output: *mut u8, group: Self) -> (u8, usize) {
        let tag = Self::compute_tag(group);
        let shuf1 = vld1q_u8(ENCODE_TABLE[tag as usize].as_ptr());
        let shuf2 = vld1q_u8(ENCODE_TABLE[tag as usize].as_ptr().add(16));
        let tbl_bytes = uint8x16x2_t(vreinterpretq_u8_u64(group.0), vreinterpretq_u8_u64(group.1));
        let out1 = vqtbl2q_u8(tbl_bytes, shuf1);
        let out2 = vqtbl2q_u8(tbl_bytes, shuf2);
        vst1q_u8(output, out1);
        vst1q_u8(output.add(16), out2);
        (tag, Self::data_len(tag))
    }

    #[inline]
    unsafe fn encode_deltas(output: *mut u8, base: Self, group: Self) -> (u8, usize) {
        let b0 = vextq_u64(base.1, group.0, 1);
        let b1 = vextq_u64(group.0, group.1, 1);
        Self::encode(
            output,
            UnsafeGroupImpl(vsubq_u64(group.0, b0), vsubq_u64(group.1, b1)),
        )
    }

    #[inline]
    unsafe fn decode(input: *const u8, tag: u8) -> (usize, Self) {
        let shuf1 = vld1q_u8(DECODE_TABLE[tag as usize].as_ptr());
        let shuf2 = vld1q_u8(DECODE_TABLE[tag as usize].as_ptr().add(16));
        let tbl_bytes = uint8x16x2_t(vld1q_u8(input), vld1q_u8(input.add(16)));
        let g0 = vqtbl2q_u8(tbl_bytes, shuf1);
        let g1 = vqtbl2q_u8(tbl_bytes, shuf2);
        (
            Self::data_len(tag),
            UnsafeGroupImpl(vreinterpretq_u64_u8(g0), vreinterpretq_u64_u8(g1)),
        )
    }

    #[inline]
    unsafe fn decode_deltas(input: *const u8, tag: u8, base: Self) -> (usize, Self) {
        let (len, group) = Self::decode(input, tag);
        // lol this is -35% throughput
        let a_b = group.0;
        let c_d = group.1;
        let z = vdupq_n_u64(0);
        let z_a = vextq_u64(z, a_b, 1);
        let a_ab = vaddq_u64(z_a, a_b);
        let p = vdupq_laneq_u64(base.1, 1);
        let pa_pab = vaddq_u64(p, a_ab);
        let b_c = vextq_u64(a_b, c_d, 1);
        let bc_cd = vaddq_u64(b_c, c_d);
        let pabc_pabcd = vaddq_u64(pa_pab, bc_cd);
        (len, UnsafeGroupImpl(pa_pab, pabc_pabcd))
    }

    #[inline]
    fn data_len(tag: u8) -> usize {
        scalar::UnsafeGroupImpl::data_len(tag)
    }

    #[inline]
    unsafe fn skip_deltas(input: *const u8, tag: u8) -> (usize, Self::Elem) {
        let (len, group) = Self::decode(input, tag);
        let half = vaddq_u64(group.0, group.1);
        (len, vpaddd_u64(half))
    }

    #[inline]
    fn data_len8(tag8: u64) -> usize {
        unsafe {
            // Load tag8 value so that we get a nibble in each of 16 8-bit lanes.
            let mut nibble_tags = vreinterpretq_u8_u64(vld1q_u64([tag8, tag8 >> 4].as_ptr()));
            nibble_tags = vandq_u8(nibble_tags, vdupq_n_u8(0xf));
            let nibble_len = vqtbl1q_u8(
                vld1q_u8([2, 3, 5, 9, 3, 4, 6, 10, 5, 6, 8, 12, 9, 10, 12, 16].as_ptr()),
                nibble_tags,
            );
            vaddlvq_u8(nibble_len) as usize
        }
    }
}

#[cfg(test)]
crate::tests::unsafe_group_test_suite!();

#[cfg(test)]
crate::tests::compat_test_suite!();
