use super::shuffle::{decode_shuffle_entry, encode_shuffle_entry};
use std::arch::aarch64::{
    uint32x4_t, vaddlvq_u8, vaddq_u32, vandq_u8, vdupq_n_u32, vdupq_n_u8, vextq_u32, vld1q_u64,
    vld1q_u8, vqtbl1q_u8, vreinterpretq_u8_u64,
};

/// Generate a table that encodes `ENTRY_LEN / ELEM_LEN` input values to contiguous bytes based on
/// `tag and `tag_len`.
pub(crate) const fn encode_shuffle_table<const ELEM_LEN: usize, const ENTRY_LEN: usize>(
    tag_len: [usize; 4],
) -> [[u8; ENTRY_LEN]; 256] {
    let mut table = [[0u8; ENTRY_LEN]; 256];
    let mut tag = 0;
    while tag < 256 {
        table[tag] =
            encode_shuffle_entry::<ELEM_LEN, ENTRY_LEN>(tag as u8, tag_len, ENTRY_LEN as u8);
        tag += 1;
    }
    table
}

/// Generate a table that decodes `ENTRY_LEN / ELEM_LEN` values from contiguous bytes based on
/// `tag and `tag_len`.
pub(crate) const fn decode_shuffle_table<const ELEM_LEN: usize, const ENTRY_LEN: usize>(
    tag_len: [usize; 4],
) -> [[u8; ENTRY_LEN]; 256] {
    let mut table = [[0u8; ENTRY_LEN]; 256];
    let mut tag = 0;
    while tag < 256 {
        table[tag] =
            decode_shuffle_entry::<ELEM_LEN, ENTRY_LEN>(tag as u8, tag_len, ENTRY_LEN as u8);
        tag += 1;
    }
    table
}

/// Generate a table mapping the lower half (nibble) of a tag to the length of those two entries.
/// This is used to speed computation of RawGroup::data_len8().
const fn generate_nibble_tag_len_table(tag_len: [usize; 4]) -> [u8; 16] {
    let mut table = [0u8; 16];
    let mut tag = 0usize;
    while tag < 16 {
        table[tag] = tag_len[tag & 0x3] as u8 + tag_len[(tag >> 2) & 0x3] as u8;
        tag += 1;
    }
    table
}

/// Shared implementation of RawGroup::data_len8().
/// This function is inline so the compiler can generate a decode table from tag_len as a constant.
#[inline(always)]
pub(crate) fn data_len8(tag_len: [usize; 4], tag8: u64) -> usize {
    unsafe {
        // Load tag8 value so that we get a nibble in each of 16 8-bit lanes.
        let mut nibble_tags = vreinterpretq_u8_u64(vld1q_u64([tag8, tag8 >> 4].as_ptr()));
        nibble_tags = vandq_u8(nibble_tags, vdupq_n_u8(0xf));
        // Shuffle to get the data length of the values in each nibble.
        let nibble_len = vqtbl1q_u8(
            vld1q_u8(generate_nibble_tag_len_table(tag_len).as_ptr()),
            nibble_tags,
        );
        // Sum across vector to get the complete length of all 8 groups.
        vaddlvq_u8(nibble_len) as usize
    }
}

/// Compute the running sum values of `deltas` starting at `delta_base`.
/// Every lane of `delta_base` is expected to contain the same value.
#[inline(always)]
pub(crate) unsafe fn sum_deltas32(delta_base: uint32x4_t, deltas: uint32x4_t) -> uint32x4_t {
    let p = delta_base;
    let z = vdupq_n_u32(0);
    let a_b_c_d = deltas;
    let z_a_b_c = vextq_u32(z, a_b_c_d, 3);
    let a_ab_bc_cd = vaddq_u32(a_b_c_d, z_a_b_c);
    let z_z_a_ab = vextq_u32(z, a_ab_bc_cd, 2);
    let pa_pab_pbc_pbd = vaddq_u32(p, a_ab_bc_cd);
    vaddq_u32(pa_pab_pbc_pbd, z_z_a_ab)
}
