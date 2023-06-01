use std::arch::aarch64::{
    vaddlvq_u8, vandq_u8, vdupq_n_u8, vld1q_u64, vld1q_u8, vqtbl1q_u8, vreinterpretq_u8_u64,
};

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

/// These functions accept a mapping from tag value -> byte length and use that to generate a table
/// that can be used to shuffle an input group to compress it.
///
/// These functions are generated by macro because we can't alter the output type using generics
/// unless we use unstable features.
macro_rules! generate_encode_shuffle_table {
    ($name:ident, $elem_type:ty) => {
        pub(crate) const fn $name(
            tag_len: [usize; 4],
        ) -> [[u8; std::mem::size_of::<$elem_type>() * 4]; 256] {
            // Default fill with 64 as it is larger than any index usable by any vtbl instruction.
            // This allows us to add to adjust the indices without overflowing.
            let mut table = [[64u8; std::mem::size_of::<$elem_type>() * 4]; 256];
            let mut tag = 0usize;
            while tag < 256 {
                let mut shuf_idx = 0;
                let mut i = 0;
                while i < 4 {
                    let vtag = (tag >> (i * 2)) & 0x3;
                    let mut j = 0;
                    while j < tag_len[vtag as usize] {
                        table[tag][shuf_idx] = (i * std::mem::size_of::<$elem_type>() + j) as u8;
                        shuf_idx += 1;
                        j += 1;
                    }
                    i += 1;
                }
                tag += 1;
            }
            table
        }
    };
}
generate_encode_shuffle_table!(tag_encode_shuffle_table32, u32);
generate_encode_shuffle_table!(tag_encode_shuffle_table64, u64);

/// These functions accept a mapping from tag value -> byte length and use that to generate a table
/// that can be used to shuffle a compressed input into an output group.
///
/// These functions are generated by macro because we can't alter the output type using generics
/// unless we use unstable features.
macro_rules! generate_decode_shuffle_table {
    ($name:ident, $elem_type:ty) => {
        pub(crate) const fn $name(
            tag_len: [usize; 4],
        ) -> [[u8; std::mem::size_of::<$elem_type>() * 4]; 256] {
            // Default fill with 64 as it is larger than any index usable by any vtbl instruction.
            // This allows us to add to adjust the indices without overflowing.
            let mut table = [[64u8; std::mem::size_of::<$elem_type>() * 4]; 256];
            let mut tag = 0usize;
            while tag < 256 {
                let mut shuf_idx = 0;
                let mut i = 0;
                while i < 4 {
                    let vtag = (tag >> (i * 2)) & 0x3;
                    let mut j = 0;
                    while j < tag_len[vtag as usize] {
                        table[tag][i * std::mem::size_of::<$elem_type>() + j] = shuf_idx;
                        shuf_idx += 1;
                        j += 1;
                    }
                    i += 1;
                }
                tag += 1;
            }
            table
        }
    };
}
generate_decode_shuffle_table!(tag_decode_shuffle_table32, u32);
generate_decode_shuffle_table!(tag_decode_shuffle_table64, u64);
