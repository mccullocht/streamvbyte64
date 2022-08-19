const fn tag_to_encoded_len(tag: u8, tag_len: [usize; 4]) -> usize {
    tag_len[tag as usize & 0x3]
        + tag_len[(tag as usize >> 2) & 0x3]
        + tag_len[(tag as usize >> 4) & 0x3]
        + tag_len[(tag as usize >> 6) & 0x3]
}

/// Generate a table that maps each group tag to the number of encoded bytes.
/// Input tag_len is the byte length for each 2-bit tag value.
pub(crate) const fn tag_length_table(tag_len: [usize; 4]) -> [u8; 256] {
    let mut table = [0u8; 256];
    let mut tag = 0usize;
    while tag < 256 {
        table[tag] = tag_to_encoded_len(tag as u8, tag_len) as u8;
        tag += 1;
    }
    table
}

const fn tag_mask32(len: usize) -> u32 {
    if len == 4 {
        u32::MAX
    } else {
        (1u32 << (len * 8)) - 1
    }
}

/// Generate a table that maps each value tag to a mask that covers values of that width.
/// Input tag_len is the byte length for each 2-bit tag value.
pub(crate) const fn tag_mask_table32(tag_len: [usize; 4]) -> [u32; 4] {
    [
        tag_mask32(tag_len[0]),
        tag_mask32(tag_len[1]),
        tag_mask32(tag_len[2]),
        tag_mask32(tag_len[3]),
    ]
}

const fn tag_mask64(len: usize) -> u64 {
    if len == 8 {
        u64::MAX
    } else {
        (1u64 << (len * 8)) - 1
    }
}

/// Generate a table that maps each value tag to a mask that covers values of that width.
/// Input tag_len is the byte length for each 2-bit tag value.
pub(crate) const fn tag_mask_table64(tag_len: [usize; 4]) -> [u64; 4] {
    [
        tag_mask64(tag_len[0]),
        tag_mask64(tag_len[1]),
        tag_mask64(tag_len[2]),
        tag_mask64(tag_len[3]),
    ]
}
