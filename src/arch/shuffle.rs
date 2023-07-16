/// Architecture independent utility for generating shuffle tables.
///
/// x86_64 and aarch64 have similar shuffle instructions that can share some logic but may not be
/// able to share exactly the same tables.

/// Generate a constant value that can be used to encode `ENTRY_LEN / ELEM_LEN` input values based
/// on `tag` and the `tag_len` distribution.
#[allow(dead_code)]
pub(crate) const fn encode_shuffle_entry<const ELEM_LEN: usize, const ENTRY_LEN: usize>(
    tag: u8,
    tag_len: [usize; 4],
    fill_byte: u8,
) -> [u8; ENTRY_LEN] {
    if ENTRY_LEN % ELEM_LEN != 0 {
        panic!("ENTRY_LEN must divide evenly by ELEM_LEN")
    }
    let num_values = ENTRY_LEN / ELEM_LEN;
    let mut entry = [fill_byte; ENTRY_LEN];
    let mut shuf_idx = 0;
    let mut i = 0;
    while i < num_values {
        let len = tag_len[tag as usize >> (i * 2) & 0x3];
        let mut j = 0;
        while j < len {
            entry[shuf_idx] = (i * ELEM_LEN + j) as u8;
            shuf_idx += 1;
            j += 1;
        }
        i += 1;
    }
    entry
}

/// Generate a constant value that can be used to decode `ENTRY_LEN / ELEM_LEN` input values based
/// on `tag` and the `tag_len` distribution.
#[allow(dead_code)]
pub(crate) const fn decode_shuffle_entry<const ELEM_LEN: usize, const ENTRY_LEN: usize>(
    tag: u8,
    tag_len: [usize; 4],
    fill_byte: u8,
) -> [u8; ENTRY_LEN] {
    if ENTRY_LEN % ELEM_LEN != 0 {
        panic!("ENTRY_LEN must divide evenly by ELEM_LEN")
    }
    let num_values = ENTRY_LEN / ELEM_LEN;
    let mut entry = [fill_byte; ENTRY_LEN];
    let mut shuf_idx = 0;
    let mut i = 0;
    while i < num_values {
        let len = tag_len[tag as usize >> (i * 2) & 0x3];
        let mut j = 0;
        while j < len {
            entry[i * ELEM_LEN + j] = shuf_idx;
            shuf_idx += 1;
            j += 1;
        }
        i += 1;
    }
    entry
}
