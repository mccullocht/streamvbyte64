const TAG_LEN: [u8; 4] = [1, 2, 4, 8];
const TAG_MASK: [u64; 4] = crate::tag_utils::tag_mask_table64(TAG_LEN);

#[inline]
fn tag_value(v: u64) -> u8 {
    // compute a 3-bit tag value in [0,7] in the same way we do for dist1234.
    // then take ~log2 to get a mapping from bytes required to length:
    // [0,1] => 0
    // [2,2] => 1
    // [3,4] => 2
    // [5,8] => 3
    let t3 = 7u32.saturating_sub(v.leading_zeros() / 8);
    (u32::BITS - t3.leading_zeros()) as u8
}

crate::unsafe_group::declare_scalar_implementation!(u64, dist1248);
