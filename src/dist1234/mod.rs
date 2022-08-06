#[cfg(all(target_arch = "aarch64", target_endian = "little"))]
mod neon;

const TAG_LEN: [u8; 4] = [1, 2, 3, 4];
const TAG_MASK: [u32; 4] = crate::tag_utils::tag_mask_table32(TAG_LEN);

#[inline]
fn tag_value(v: u32) -> u8 {
    3u32.saturating_sub(v.leading_zeros() / 8) as u8
}

crate::unsafe_group::declare_scalar_implementation!(u32, dist1234);
