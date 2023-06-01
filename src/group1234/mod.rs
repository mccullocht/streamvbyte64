#[cfg(all(target_arch = "aarch64", target_endian = "little"))]
mod neon;

use crate::group_impl;
use crate::Group32;

const TAG_LEN: [usize; 4] = [1, 2, 3, 4];
const TAG_MASK: [u32; 4] = crate::tag_utils::tag_mask_table32(TAG_LEN);

#[inline]
fn tag_value(v: u32) -> u8 {
    3u32.saturating_sub(v.leading_zeros() / 8) as u8
}

crate::raw_group::declare_scalar_implementation!(u32, group1234);

#[derive(Clone, Copy)]
enum Impl {
    Scalar,
    #[cfg(all(target_arch = "aarch64", target_endian = "little"))]
    Neon,
}

/// `Group1234` packs groups of 4 32-bit integers into lengths of 1, 2, 3, or 4 bytes.
/// This implementation has acceleration support on little-endian `aarch64` targets using `NEON` instructions.
#[derive(Clone, Copy)]
pub struct Group1234(Impl);

impl Group1234 {
    fn scalar_impl() -> Self {
        Group1234(Impl::Scalar)
    }
}

impl Group32 for Group1234 {
    fn new() -> Self {
        #[cfg(all(target_arch = "aarch64", target_endian = "little"))]
        {
            if std::arch::is_aarch64_feature_detected!("neon") {
                return Group1234(Impl::Neon);
            }
        }
        Self::scalar_impl()
    }

    fn encode(&self, values: &[u32], tags: &mut [u8], encoded: &mut [u8]) -> usize {
        match self.0 {
            Impl::Scalar => group_impl::encode::<scalar::RawGroupImpl>(values, tags, encoded),
            #[cfg(all(target_arch = "aarch64", target_endian = "little"))]
            Impl::Neon => group_impl::encode::<neon::RawGroupImpl>(values, tags, encoded),
        }
    }

    fn encode_deltas(
        &self,
        initial: u32,
        values: &[u32],
        tags: &mut [u8],
        encoded: &mut [u8],
    ) -> usize {
        match self.0 {
            Impl::Scalar => {
                group_impl::encode_deltas::<scalar::RawGroupImpl>(initial, values, tags, encoded)
            }
            #[cfg(all(target_arch = "aarch64", target_endian = "little"))]
            Impl::Neon => {
                group_impl::encode_deltas::<neon::RawGroupImpl>(initial, values, tags, encoded)
            }
        }
    }

    fn decode(&self, tags: &[u8], encoded: &[u8], values: &mut [u32]) -> usize {
        match self.0 {
            Impl::Scalar => group_impl::decode::<scalar::RawGroupImpl>(tags, encoded, values),
            #[cfg(all(target_arch = "aarch64", target_endian = "little"))]
            Impl::Neon => group_impl::decode::<neon::RawGroupImpl>(tags, encoded, values),
        }
    }

    fn decode_deltas(
        &self,
        initial: u32,
        tags: &[u8],
        encoded: &[u8],
        values: &mut [u32],
    ) -> usize {
        match self.0 {
            Impl::Scalar => {
                group_impl::decode_deltas::<scalar::RawGroupImpl>(initial, tags, encoded, values)
            }
            #[cfg(all(target_arch = "aarch64", target_endian = "little"))]
            Impl::Neon => {
                group_impl::decode_deltas::<neon::RawGroupImpl>(initial, tags, encoded, values)
            }
        }
    }

    fn data_len(&self, tags: &[u8]) -> usize {
        match self.0 {
            Impl::Scalar => group_impl::data_len::<scalar::RawGroupImpl>(tags),
            #[cfg(all(target_arch = "aarch64", target_endian = "little"))]
            Impl::Neon => group_impl::data_len::<neon::RawGroupImpl>(tags),
        }
    }

    fn skip_deltas(&self, tags: &[u8], encoded: &[u8]) -> (usize, u32) {
        match self.0 {
            Impl::Scalar => group_impl::skip_deltas::<scalar::RawGroupImpl>(tags, encoded),
            #[cfg(all(target_arch = "aarch64", target_endian = "little"))]
            Impl::Neon => group_impl::skip_deltas::<neon::RawGroupImpl>(tags, encoded),
        }
    }
}

#[cfg(test)]
crate::tests::group_test_suite!(Group32, Group1234);
