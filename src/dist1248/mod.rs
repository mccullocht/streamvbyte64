#[cfg(all(target_arch = "aarch64", target_endian = "little"))]
mod neon;

use crate::{group_impl, Group64};

const TAG_LEN: [usize; 4] = [1, 2, 4, 8];
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

#[derive(Clone, Copy)]
enum Impl {
    Scalar,
    #[cfg(all(target_arch = "aarch64", target_endian = "little"))]
    Neon,
}

/// `Group1234` packs groups of 4 64-bit integers into lengths of 1, 2, 4, or 8 bytes.
/// This implementation has acceleration support on little-endian `aarch64` targets using `NEON` instructions.
#[derive(Clone, Copy)]
pub struct Group1248(Impl);

impl Group64 for Group1248 {
    fn new() -> Self {
        #[cfg(all(target_arch = "aarch64", target_endian = "little"))]
        {
            if std::arch::is_aarch64_feature_detected!("neon") {
                return Group1248(Impl::Neon);
            }
        }
        Group1248(Impl::Scalar)
    }

    fn encode(&self, values: &[u64], tags: &mut [u8], encoded: &mut [u8]) -> usize {
        match self.0 {
            Impl::Scalar => group_impl::encode::<scalar::UnsafeGroupImpl>(values, tags, encoded),
            Impl::Neon => group_impl::encode::<neon::UnsafeGroupImpl>(values, tags, encoded),
        }
    }

    fn encode_deltas(
        &self,
        initial: u64,
        values: &[u64],
        tags: &mut [u8],
        encoded: &mut [u8],
    ) -> usize {
        match self.0 {
            Impl::Scalar => {
                group_impl::encode_deltas::<scalar::UnsafeGroupImpl>(initial, values, tags, encoded)
            }
            Impl::Neon => {
                group_impl::encode_deltas::<neon::UnsafeGroupImpl>(initial, values, tags, encoded)
            }
        }
    }

    fn decode(&self, tags: &[u8], encoded: &[u8], values: &mut [u64]) -> usize {
        match self.0 {
            Impl::Scalar => group_impl::decode::<scalar::UnsafeGroupImpl>(tags, encoded, values),
            Impl::Neon => group_impl::decode::<neon::UnsafeGroupImpl>(tags, encoded, values),
        }
    }

    fn decode_deltas(
        &self,
        initial: u64,
        tags: &[u8],
        encoded: &[u8],
        values: &mut [u64],
    ) -> usize {
        // TODO: reenable the neon implementation when performance matches scalar.
        group_impl::decode_deltas::<scalar::UnsafeGroupImpl>(initial, tags, encoded, values)
    }

    fn data_len(&self, tags: &[u8]) -> usize {
        match self.0 {
            Impl::Scalar => group_impl::data_len::<scalar::UnsafeGroupImpl>(tags),
            Impl::Neon => group_impl::data_len::<neon::UnsafeGroupImpl>(tags),
        }
    }

    fn skip_deltas(&self, tags: &[u8], encoded: &[u8]) -> (usize, u64) {
        match self.0 {
            Impl::Scalar => group_impl::skip_deltas::<scalar::UnsafeGroupImpl>(tags, encoded),
            Impl::Neon => group_impl::skip_deltas::<neon::UnsafeGroupImpl>(tags, encoded),
        }
    }
}

#[cfg(test)]
crate::tests::group_test_suite!(Group64, Group1248);
