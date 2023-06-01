#[cfg(all(target_arch = "aarch64", target_endian = "little"))]
mod neon;

use crate::coding_descriptor::CodingDescriptor;
use crate::{group_impl, Group32};

#[derive(Copy, Clone, Debug)]
pub(crate) struct CodingDescriptor0124;

impl CodingDescriptor for CodingDescriptor0124 {
    type Elem = u32;

    const TAG_LEN: [usize; 4] = [0, 1, 2, 4];
    const TAG_MAX: [Self::Elem; 4] = crate::tag_utils::tag_mask_table32(Self::TAG_LEN);

    #[inline]
    fn tag_value(value: Self::Elem) -> (u8, usize) {
        let tag = TAG_VALUE_MAP[4 - (value.leading_zeros() as usize / 8)];
        (tag as u8, Self::TAG_LEN[tag])
    }

    #[inline(always)]
    fn data_len(tag: u8) -> usize {
        LENGTH_TABLE[tag as usize] as usize
    }
}
const LENGTH_TABLE: [u8; 256] = crate::tag_utils::tag_length_table(CodingDescriptor0124::TAG_LEN);
const TAG_VALUE_MAP: [usize; 5] = [0, 1, 2, 3, 3];

mod scalar {
    use super::CodingDescriptor0124;

    pub(crate) type RawGroupImpl =
        crate::raw_group::scalar::ScalarRawGroupImpl<CodingDescriptor0124>;

    #[cfg(test)]
    crate::tests::raw_group_test_suite!();
}

#[derive(Clone, Copy)]
enum Impl {
    Scalar,
    #[cfg(all(target_arch = "aarch64", target_endian = "little"))]
    Neon,
}

/// `Group0124` packs groups of 4 32-bit integers into lengths of 0, 1, 2, or 4 bytes.
/// This implementation has acceleration support on little-endian `aarch64` targets using `NEON` instructions.
#[derive(Clone, Copy)]
pub struct Group0124(Impl);

impl Group32 for Group0124 {
    fn new() -> Self {
        #[cfg(all(target_arch = "aarch64", target_endian = "little"))]
        {
            if std::arch::is_aarch64_feature_detected!("neon") {
                return Group0124(Impl::Neon);
            }
        }
        Group0124(Impl::Scalar)
    }

    fn encode(&self, values: &[u32], tags: &mut [u8], encoded: &mut [u8]) -> usize {
        match self.0 {
            Impl::Scalar => group_impl::encode::<scalar::RawGroupImpl>(values, tags, encoded),
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
            Impl::Neon => {
                group_impl::encode_deltas::<neon::RawGroupImpl>(initial, values, tags, encoded)
            }
        }
    }

    fn decode(&self, tags: &[u8], encoded: &[u8], values: &mut [u32]) -> usize {
        match self.0 {
            Impl::Scalar => group_impl::decode::<scalar::RawGroupImpl>(tags, encoded, values),
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
            Impl::Neon => {
                group_impl::decode_deltas::<neon::RawGroupImpl>(initial, tags, encoded, values)
            }
        }
    }

    fn data_len(&self, tags: &[u8]) -> usize {
        match self.0 {
            Impl::Scalar => group_impl::data_len::<scalar::RawGroupImpl>(tags),
            Impl::Neon => group_impl::data_len::<neon::RawGroupImpl>(tags),
        }
    }

    fn skip_deltas(&self, tags: &[u8], encoded: &[u8]) -> (usize, u32) {
        match self.0 {
            Impl::Scalar => group_impl::skip_deltas::<scalar::RawGroupImpl>(tags, encoded),
            Impl::Neon => group_impl::skip_deltas::<neon::RawGroupImpl>(tags, encoded),
        }
    }
}

#[cfg(test)]
crate::tests::group_test_suite!(Group32, Group0124, CodingDescriptor0124);
