#[cfg(all(target_arch = "aarch64", target_endian = "little"))]
mod neon;
#[cfg(target_arch = "x86_64")]
mod ssse3;

use crate::coder_impl;
use crate::coding_descriptor::CodingDescriptor;
use crate::Coder;

#[derive(Copy, Clone, Debug)]
pub(crate) struct CodingDescriptor1234;

impl CodingDescriptor for CodingDescriptor1234 {
    type Elem = u32;

    const TAG_LEN: [usize; 4] = [1, 2, 3, 4];
    const TAG_MAX: [Self::Elem; 4] = crate::tag_utils::tag_mask_table32(Self::TAG_LEN);

    #[inline]
    fn tag_value(value: Self::Elem) -> (u8, usize) {
        let tag = 3u32.saturating_sub(value.leading_zeros() / 8);
        (tag as u8, tag as usize + 1)
    }

    #[inline(always)]
    fn data_len(tag: u8) -> usize {
        LENGTH_TABLE[tag as usize] as usize
    }
}
const LENGTH_TABLE: [u8; 256] = crate::tag_utils::tag_length_table(CodingDescriptor1234::TAG_LEN);

mod scalar {
    pub(crate) type RawGroupImpl =
        crate::raw_group::scalar::ScalarRawGroupImpl<super::CodingDescriptor1234>;

    #[cfg(test)]
    crate::tests::raw_group_test_suite!();
}

#[derive(Clone, Copy)]
enum Impl {
    Scalar,
    #[cfg(all(target_arch = "aarch64", target_endian = "little"))]
    Neon,
    #[cfg(target_arch = "x86_64")]
    SSSE3,
}

/// `Coder1234` packs 32-bit integers into lengths of 1, 2, 3, or 4 bytes.
///
/// This implementation has acceleration support on little-endian `aarch64` targets using `NEON` instructions.
#[derive(Clone, Copy)]
pub struct Coder1234(Impl);

impl Coder for Coder1234 {
    type Elem = u32;

    fn new() -> Self {
        #[cfg(all(target_arch = "aarch64", target_endian = "little"))]
        {
            if std::arch::is_aarch64_feature_detected!("neon") {
                return Coder1234(Impl::Neon);
            }
        }
        #[cfg(target_arch = "x86_64")]
        {
            if std::arch::is_x86_feature_detected!("ssse3") {
                return Coder1234(Impl::SSSE3);
            }
        }
        Coder1234(Impl::Scalar)
    }

    fn encode(&self, values: &[u32], tags: &mut [u8], encoded: &mut [u8]) -> usize {
        match self.0 {
            Impl::Scalar => coder_impl::encode::<scalar::RawGroupImpl>(values, tags, encoded),
            #[cfg(all(target_arch = "aarch64", target_endian = "little"))]
            Impl::Neon => coder_impl::encode::<neon::RawGroupImpl>(values, tags, encoded),
            #[cfg(target_arch = "x86_64")]
            Impl::SSSE3 => coder_impl::encode::<ssse3::RawGroupImpl>(values, tags, encoded),
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
                coder_impl::encode_deltas::<scalar::RawGroupImpl>(initial, values, tags, encoded)
            }
            #[cfg(all(target_arch = "aarch64", target_endian = "little"))]
            Impl::Neon => {
                coder_impl::encode_deltas::<neon::RawGroupImpl>(initial, values, tags, encoded)
            }
            #[cfg(target_arch = "x86_64")]
            Impl::SSSE3 => {
                coder_impl::encode_deltas::<ssse3::RawGroupImpl>(initial, values, tags, encoded)
            }
        }
    }

    fn decode(&self, tags: &[u8], encoded: &[u8], values: &mut [u32]) -> usize {
        match self.0 {
            Impl::Scalar => coder_impl::decode::<scalar::RawGroupImpl>(tags, encoded, values),
            #[cfg(all(target_arch = "aarch64", target_endian = "little"))]
            Impl::Neon => coder_impl::decode::<neon::RawGroupImpl>(tags, encoded, values),
            #[cfg(target_arch = "x86_64")]
            Impl::SSSE3 => coder_impl::decode::<ssse3::RawGroupImpl>(tags, encoded, values),
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
                coder_impl::decode_deltas::<scalar::RawGroupImpl>(initial, tags, encoded, values)
            }
            #[cfg(all(target_arch = "aarch64", target_endian = "little"))]
            Impl::Neon => {
                coder_impl::decode_deltas::<neon::RawGroupImpl>(initial, tags, encoded, values)
            }
            #[cfg(target_arch = "x86_64")]
            Impl::SSSE3 => {
                coder_impl::decode_deltas::<ssse3::RawGroupImpl>(initial, tags, encoded, values)
            }
        }
    }

    fn data_len(&self, tags: &[u8]) -> usize {
        match self.0 {
            Impl::Scalar => coder_impl::data_len::<scalar::RawGroupImpl>(tags),
            #[cfg(all(target_arch = "aarch64", target_endian = "little"))]
            Impl::Neon => coder_impl::data_len::<neon::RawGroupImpl>(tags),
            #[cfg(target_arch = "x86_64")]
            Impl::SSSE3 => coder_impl::data_len::<ssse3::RawGroupImpl>(tags),
        }
    }

    fn skip_deltas(&self, tags: &[u8], encoded: &[u8]) -> (usize, u32) {
        match self.0 {
            Impl::Scalar => coder_impl::skip_deltas::<scalar::RawGroupImpl>(tags, encoded),
            #[cfg(all(target_arch = "aarch64", target_endian = "little"))]
            Impl::Neon => coder_impl::skip_deltas::<neon::RawGroupImpl>(tags, encoded),
            #[cfg(target_arch = "x86_64")]
            Impl::SSSE3 => coder_impl::skip_deltas::<ssse3::RawGroupImpl>(tags, encoded),
        }
    }
}

#[cfg(test)]
crate::tests::coder_test_suite!(Coder1234, CodingDescriptor1234);
