#[cfg(all(target_arch = "aarch64", target_endian = "little"))]
mod neon;
#[cfg(target_arch = "x86_64")]
mod ssse3;

use crate::coding_descriptor::CodingDescriptor;
use crate::{coder_impl, Coder};

#[derive(Copy, Clone, Debug)]
pub(crate) struct CodingDescriptor1248;

impl CodingDescriptor for CodingDescriptor1248 {
    type Elem = u64;

    const TAG_LEN: [usize; 4] = [1, 2, 4, 8];
    const TAG_MAX: [Self::Elem; 4] = crate::tag_utils::tag_mask_table64(Self::TAG_LEN);

    #[inline]
    fn tag_value(value: Self::Elem) -> (u8, usize) {
        // compute a 3-bit tag value in [0,7] in the same way we do for dist1234.
        // then take ~log2 to get a mapping from bytes required to length:
        // [0,1] => 0
        // [2,2] => 1
        // [3,4] => 2
        // [5,8] => 3
        let t3 = 7u32.saturating_sub(value.leading_zeros() / 8);
        let tag = u32::BITS - t3.leading_zeros();
        (tag as u8, Self::TAG_LEN[tag as usize])
    }

    #[inline(always)]
    fn data_len(tag: u8) -> usize {
        LENGTH_TABLE[tag as usize] as usize
    }
}
const LENGTH_TABLE: [u8; 256] = crate::tag_utils::tag_length_table(CodingDescriptor1248::TAG_LEN);

mod scalar {
    pub(crate) type RawGroupImpl =
        crate::raw_group::scalar::ScalarRawGroupImpl<super::CodingDescriptor1248>;

    #[cfg(test)]
    crate::tests::raw_group_test_suite!();
}

#[derive(Clone, Copy)]
enum Impl {
    Scalar,
    #[cfg(all(target_arch = "aarch64", target_endian = "little"))]
    Neon,
    #[cfg(all(target_arch = "x86_64"))]
    #[allow(dead_code)]
    SSSE3,
}

/// `Coder1248` packs 64-bit integers into lengths of 1, 2, 4, or 8 bytes.
///
/// This implementation has acceleration support on little-endian `aarch64` targets using `NEON` instructions.
#[derive(Clone, Copy)]
pub struct Coder1248(Impl);

impl Coder for Coder1248 {
    type Elem = u64;

    fn new() -> Self {
        #[cfg(all(target_arch = "aarch64", target_endian = "little"))]
        {
            if std::arch::is_aarch64_feature_detected!("neon") {
                return Coder1248(Impl::Neon);
            }
        }
        #[cfg(all(target_arch = "x86_64", target_feature = "sse3"))]
        {
            if std::arch::is_x86_feature_detected!("ssse3") {
                return Coder1234(Impl::SSSE3);
            }
        }
        Coder1248(Impl::Scalar)
    }

    fn encode(&self, values: &[u64], tags: &mut [u8], encoded: &mut [u8]) -> usize {
        match self.0 {
            Impl::Scalar => coder_impl::encode::<scalar::RawGroupImpl>(values, tags, encoded),
            #[cfg(all(target_arch = "aarch64", target_endian = "little"))]
            Impl::Neon => coder_impl::encode::<neon::RawGroupImpl>(values, tags, encoded),
            #[cfg(target_arch = "x86_64")]
            Impl::SSSE3 => coder_impl::encode::<scalar::RawGroupImpl>(values, tags, encoded),
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
                coder_impl::encode_deltas::<scalar::RawGroupImpl>(initial, values, tags, encoded)
            }
            #[cfg(all(target_arch = "aarch64", target_endian = "little"))]
            Impl::Neon => {
                coder_impl::encode_deltas::<neon::RawGroupImpl>(initial, values, tags, encoded)
            }
            #[cfg(target_arch = "x86_64")]
            Impl::SSSE3 => {
                coder_impl::encode_deltas::<scalar::RawGroupImpl>(initial, values, tags, encoded)
            }
        }
    }

    fn decode(&self, tags: &[u8], encoded: &[u8], values: &mut [u64]) -> usize {
        match self.0 {
            Impl::Scalar => coder_impl::decode::<scalar::RawGroupImpl>(tags, encoded, values),
            #[cfg(all(target_arch = "aarch64", target_endian = "little"))]
            Impl::Neon => coder_impl::decode::<neon::RawGroupImpl>(tags, encoded, values),
            #[cfg(target_arch = "x86_64")]
            Impl::SSSE3 => coder_impl::decode::<scalar::RawGroupImpl>(tags, encoded, values),
        }
    }

    fn decode_deltas(
        &self,
        initial: u64,
        tags: &[u8],
        encoded: &[u8],
        values: &mut [u64],
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
                coder_impl::decode_deltas::<scalar::RawGroupImpl>(initial, tags, encoded, values)
            }
        }
    }

    fn data_len(&self, tags: &[u8]) -> usize {
        match self.0 {
            Impl::Scalar => coder_impl::data_len::<scalar::RawGroupImpl>(tags),
            #[cfg(all(target_arch = "aarch64", target_endian = "little"))]
            Impl::Neon => coder_impl::data_len::<neon::RawGroupImpl>(tags),
            #[cfg(target_arch = "x86_64")]
            Impl::SSSE3 => coder_impl::data_len::<scalar::RawGroupImpl>(tags),
        }
    }

    fn skip_deltas(&self, tags: &[u8], encoded: &[u8]) -> (usize, u64) {
        match self.0 {
            Impl::Scalar => coder_impl::skip_deltas::<scalar::RawGroupImpl>(tags, encoded),
            #[cfg(all(target_arch = "aarch64", target_endian = "little"))]
            Impl::Neon => coder_impl::skip_deltas::<neon::RawGroupImpl>(tags, encoded),
            #[cfg(target_arch = "x86_64")]
            Impl::SSSE3 => coder_impl::skip_deltas::<scalar::RawGroupImpl>(tags, encoded),
        }
    }
}

#[cfg(test)]
crate::tests::coder_test_suite!(Coder1248, CodingDescriptor1248);
