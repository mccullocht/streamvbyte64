#[cfg(all(target_arch = "aarch64", target_endian = "little"))]
mod neon;

use crate::{group_impl, Group32};

const TAG_LEN: [usize; 4] = [0, 1, 2, 4];
const TAG_MASK: [u32; 4] = crate::tag_utils::tag_mask_table32(TAG_LEN);

const TAG_VALUE_MAP: [usize; 5] = [0, 1, 2, 3, 3];
#[inline]
fn tag_value(v: u32) -> u8 {
    TAG_VALUE_MAP[4 - (v.leading_zeros() as usize / 8)] as u8
}

crate::unsafe_group::declare_scalar_implementation!(u32, group0124);

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
            Impl::Scalar => group_impl::encode::<scalar::UnsafeGroupImpl>(values, tags, encoded),
            Impl::Neon => group_impl::encode::<neon::UnsafeGroupImpl>(values, tags, encoded),
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
                group_impl::encode_deltas::<scalar::UnsafeGroupImpl>(initial, values, tags, encoded)
            }
            Impl::Neon => {
                group_impl::encode_deltas::<neon::UnsafeGroupImpl>(initial, values, tags, encoded)
            }
        }
    }

    fn decode(&self, tags: &[u8], encoded: &[u8], values: &mut [u32]) -> usize {
        match self.0 {
            Impl::Scalar => group_impl::decode::<scalar::UnsafeGroupImpl>(tags, encoded, values),
            Impl::Neon => group_impl::decode::<neon::UnsafeGroupImpl>(tags, encoded, values),
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
                group_impl::decode_deltas::<scalar::UnsafeGroupImpl>(initial, tags, encoded, values)
            }
            Impl::Neon => {
                group_impl::decode_deltas::<neon::UnsafeGroupImpl>(initial, tags, encoded, values)
            }
        }
    }

    fn data_len(&self, tags: &[u8]) -> usize {
        match self.0 {
            Impl::Scalar => group_impl::data_len::<scalar::UnsafeGroupImpl>(tags),
            Impl::Neon => group_impl::data_len::<neon::UnsafeGroupImpl>(tags),
        }
    }

    fn skip_deltas(&self, tags: &[u8], encoded: &[u8]) -> (usize, u32) {
        match self.0 {
            Impl::Scalar => group_impl::skip_deltas::<scalar::UnsafeGroupImpl>(tags, encoded),
            Impl::Neon => group_impl::skip_deltas::<neon::UnsafeGroupImpl>(tags, encoded),
        }
    }
}

// TODO: duplicated from other group impls; FIXME
#[cfg(test)]
mod tests {
    extern crate rand;

    use super::Group0124;
    use crate::Group32;
    use rand::distributions::Uniform;
    use rand::prelude::*;

    fn generate_array(len: usize, max_bytes: usize) -> Vec<u32> {
        assert!(max_bytes <= 4);
        let seed: &[u8; 32] = &[0xabu8; 32];
        let mut rng = StdRng::from_seed(*seed);
        let max_val = (0..max_bytes).fold(0u32, |acc, i| acc | (0xff << i * 8));
        let between = Uniform::from(0..=max_val);
        (0..len).map(|_| between.sample(&mut rng)).collect()
    }

    fn generate_cumulative_array(len: usize, max_bytes: usize, initial: u32) -> Vec<u32> {
        let mut values = generate_array(len, max_bytes);
        let mut cum = initial;
        for v in values.iter_mut() {
            cum = cum.wrapping_add(*v);
            *v = cum;
        }
        values
    }

    #[test]
    fn encode_decode() {
        let coder = Group0124::new();
        for max_bytes in [0, 1, 2, 4] {
            let expected = generate_array(65536, max_bytes);
            let (tbytes, dbytes) = Group0124::max_compressed_bytes(expected.len());
            let mut tags = vec![0u8; tbytes];
            let mut data = vec![0u8; dbytes];

            let data_len = coder.encode(&expected, &mut tags, &mut data);
            assert!(data_len <= max_bytes * expected.len());
            data.resize(data_len, 0u8);
            data.shrink_to_fit();
            assert_eq!(data_len, coder.data_len(&tags));
            let mut actual = vec![0u32; expected.len()];
            assert_eq!(data_len, coder.decode(&tags, &data, &mut actual));

            assert_eq!(expected.len(), actual.len(), "max_bytes={}", max_bytes);
            for i in 0..expected.len() {
                assert_eq!(
                    expected[i],
                    actual[i],
                    "Value mismatch max_bytes={} at index {} expected context={:?} actual context={:?}",
                    max_bytes,
                    i,
                    &expected[i.saturating_sub(5)..std::cmp::min(expected.len(), i + 5)],
                    &actual[i.saturating_sub(5)..std::cmp::min(expected.len(), i + 5)],
                );
            }
        }
    }

    #[test]
    fn encode_decode_deltas() {
        let coder = Group0124::new();
        for initial in 0u32..2 {
            for max_bytes in [0, 1, 2, 4] {
                let expected = generate_cumulative_array(65536, max_bytes, initial);
                let (tbytes, dbytes) = Group0124::max_compressed_bytes(expected.len());
                let mut tags = vec![0u8; tbytes];
                let mut data = vec![0u8; dbytes];

                let data_len = coder.encode_deltas(initial, &expected, &mut tags, &mut data);
                assert!(
                    data_len <= max_bytes * expected.len(),
                    "{} {}",
                    data_len,
                    max_bytes * expected.len()
                );
                data.resize(data_len, 0u8);
                data.shrink_to_fit();
                assert_eq!(
                    (data_len, expected.last().unwrap().wrapping_sub(initial)),
                    coder.skip_deltas(&tags, &data)
                );
                let mut actual = vec![0u32; expected.len()];
                assert_eq!(
                    data_len,
                    coder.decode_deltas(initial, &tags, &data, &mut actual)
                );

                assert_eq!(expected.len(), actual.len(), "max_bytes={}", max_bytes);
                for i in 0..expected.len() {
                    assert_eq!(
                        expected[i],
                        actual[i],
                        "Value mismatch max_bytes={} at index {} expected context={:?} actual context={:?}",
                        max_bytes,
                        i,
                        &expected[i.saturating_sub(5)..std::cmp::min(expected.len(), i + 5)],
                        &actual[i.saturating_sub(5)..std::cmp::min(expected.len(), i + 5)],
                    );
                }
            }
        }
    }

    // TODO: test boundary conditions on encode and decode unrolling.
}
