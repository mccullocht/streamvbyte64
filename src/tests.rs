use crate::unsafe_group::UnsafeGroup;
use num_traits::{One, PrimInt, Zero};
use std::iter::Iterator;

pub(crate) fn test_tag_len<TGroup: UnsafeGroup>() {
    assert!(TGroup::TAG_LEN
        .iter()
        .zip(TGroup::TAG_LEN.iter().skip(1))
        .all(|(a, b)| *a < *b));
    assert!(TGroup::TAG_LEN
        .iter()
        .all(|v| *v as usize <= std::mem::size_of::<TGroup::Elem>()));
}

/// Represents a single test group for encoding or decoding.
pub(crate) struct TestGroup<Elem>
where
    Elem: PrimInt,
{
    /// Contents of the group.
    pub group: [Elem; 4],
    /// Tag value associated with the group.
    pub tag: u8,
    /// Length of the group when encoded.
    pub data_len: usize,
}

/// Yields all possible tag values and a group that fits that tag profile.
/// dist must be sorted and values must be <= std::mem::sizeof::<Elem>().
pub(crate) struct TagIter<Elem>
where
    Elem: PrimInt,
{
    tag: u8,
    dist: [usize; 4],
    masks: [Elem; 4],
}

impl<Elem> TagIter<Elem>
where
    Elem: PrimInt,
{
    pub fn new(dist: [usize; 4], masks: [Elem; 4]) -> Self {
        assert!(dist.iter().zip(dist.iter().skip(1)).all(|(a, b)| *a < *b));
        assert!(dist
            .iter()
            .all(|v| *v as usize <= std::mem::size_of::<Elem>()));

        TagIter {
            tag: 0,
            dist,
            masks,
        }
    }

    pub fn mask_patterns(patterns: [u8; 4]) -> [Elem; 4] {
        let size = std::mem::size_of::<Elem>();
        let mut masks = [Elem::zero(); 4];
        for (pat, mask) in patterns.into_iter().zip(masks.iter_mut()) {
            for i in 0..size {
                *mask = *mask | (Elem::from(pat).unwrap() << (i * 8));
            }
        }
        masks
    }

    fn byteval(nbytes: usize) -> Elem {
        let size = std::mem::size_of::<Elem>();
        debug_assert!(nbytes <= size);
        if nbytes == size {
            !Elem::zero()
        } else {
            (Elem::one() << (nbytes * 8)) - Elem::one()
        }
    }
}

impl<Elem> Iterator for TagIter<Elem>
where
    Elem: PrimInt,
{
    type Item = TestGroup<Elem>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.tag == 255 {
            return None;
        }

        let tag = self.tag;
        let vlens = [
            self.dist[tag as usize & 0x3],
            self.dist[(tag as usize >> 2) & 0x3],
            self.dist[(tag as usize >> 4) & 0x3],
            self.dist[(tag as usize >> 6) & 0x3],
        ];
        let group = [
            Self::byteval(vlens[0] as usize) & self.masks[0],
            Self::byteval(vlens[1] as usize) & self.masks[1],
            Self::byteval(vlens[2] as usize) & self.masks[2],
            Self::byteval(vlens[3] as usize) & self.masks[3],
        ];

        self.tag += 1;
        Some(TestGroup {
            group,
            tag,
            data_len: vlens.into_iter().sum::<usize>(),
        })
    }
}

fn extract_group<TGroup: UnsafeGroup>(group: TGroup) -> [TGroup::Elem; 4] {
    let mut buf = [TGroup::Elem::zero(); 4];
    unsafe {
        TGroup::store_unaligned(buf.as_mut_ptr(), group);
    }
    buf
}

// Test encoding with EGroup and decoding with DGroup
pub(crate) fn test_encode_decode<EGroup: UnsafeGroup, DGroup: UnsafeGroup>()
where
    <EGroup as UnsafeGroup>::Elem: PartialEq<<DGroup as UnsafeGroup>::Elem>,
{
    unsafe {
        for test in TagIter::<EGroup::Elem>::new(
            EGroup::TAG_LEN,
            TagIter::<EGroup::Elem>::mask_patterns([0x1a, 0x1b, 0x1c, 0x1d]),
        ) {
            let egroup = EGroup::load_unaligned(test.group.as_ptr());
            let mut enc = [255u8; 64];
            let (etag, elen) = EGroup::encode(enc.as_mut_ptr(), egroup);
            assert_eq!(test.tag, etag);
            assert!(elen <= enc.len());
            assert_eq!(test.data_len, elen);
            assert_eq!(elen, EGroup::data_len(etag));

            let (dlen, dgroup) = DGroup::decode(enc.as_ptr(), test.tag);
            assert_eq!(elen, dlen);
            assert_eq!(test.group, extract_group(dgroup));
        }
    }
}

// Integrates deltas to produce a running sum at each entry.
fn integrate_delta<Elem: PrimInt>(base: Elem, delta: [Elem; 4]) -> [Elem; 4] {
    [
        base + delta[0],
        base + delta[0] + delta[1],
        base + delta[0] + delta[1] + delta[2],
        base + delta[0] + delta[1] + delta[2] + delta[3],
    ]
}

// Returns a mask that contains the smallest element of each tag byte length.
fn smol_mask<TGroup: UnsafeGroup>() -> TGroup::Elem {
    let mut base_mask = TGroup::Elem::one();
    for l in TGroup::TAG_LEN.iter().skip_while(|l| **l == 0).take(3) {
        base_mask = base_mask | (TGroup::Elem::one() << (*l as usize * 8));
    }
    base_mask
}

// Test encoding deltas with EGroup and decoding them with DGroup
pub(crate) fn test_encode_decode_deltas<EGroup: UnsafeGroup, DGroup: UnsafeGroup>()
where
    <EGroup as UnsafeGroup>::Elem: PartialEq<<DGroup as UnsafeGroup>::Elem>,
{
    unsafe {
        for test in TagIter::<EGroup::Elem>::new(EGroup::TAG_LEN, [smol_mask::<EGroup>(); 4]) {
            let base = EGroup::Elem::one();
            let integrated = integrate_delta(base, test.group);
            let egroup = EGroup::load_unaligned(integrated.as_ptr());
            let mut enc = [255u8; 64];
            let (etag, elen) = EGroup::encode_deltas(enc.as_mut_ptr(), EGroup::set1(base), egroup);
            assert_eq!(test.tag, etag);
            assert_eq!(test.data_len, elen);
            assert!(elen <= enc.len());
            assert_eq!(elen, EGroup::data_len(etag));

            let (dlen, dgroup) =
                DGroup::decode_deltas(enc.as_ptr(), test.tag, DGroup::set1(DGroup::Elem::one()));
            assert_eq!(elen, dlen);
            assert_eq!(integrated, extract_group(dgroup));
        }
    }
}

// Test encoding deltas with EGroup and skipping them with DGroup.
pub(crate) fn test_skip_deltas<EGroup: UnsafeGroup, DGroup: UnsafeGroup>()
where
    <EGroup as UnsafeGroup>::Elem: PartialEq<<DGroup as UnsafeGroup>::Elem>,
{
    unsafe {
        for test in TagIter::<EGroup::Elem>::new(EGroup::TAG_LEN, [smol_mask::<EGroup>(); 4]) {
            let base = EGroup::Elem::one();
            let integrated = integrate_delta(base, test.group);
            let egroup = EGroup::load_unaligned(integrated.as_ptr());
            let mut enc = [0u8; 64];
            let (_, enc_len) = EGroup::encode_deltas(enc.as_mut_ptr(), EGroup::set1(base), egroup);

            let (skip_len, sum) = DGroup::skip_deltas(enc.as_ptr(), test.tag);
            assert_eq!(enc_len, skip_len);
            let mut expected_sum = test.group[0];
            for i in 1..4 {
                expected_sum = expected_sum + test.group[i];
            }
            assert_eq!(expected_sum, sum);
        }
    }
}

// Test encoding with EGroup and decoding a superblock with DGroup.
pub(crate) fn test_decode8<EGroup: UnsafeGroup, DGroup: UnsafeGroup>()
where
    <EGroup as UnsafeGroup>::Elem: PartialEq<<DGroup as UnsafeGroup>::Elem>,
{
    unsafe {
        let test_groups = TagIter::<EGroup::Elem>::new(EGroup::TAG_LEN, [smol_mask::<EGroup>(); 4])
            .collect::<Vec<_>>();
        for group8 in test_groups.chunks_exact(8) {
            let group_tags = group8.iter().map(|tg| tg.tag).collect::<Vec<_>>();
            let group_values = group8
                .iter()
                .map(|tg| tg.group)
                .flatten()
                .collect::<Vec<_>>();
            let expected_len = group8.iter().map(|tg| tg.data_len).sum::<usize>();

            let mut enc = [0u8; 256];
            let mut etag = Vec::<u8>::with_capacity(8);
            let mut elen = 0usize;
            for group in group_values.chunks_exact(4) {
                let egroup = EGroup::load_unaligned(group.as_ptr());
                let (t, l) = EGroup::encode(enc.as_mut_ptr().add(elen), egroup);
                etag.push(t);
                elen += l;
            }

            assert_eq!(group_tags, etag);
            assert_eq!(expected_len, elen);

            let mut tag8_bytes = [0u8; 8];
            tag8_bytes.copy_from_slice(&etag);
            let tag8 = u64::from_le_bytes(tag8_bytes);
            assert_eq!(elen, DGroup::data_len8(tag8));

            let mut actual = [DGroup::Elem::zero(); 32];
            let dlen = DGroup::decode8(enc.as_ptr(), tag8, actual.as_mut_ptr());
            assert_eq!(elen, dlen);
            assert_eq!(group_values, actual);
        }
    }
}

// Test encoding deltas with EGroup and decoding a superblock with DGroup.
pub(crate) fn test_decode_deltas8<EGroup: UnsafeGroup, DGroup: UnsafeGroup>()
where
    <EGroup as UnsafeGroup>::Elem: PartialEq<<DGroup as UnsafeGroup>::Elem>,
{
    unsafe {
        let test_groups = TagIter::<EGroup::Elem>::new(EGroup::TAG_LEN, [smol_mask::<EGroup>(); 4])
            .collect::<Vec<_>>();
        for group8 in test_groups.chunks_exact(8) {
            let group_tags = group8.iter().map(|tg| tg.tag).collect::<Vec<_>>();
            let mut sum = EGroup::Elem::one();
            let group_values = group8
                .iter()
                .map(|tg| tg.group)
                .flatten()
                .map(|v| {
                    sum = sum + v;
                    sum
                })
                .collect::<Vec<_>>();
            let expected_len = group8.iter().map(|tg| tg.data_len).sum::<usize>();

            let mut enc = [255u8; 256];
            let mut etag = Vec::<u8>::with_capacity(8);
            let mut elen = 0usize;
            let mut ebase = EGroup::set1(EGroup::Elem::one());
            for group in group_values.chunks_exact(4) {
                let egroup = EGroup::load_unaligned(group.as_ptr());
                let (t, l) = EGroup::encode_deltas(enc.as_mut_ptr().add(elen), ebase, egroup);
                etag.push(t);
                elen += l;
                ebase = egroup;
            }

            assert_eq!(group_tags, etag);
            assert_eq!(expected_len, elen);

            let mut tag8_bytes = [0u8; 8];
            tag8_bytes.copy_from_slice(&etag);
            let tag8 = u64::from_le_bytes(tag8_bytes);
            assert_eq!(elen, DGroup::data_len8(tag8));

            let mut actual = [DGroup::Elem::zero(); 32];
            let (dlen, dgroup) = DGroup::decode_deltas8(
                enc.as_ptr(),
                tag8,
                DGroup::set1(DGroup::Elem::one()),
                actual.as_mut_ptr(),
            );
            assert_eq!(elen, dlen);
            assert_eq!(group_values, actual);
            assert_eq!(actual[(7 * 4)..], extract_group(dgroup));
        }
    }
}

// Test encoding deltas with EGroup and skipping a superblock with DGroup.
pub(crate) fn test_skip_deltas8<EGroup: UnsafeGroup, DGroup: UnsafeGroup>()
where
    <EGroup as UnsafeGroup>::Elem: PartialEq<<DGroup as UnsafeGroup>::Elem>,
{
    unsafe {
        let test_groups = TagIter::<EGroup::Elem>::new(EGroup::TAG_LEN, [smol_mask::<EGroup>(); 4])
            .collect::<Vec<_>>();
        for group8 in test_groups.chunks_exact(8) {
            let group_tags = group8.iter().map(|tg| tg.tag).collect::<Vec<_>>();
            let mut sum = EGroup::Elem::one();
            let group_values = group8
                .iter()
                .map(|tg| tg.group)
                .flatten()
                .map(|v| {
                    sum = sum + v;
                    sum
                })
                .collect::<Vec<_>>();
            let expected_len = group8.iter().map(|tg| tg.data_len).sum::<usize>();

            let mut enc = [255u8; 256];
            let mut etag = Vec::<u8>::with_capacity(8);
            let mut elen = 0usize;
            let mut ebase = EGroup::set1(EGroup::Elem::one());
            for group in group_values.chunks_exact(4) {
                let egroup = EGroup::load_unaligned(group.as_ptr());
                let (t, l) = EGroup::encode_deltas(enc.as_mut_ptr().add(elen), ebase, egroup);
                etag.push(t);
                elen += l;
                ebase = egroup;
            }

            assert_eq!(group_tags, etag);
            assert_eq!(expected_len, elen);

            let mut tag8_bytes = [0u8; 8];
            tag8_bytes.copy_from_slice(&etag);
            let tag8 = u64::from_le_bytes(tag8_bytes);

            let (dlen, dsum) = DGroup::skip_deltas8(enc.as_ptr(), tag8);
            assert_eq!(elen, dlen);
            let expected_sum = group8
                .iter()
                .map(|tg| tg.group[0] + tg.group[1] + tg.group[2] + tg.group[3])
                .fold(EGroup::Elem::zero(), |acc, v| acc + v);
            assert_eq!(expected_sum, dsum);
        }
    }
}

/// Define `group_suite` module with conformance tests for `UnsafeGroup` implementations.
/// Invoke this inside the module defining your `UnsafeGroupImpl`.
macro_rules! unsafe_group_test_suite {
    () => {
        #[cfg(test)]
        mod group_suite {
            use super::UnsafeGroupImpl;

            #[test]
            fn tag_len() {
                crate::tests::test_tag_len::<UnsafeGroupImpl>();
            }

            #[test]
            fn encode_decode() {
                crate::tests::test_encode_decode::<UnsafeGroupImpl, UnsafeGroupImpl>();
            }

            #[test]
            fn encode_decode_deltas() {
                crate::tests::test_encode_decode_deltas::<UnsafeGroupImpl, UnsafeGroupImpl>();
            }

            #[test]
            fn skip_deltas() {
                crate::tests::test_skip_deltas::<UnsafeGroupImpl, UnsafeGroupImpl>();
            }

            #[test]
            fn decode8() {
                crate::tests::test_decode8::<UnsafeGroupImpl, UnsafeGroupImpl>();
            }

            #[test]
            fn decode_deltas8() {
                crate::tests::test_decode_deltas8::<UnsafeGroupImpl, UnsafeGroupImpl>();
            }

            #[test]
            fn skip_deltas8() {
                crate::tests::test_skip_deltas8::<UnsafeGroupImpl, UnsafeGroupImpl>();
            }
        }
    };
}

pub(crate) use unsafe_group_test_suite;

/// Define `compat_suite` module that ensures your `UnsafeGroup` implementation is compatible with the scalar implementation.
/// Invoke this inside the module defining your `UnsafeGroupImpl`.
macro_rules! compat_test_suite {
    () => {
        #[cfg(test)]
        mod compat_suite {
            use super::scalar::UnsafeGroupImpl as ScalarGroupImpl;
            use super::UnsafeGroupImpl as SIMDGroupImpl;

            // These tests invoke shared test method but vary the encoding and decoding group implementations.

            #[test]
            fn encode_decode() {
                crate::tests::test_encode_decode::<ScalarGroupImpl, SIMDGroupImpl>();
                crate::tests::test_encode_decode::<SIMDGroupImpl, ScalarGroupImpl>();
            }

            #[test]
            fn encode_decode_deltas() {
                crate::tests::test_encode_decode_deltas::<ScalarGroupImpl, SIMDGroupImpl>();
                crate::tests::test_encode_decode_deltas::<SIMDGroupImpl, ScalarGroupImpl>();
            }

            #[test]
            fn skip_deltas() {
                crate::tests::test_skip_deltas::<ScalarGroupImpl, SIMDGroupImpl>();
                crate::tests::test_skip_deltas::<SIMDGroupImpl, ScalarGroupImpl>();
            }

            #[test]
            fn decode8() {
                crate::tests::test_decode8::<ScalarGroupImpl, SIMDGroupImpl>();
                crate::tests::test_decode8::<SIMDGroupImpl, ScalarGroupImpl>();
            }

            #[test]
            fn decode_deltas8() {
                crate::tests::test_decode_deltas8::<ScalarGroupImpl, SIMDGroupImpl>();
                crate::tests::test_decode_deltas8::<SIMDGroupImpl, ScalarGroupImpl>();
            }

            #[test]
            fn skip_deltas8() {
                crate::tests::test_skip_deltas8::<ScalarGroupImpl, SIMDGroupImpl>();
                crate::tests::test_skip_deltas8::<SIMDGroupImpl, ScalarGroupImpl>();
            }
        }
    };
}

pub(crate) use compat_test_suite;
