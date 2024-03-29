use crunchy::unroll;
use num_traits::{ops::wrapping::WrappingAdd, PrimInt, WrappingSub, Zero};
use std::fmt::Debug;

/// Implementation of streamvbyte coding in groups of 4 (or multiples of 4).
///
/// A RawGroup object represents a single group of exactly 4 integers of `Elem`, although methods
/// may be provided to code multiple groups at once into memory.
///
/// Many of these methods are unsafe as they read or write raw pointers or perform unaligned memory
/// operations.
pub(crate) trait RawGroup: Sized + Copy + Debug {
    /// Element type used in each group.
    type Elem: PrimInt + Debug + WrappingAdd + WrappingSub;

    /// Map from the two-bit tag value for a single value to the encoded length.
    /// All of the length values must be <= std::mem::sizeof::<Self::Elem>().
    const TAG_LEN: [usize; 4];

    /// Returns a group where all members of the group are set to value.
    fn set1(value: Self::Elem) -> Self;

    /// Load a group of 4 elements from ptr.
    unsafe fn load_unaligned(ptr: *const Self::Elem) -> Self;

    /// Store a group of 4 element to ptr.
    unsafe fn store_unaligned(ptr: *mut Self::Elem, group: Self);

    /// Encode the contents of group to output.
    /// Returns the one-byte tag for this group and the number of bytes written to output.
    ///
    /// _Safety_: this function may write up to std::mem::sizeof::<Self::Elem>() * 4 bytes.
    ///           this function may perform unaligned loads.
    unsafe fn encode(output: *mut u8, group: Self) -> (u8, usize);

    /// Encodes group as deltas against the last element of base to output.
    /// Returns the one-byte tag for this group and the number of bytes written to output.
    ///
    /// _Safety_: this function may write up to std::mem::sizeof::<Self::Elem>() * 4 bytes.
    ///           this function may perform unaligned loads.
    unsafe fn encode_deltas(output: *mut u8, base: Self, group: Self) -> (u8, usize);

    /// Decode the contents of the group with tag from input.
    /// Returns the decoded group and the number of bytes read from input.
    ///
    /// _Safety_: this function may read up to std::mem::sizeof::<Self::Elem>() * 4 bytes.
    ///           this function may perform unaligned loads.
    unsafe fn decode(input: *const u8, tag: u8) -> (usize, Self);

    // Decode the contents of the groups as deltas from base with tag from input.
    /// Returns the decoded group and the number of bytes read from input.
    ///
    /// _Safety_: this function may read up to std::mem::sizeof::<Self::Elem>() * 4 bytes.
    ///           this function may perform unaligned loads.
    unsafe fn decode_deltas(input: *const u8, tag: u8, base: Self) -> (usize, Self);

    /// Returns the number of bytes a group with the given tag occupies.
    fn data_len(tag: u8) -> usize;

    /// Skips the group of deltas at input with tag.
    /// Returns the number of input bytes and the sum of the delta values.
    ///
    /// _Safety_: this function may read up to std::mem::sizeof::<Self::Elem>() * 4 bytes.
    ///           this function may perform unaligned loads.
    unsafe fn skip_deltas(input: *const u8, tag: u8) -> (usize, Self::Elem);

    /// Decode 8 groups and write them to output.
    /// Returns the number of input bytes read.
    ///
    /// _Safety_: this function may read up to std::mem::sizeof::<Self::Elem>() * 32 bytes.
    ///           this function may perform unaligned loads.
    #[inline]
    unsafe fn decode8(input: *const u8, tag8: u64, output: *mut Self::Elem) -> usize {
        default_decode8::<Self>(input, tag8, output)
    }

    /// Decode 8 groups as deltas from a base value and write them to output.
    /// Returns the number of input bytes read and the last decoded group.
    ///
    /// _Safety_: this function may read up to std::mem::sizeof::<Self::Elem>() * 32 bytes.
    ///           this function may perform unaligned loads.
    #[inline]
    unsafe fn decode_deltas8(
        input: *const u8,
        tag8: u64,
        base: Self,
        output: *mut Self::Elem,
    ) -> (usize, Self) {
        default_decode_deltas8(input, tag8, base, output)
    }

    /// Returns the number of encoded bytes for 8 groups represented as a single 8-byte value.
    #[inline]
    fn data_len8(tag8: u64) -> usize {
        tag8.to_ne_bytes()
            .into_iter()
            .map(|tag| Self::data_len(tag))
            .sum()
    }

    /// Skip 8 groups as deltas.
    /// Returns the number of input bytes read and the sum of the value decoded.
    ///
    /// _Safety_: this function may read up to std::mem::sizeof::<Self::Elem>() * 32 bytes.
    ///           this function may perform unaligned loads.
    #[inline]
    unsafe fn skip_deltas8(input: *const u8, tag8: u64) -> (usize, Self::Elem) {
        default_skip_deltas8::<Self>(input, tag8)
    }
}

#[allow(dead_code)]
#[inline(always)]
pub(crate) unsafe fn default_decode8<G: RawGroup>(
    input: *const u8,
    tag8: u64,
    output: *mut G::Elem,
) -> usize {
    let mut read = 0usize;
    unroll! {
        for i in 0..8 {
            let tag = ((tag8 >> (i * 8)) & 0xff) as u8;
            let (r, group) = G::decode(input.add(read), tag);
            G::store_unaligned(output.add(i * 4), group);
            read += r;
        }
    }
    read
}

#[allow(dead_code)]
#[inline(always)]
pub(crate) unsafe fn default_decode_deltas8<G: RawGroup>(
    input: *const u8,
    tag8: u64,
    base: G,
    output: *mut G::Elem,
) -> (usize, G) {
    let mut read = 0usize;
    let mut prev = base;
    unroll! {
        for i in 0..8 {
            let tag = ((tag8 >> (i * 8)) & 0xff) as u8;
            let (r, group) = G::decode_deltas(input.add(read), tag, prev);
            G::store_unaligned(output.add(i * 4), group);
            read += r;
            prev = group;
        }
    }
    (read, prev)
}

// This may be dead on configurations without any SIMD implementations.
#[allow(dead_code)]
#[inline(always)]
pub(crate) unsafe fn default_skip_deltas8<G: RawGroup>(
    input: *const u8,
    tag8: u64,
) -> (usize, G::Elem)
where
    <G as RawGroup>::Elem: WrappingAdd,
{
    let mut read = 0usize;
    let mut sum = G::Elem::zero();
    unroll! {
        for i in 0..8 {
            let tag = ((tag8 >> (i * 8)) & 0xff) as u8;
            let (r, s) = G::skip_deltas(input.add(read), tag);
            read += r;
            sum = sum.wrapping_add(&s);
        }
    }
    (read, sum)
}

pub(crate) mod scalar {
    use crunchy::unroll;
    use num_traits::{ops::wrapping::WrappingAdd, ops::wrapping::WrappingSub, PrimInt, Zero};
    use std::ptr::{read_unaligned, write_unaligned};

    use super::RawGroup;
    use crate::coding_descriptor::CodingDescriptor;

    /// A scalar implementation of `RawGroup`.
    ///
    /// This implementation is essentially derived from the `CodingDescriptor` parameter which is
    /// used to adjust behavior based on tag length distribution and element type and does not use
    /// any sort of SIMD acceleration.
    #[derive(Copy, Clone, Debug)]
    pub(crate) struct ScalarRawGroupImpl<D: CodingDescriptor>([D::Elem; 4]);

    impl<D> RawGroup for ScalarRawGroupImpl<D>
    where
        D: CodingDescriptor,
    {
        type Elem = D::Elem;

        const TAG_LEN: [usize; 4] = D::TAG_LEN;

        #[inline]
        fn set1(value: Self::Elem) -> Self {
            ScalarRawGroupImpl([value; 4])
        }

        #[inline]
        unsafe fn load_unaligned(ptr: *const Self::Elem) -> Self {
            ScalarRawGroupImpl(read_unaligned(ptr as *const [Self::Elem; 4]))
        }

        #[inline]
        unsafe fn store_unaligned(ptr: *mut Self::Elem, group: Self) {
            write_unaligned(ptr as *mut [Self::Elem; 4], group.0)
        }

        #[inline]
        unsafe fn encode(output: *mut u8, group: Self) -> (u8, usize) {
            let mut tag = 0;
            let mut written = 0;
            unroll! {
                for i in 0..4 {
                    let v = group.0[i];
                    write_unaligned(output.add(written) as *mut Self::Elem, v.to_le());
                    let (vtag, len) = D::tag_value(v);
                    tag |= vtag << (i * 2);
                    written += len;
                }
            }
            (tag, written)
        }

        #[inline]
        unsafe fn encode_deltas(output: *mut u8, base: Self, group: Self) -> (u8, usize) {
            let deltas = ScalarRawGroupImpl([
                group.0[0].wrapping_sub(&base.0[3]),
                group.0[1].wrapping_sub(&group.0[0]),
                group.0[2].wrapping_sub(&group.0[1]),
                group.0[3].wrapping_sub(&group.0[2]),
            ]);
            Self::encode(output, deltas)
        }

        #[inline]
        unsafe fn decode(input: *const u8, tag: u8) -> (usize, Self) {
            let mut buf = [Self::Elem::zero(); 4];
            let mut read = 0usize;
            unroll! {
                for i in 0..4 {
                    let vtag = (tag >> (i * 2)) & 0x3;
                    buf[i] = Self::Elem::from_le(read_unaligned(input.add(read) as *const Self::Elem))
                        & D::TAG_MAX[vtag as usize];
                    read += Self::TAG_LEN[vtag as usize];
                }
            }
            (read, ScalarRawGroupImpl(buf))
        }

        #[inline]
        unsafe fn decode_deltas(input: *const u8, tag: u8, base: Self) -> (usize, Self) {
            let (read, deltas) = Self::decode(input, tag);
            let mut group = [Self::Elem::zero(); 4];
            group[0] = base.0[3].wrapping_add(&deltas.0[0]);
            group[1] = group[0].wrapping_add(&deltas.0[1]);
            group[2] = group[1].wrapping_add(&deltas.0[2]);
            group[3] = group[2].wrapping_add(&deltas.0[3]);
            (read, ScalarRawGroupImpl(group))
        }

        #[inline]
        fn data_len(tag: u8) -> usize {
            D::data_len(tag)
        }

        #[inline]
        unsafe fn skip_deltas(input: *const u8, tag: u8) -> (usize, Self::Elem) {
            let (read, group) = Self::decode(input, tag);
            (
                read,
                group
                    .0
                    .iter()
                    .fold(Self::Elem::zero(), |s, v| s.wrapping_add(v)),
            )
        }
    }
}
