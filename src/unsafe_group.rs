use crunchy::unroll;
use num_traits::{ops::wrapping::WrappingAdd, PrimInt, Zero};
use std::fmt::Debug;

/// Represents a single group of 4 integers to compress or decompress.
/// Most implementations of this trait are unsafe in that they will read or write
/// the byte size of the group on every operation and need to be wrapped to behave
/// safely on arbitrary length slices.
pub(crate) trait UnsafeGroup: Sized + Copy + Debug {
    /// Element type used in each group.
    type Elem: PrimInt + Debug + WrappingAdd;

    /// Map from the two-bit tag value for a single value to the encoded length.
    /// All of the length values must be <= std::mem::sizeof::<Self::Elem>().
    const TAG_LEN: [u8; 4];

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
            .map(|tag| Self::data_len(tag) as usize)
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
pub(crate) unsafe fn default_decode8<G: UnsafeGroup>(
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
pub(crate) unsafe fn default_decode_deltas8<G: UnsafeGroup>(
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
pub(crate) unsafe fn default_skip_deltas8<G: UnsafeGroup>(
    input: *const u8,
    tag8: u64,
) -> (usize, G::Elem)
where
    <G as UnsafeGroup>::Elem: WrappingAdd,
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

/// Creates a scalar implementation for a element with and encoded byte width distribution.
/// The module must declare the following items at the module level:
///     fn tag_value(value: $elem) -> u8;
///     const TAG_LEN: [u8; 4];
///     const TAG_MASK: [$elem; 4];
macro_rules! declare_scalar_implementation {
    ($elem: ty, $distmod:ident) => {
        mod scalar {
            use crunchy::unroll;
            use num_traits::Zero;
            use std::ptr::{read_unaligned, write_unaligned};

            use crate::unsafe_group::UnsafeGroup;
            use crate::$distmod::{tag_value, TAG_LEN, TAG_MASK};

            const LENGTH_TABLE: [u8; 256] = crate::tag_utils::tag_length_table(TAG_LEN);

            #[derive(Clone, Copy, Debug)]
            pub(crate) struct UnsafeGroupImpl([$elem; 4]);

            impl UnsafeGroup for UnsafeGroupImpl {
                type Elem = $elem;

                const TAG_LEN: [u8; 4] = TAG_LEN;

                #[inline]
                fn set1(value: Self::Elem) -> Self {
                    UnsafeGroupImpl([value; 4])
                }

                #[inline]
                unsafe fn load_unaligned(ptr: *const Self::Elem) -> Self {
                    UnsafeGroupImpl(read_unaligned(ptr as *const [Self::Elem; 4]))
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
                            write_unaligned(output.add(written) as *mut Self::Elem, v);
                            let vtag = tag_value(v);
                            tag |= vtag << (i * 2);
                            written += Self::TAG_LEN[vtag as usize] as usize;
                        }
                    }
                    (tag, written)
                }

                #[inline]
                unsafe fn encode_deltas(output: *mut u8, base: Self, group: Self) -> (u8, usize) {
                    let deltas = UnsafeGroupImpl([
                        group.0[0].wrapping_sub(base.0[3]),
                        group.0[1].wrapping_sub(group.0[0]),
                        group.0[2].wrapping_sub(group.0[1]),
                        group.0[3].wrapping_sub(group.0[2]),
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
                            buf[i] = read_unaligned(input.add(read) as *const Self::Elem)
                                & TAG_MASK[vtag as usize];
                            read += Self::TAG_LEN[vtag as usize] as usize;
                        }
                    }
                    (read, UnsafeGroupImpl(buf))
                }

                #[inline]
                unsafe fn decode_deltas(input: *const u8, tag: u8, base: Self) -> (usize, Self) {
                    let (read, deltas) = Self::decode(input, tag);
                    let mut group = [Self::Elem::zero(); 4];
                    group[0] = base.0[3].wrapping_add(deltas.0[0]);
                    group[1] = group[0].wrapping_add(deltas.0[1]);
                    group[2] = group[1].wrapping_add(deltas.0[2]);
                    group[3] = group[2].wrapping_add(deltas.0[3]);
                    (read, UnsafeGroupImpl(group))
                }

                #[inline]
                fn data_len(tag: u8) -> usize {
                    LENGTH_TABLE[tag as usize] as usize
                }

                #[inline]
                unsafe fn skip_deltas(input: *const u8, tag: u8) -> (usize, Self::Elem) {
                    let (read, group) = Self::decode(input, tag);
                    (
                        read,
                        group
                            .0
                            .iter()
                            .fold(Self::Elem::zero(), |s, v| s.wrapping_add(*v)),
                    )
                }
            }

            #[cfg(test)]
            crate::tests::unsafe_group_test_suite!();
        }
    };
}

pub(crate) use declare_scalar_implementation;
