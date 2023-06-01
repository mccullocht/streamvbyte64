use num_traits::{ops::wrapping::WrappingAdd, PrimInt, WrappingSub};
use std::fmt::Debug;

/// `CodingDescriptor` captures the parameters of a particular coding: element size, value size for
/// each tag, and other settings.
pub(crate) trait CodingDescriptor: Debug + Copy {
    /// The input/output element type for this encoding.
    ///
    /// This is typically u32 or u64.
    type Elem: PrimInt + Debug + WrappingAdd + WrappingSub;

    /// Map from the two-bit tag value for a single value to the encoded length.
    /// All of the length values must be `<= std::mem::sizeof::<Self::Elem>()`.
    const TAG_LEN: [usize; 4];

    /// Maximum value that can be encoded for each 2-bit tag value.
    const TAG_MAX: [Self::Elem; 4];

    /// Returns the 2-bit tag value and number of bytes used to encode `value`.
    fn tag_value(value: Self::Elem) -> (u8, usize);

    /// Returns the number of bytes a group of 4 with the given `tag` occupies.
    fn data_len(tag: u8) -> usize;
}
