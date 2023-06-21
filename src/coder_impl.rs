use crate::raw_group::RawGroup;
use num_traits::{ops::wrapping::WrappingAdd, Zero};

trait EncodeSink<G>
where
    G: RawGroup,
{
    unsafe fn handle(&mut self, data: *mut u8, group: G) -> (u8, usize);
}

#[inline]
fn encode_to_sink<G: RawGroup, H: EncodeSink<G>>(
    values: &[G::Elem],
    tags: &mut [u8],
    data: &mut [u8],
    sink: &mut H,
) -> usize {
    assert_eq!(values.len() % 4, 0);
    let num_groups = values.len() / 4;
    assert!(num_groups <= tags.len());
    assert!(num_groups * G::TAG_LEN[3] * 4 <= data.len());

    let mut written = 0usize;
    for (input_group, output_tag) in values.chunks_exact(4).zip(tags.iter_mut()) {
        unsafe {
            let group = G::load_unaligned(input_group.as_ptr());
            let (t, l) = sink.handle(data.as_mut_ptr().add(written), group);
            *output_tag = t;
            written += l;
        }
    }

    written
}

struct StandardEncodeSink;

impl<G> EncodeSink<G> for StandardEncodeSink
where
    G: RawGroup,
{
    #[inline]
    unsafe fn handle(&mut self, data: *mut u8, group: G) -> (u8, usize) {
        G::encode(data, group)
    }
}

pub(crate) fn encode<G: RawGroup>(
    values: &[G::Elem],
    tags: &mut [u8],
    encoded: &mut [u8],
) -> usize {
    encode_to_sink::<G, _>(values, tags, encoded, &mut StandardEncodeSink)
}

struct DeltaEncodeSink<G>(G)
where
    G: RawGroup;

impl<G> EncodeSink<G> for DeltaEncodeSink<G>
where
    G: RawGroup,
{
    #[inline]
    unsafe fn handle(&mut self, data: *mut u8, group: G) -> (u8, usize) {
        let r = G::encode_deltas(data, self.0, group);
        self.0 = group;
        r
    }
}

#[inline]
pub(crate) fn encode_deltas<G: RawGroup>(
    initial: G::Elem,
    values: &[G::Elem],
    tags: &mut [u8],
    encoded: &mut [u8],
) -> usize {
    encode_to_sink::<G, _>(
        values,
        tags,
        encoded,
        &mut DeltaEncodeSink(G::set1(initial)),
    )
}

trait DecodeSink {
    unsafe fn handle1(&mut self, tag_index: usize, tag: u8, data: *const u8) -> usize;
    unsafe fn handle8(&mut self, tag_index: usize, tag8: u64, data: *const u8) -> usize;
}

#[inline]
fn decode_to_sink<G: RawGroup, S: DecodeSink>(tags: &[u8], data: &[u8], sink: &mut S) -> usize {
    let mut read = 0usize;
    let mut tag_index = 0;
    for tag_chunk in tags.chunks_exact(8) {
        let tag8 = unsafe { std::ptr::read_unaligned(tag_chunk.as_ptr() as *const u64) };
        // G will read the the length of the encoded data plus some additional data for the last group.
        // Break out to group-by-group decoding if there not enough space left in the data buffer.
        // TODO: add a slop constant to RawGroup; would be smaller for scalar implementations (size_of<Elem>() - 1)
        let max_read = G::data_len8(tag8) + (G::TAG_LEN[3] - G::TAG_LEN[0]) * 4;
        if read + max_read > data.len() {
            break;
        }
        read += unsafe { sink.handle8(tag_index, tag8, data.as_ptr().add(read)) };
        tag_index += 8;
    }

    for tag in &tags[tag_index..] {
        if read + G::TAG_LEN[3] * 4 > data.len() {
            break;
        }
        read += unsafe { sink.handle1(tag_index, *tag, data.as_ptr().add(read)) };
        tag_index += 1;
    }

    let remainder = &tags[tag_index..];
    if !remainder.is_empty() {
        // read <= data.len() as groups with the smallest tag len of 0 may still decode an empty buffer.
        assert!(read <= data.len());
        // data contains at most G::TAG_LEN[3] * 4 bytes, so allocate a scratch buffer that is double that length and copy so
        // that we can continue to use "unsafe" loads. Note that G::TAG_LEN[3] is not const in this context so we just insert a
        // value that should work if elements are 8 bytes wide.
        let mut buf = [0u8; 64];
        buf[..(data.len() - read)].copy_from_slice(&data[read..]);
        let mut bufr = 0usize;
        for tag in remainder {
            // Anything beyond the max read length of one group did not appear in the input data and should not be used.
            assert!(bufr < G::TAG_LEN[3] * 4);
            bufr += unsafe { sink.handle1(tag_index, *tag, buf.as_ptr().add(bufr)) };
            tag_index += 1;
        }
        read += bufr;
        // In this case we may read beyond the end of data as we copied some bytes into a larger buffer.
        // assert to ensure that those extra bytes were not consumed and included in output.
        assert!(read <= data.len());
    }

    debug_assert_eq!(tag_index, tags.len());

    read
}

struct StandardDecodeSink<G>(*mut G::Elem)
where
    G: RawGroup;

impl<G> DecodeSink for StandardDecodeSink<G>
where
    G: RawGroup,
{
    #[inline]
    unsafe fn handle1(&mut self, tag_index: usize, tag: u8, data: *const u8) -> usize {
        let (read, group) = G::decode(data, tag);
        G::store_unaligned(self.0.add(tag_index * 4), group);
        read
    }

    #[inline]
    unsafe fn handle8(&mut self, tag_index: usize, tag8: u64, data: *const u8) -> usize {
        G::decode8(data, tag8, self.0.add(tag_index * 4))
    }
}

#[inline]
pub(crate) fn decode<G: RawGroup>(tags: &[u8], encoded: &[u8], values: &mut [G::Elem]) -> usize {
    assert_eq!(values.len() % 4, 0);
    assert!(tags.len() >= values.len() / 4);
    decode_to_sink::<G, _>(
        tags,
        encoded,
        &mut StandardDecodeSink::<G>(values.as_mut_ptr()),
    )
}

struct DeltaDecodeSink<G>(*mut G::Elem, G)
where
    G: RawGroup;

impl<G> DecodeSink for DeltaDecodeSink<G>
where
    G: RawGroup,
{
    #[inline]
    unsafe fn handle1(&mut self, tag_index: usize, tag: u8, data: *const u8) -> usize {
        let (read, group) = G::decode_deltas(data, tag, self.1);
        G::store_unaligned(self.0.add(tag_index * 4), group);
        self.1 = group;
        read
    }

    #[inline]
    unsafe fn handle8(&mut self, tag_index: usize, tag8: u64, data: *const u8) -> usize {
        let (read, group) = G::decode_deltas8(data, tag8, self.1, self.0.add(tag_index * 4));
        self.1 = group;
        read
    }
}

#[inline]
pub(crate) fn decode_deltas<G: RawGroup>(
    initial: G::Elem,
    tags: &[u8],
    encoded: &[u8],
    values: &mut [G::Elem],
) -> usize {
    assert_eq!(values.len() % 4, 0);
    assert!(tags.len() >= values.len() / 4);
    decode_to_sink::<G, _>(
        tags,
        encoded,
        &mut DeltaDecodeSink::<G>(values.as_mut_ptr(), G::set1(initial)),
    )
}

#[inline]
pub(crate) fn data_len<G: RawGroup>(tags: &[u8]) -> usize {
    let mut len = 0usize;
    let chunks = tags.chunks_exact(8);
    for tag in chunks.remainder() {
        len += G::data_len(*tag);
    }
    for tag8_bytes in chunks {
        let tag8 = unsafe { std::ptr::read_unaligned(tag8_bytes.as_ptr() as *const u64) };
        len += G::data_len8(tag8)
    }
    len
}

struct SkipDeltasSink<G: RawGroup>(G::Elem);

impl<G> DecodeSink for SkipDeltasSink<G>
where
    G: RawGroup,
    <G as RawGroup>::Elem: WrappingAdd,
{
    #[inline]
    unsafe fn handle1(&mut self, _tag_index: usize, tag: u8, data: *const u8) -> usize {
        let (r, s) = G::skip_deltas(data, tag);
        self.0 = self.0.wrapping_add(&s);
        r
    }

    #[inline]
    unsafe fn handle8(&mut self, _tag_index: usize, tag8: u64, data: *const u8) -> usize {
        let (r, s) = G::skip_deltas8(data, tag8);
        self.0 = self.0.wrapping_add(&s);
        r
    }
}

#[inline]
pub(crate) fn skip_deltas<G: RawGroup>(tags: &[u8], data: &[u8]) -> (usize, G::Elem)
where
    <G as RawGroup>::Elem: WrappingAdd,
{
    let mut sink = SkipDeltasSink::<G>(G::Elem::zero());
    let read = decode_to_sink::<G, _>(tags, data, &mut sink);
    (read, sink.0)
}
