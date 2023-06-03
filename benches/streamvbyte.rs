use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use num_traits::{PrimInt, WrappingAdd};
use rand::distributions::{Uniform, WeightedIndex};
use rand::prelude::*;
use std::fmt;
use std::ops::RangeInclusive;
use streamvbyte64::{Group0124, Group1234, Group1248, Group32, Group64};

const ZIPF_WEIGHTS: [usize; 8] = [840, 420, 280, 210, 168, 140, 120, 105];
const ARRAY_LEN: usize = 1024;

fn range_for_byte_size(n: usize) -> RangeInclusive<u64> {
    match n {
        0 => 0..=0,
        1 => 0x1..=0xff,
        2 => 0x100..=0xffff,
        3 => 0x10000..=0xffffff,
        4 => 0x1000000..=0xffffffff,
        5 => 0x100000000..=0xffffffffff,
        6 => 0x10000000000..=0xffffffffffff,
        7 => 0x1000000000000..=0xffffffffffffff,
        8 => 0x100000000000000..=0xffffffffffffffff,
        _ => unreachable!(),
    }
}

// Generate an array of len with values no larger than max_bytes with a zipf-ian distribution.
fn generate_array<I: PrimInt>(len: usize, max_bytes: usize) -> Vec<I> {
    assert!(max_bytes <= std::mem::size_of::<I>());
    let mut len_rng = StdRng::from_seed([0xabu8; 32]);
    let len_dist = WeightedIndex::new(&ZIPF_WEIGHTS[..max_bytes]).unwrap();
    let mut value_rng = StdRng::from_seed([0xcdu8; 32]);
    len_dist
        .sample_iter(&mut len_rng)
        .take(len)
        .map(|n| Uniform::from(range_for_byte_size(n + 1)).sample(&mut value_rng))
        .map(|n| I::from(n).unwrap())
        .collect()
}

fn generate_cumulative_array<I: PrimInt + WrappingAdd>(
    len: usize,
    max_bytes: usize,
    initial: I,
) -> Vec<I> {
    let mut values = generate_array::<I>(len, max_bytes);
    let mut cum = initial;
    for v in values.iter_mut() {
        cum = cum.wrapping_add(v);
        *v = cum;
    }
    values
}

struct Streams {
    len: usize,
    tags: Vec<u8>,
    data: Vec<u8>,
}

struct BenchCase {
    len: usize,
    max_bytes: usize,
}

impl fmt::Display for BenchCase {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "len={}/max_bytes={}", self.len, self.max_bytes)
    }
}

macro_rules! bm_group_trait {
    ($bm_fn:ident, $elem_type:ty, $group_trait:path) => {
        fn $bm_fn<G>(name: &str, max_bytes: &[usize], c: &mut Criterion)
        where
            G: $group_trait,
        {
            fn encoded_stream<G: $group_trait>(
                coder: &G,
                values: &[$elem_type],
                delta: bool,
            ) -> Streams {
                let (tbytes, dbytes) = G::max_compressed_bytes(values.len());
                let mut tags = vec![0u8; tbytes];
                let mut data = vec![0u8; dbytes];
                let data_len = if delta {
                    coder.encode_deltas(1, &values, &mut tags, &mut data)
                } else {
                    coder.encode(&values, &mut tags, &mut data)
                };
                data.resize(data_len, 0);
                data.shrink_to_fit();
                Streams {
                    len: values.len(),
                    tags,
                    data,
                }
            }

            let coder = G::new();
            let mut bm_group = c.benchmark_group(name);
            bm_group.throughput(Throughput::Elements(ARRAY_LEN as u64));
            let max_data_len = ARRAY_LEN * std::mem::size_of::<$elem_type>();
            for max_bytes in max_bytes {
                let input_values = generate_array(ARRAY_LEN, *max_bytes);
                bm_group.bench_with_input(
                    BenchmarkId::new("encode", max_bytes),
                    &input_values,
                    |b, v| {
                        let (tbytes, dbytes) = G::max_compressed_bytes(v.len());
                        let mut tags = vec![0u8; tbytes];
                        let mut data = vec![0u8; dbytes];
                        b.iter(|| assert!(coder.encode(&v, &mut tags, &mut data) <= max_data_len))
                    },
                );

                let input_delta_values = generate_cumulative_array(ARRAY_LEN, *max_bytes, 1);
                bm_group.bench_with_input(
                    BenchmarkId::new("encode_deltas", max_bytes),
                    &input_delta_values,
                    |b, v| {
                        let (tbytes, dbytes) = G::max_compressed_bytes(v.len());
                        let mut tags = vec![0u8; tbytes];
                        let mut data = vec![0u8; dbytes];
                        b.iter(|| {
                            assert!(
                                coder.encode_deltas(1, &v, &mut tags, &mut data) <= max_data_len
                            )
                        })
                    },
                );

                let encoded_streams = encoded_stream(&coder, &input_values, false);
                bm_group.bench_with_input(
                    BenchmarkId::new("decode", max_bytes),
                    &encoded_streams,
                    |b, s| {
                        let mut values = vec![0; s.len];
                        b.iter(|| {
                            assert!(coder.decode(&s.tags, &s.data, &mut values) <= max_data_len)
                        })
                    },
                );
                let encoded_delta_streams = encoded_stream(&coder, &input_delta_values, true);
                bm_group.bench_with_input(
                    BenchmarkId::new("decode_deltas", max_bytes),
                    &encoded_delta_streams,
                    |b, s| {
                        let mut values = vec![0; s.len];
                        b.iter(|| {
                            assert!(
                                coder.decode_deltas(1, &s.tags, &s.data, &mut values)
                                    <= max_data_len
                            )
                        })
                    },
                );
                bm_group.bench_with_input(
                    BenchmarkId::new("skip_deltas", max_bytes),
                    &encoded_delta_streams,
                    |b, s| {
                        b.iter(|| assert!(coder.skip_deltas(&s.tags, &s.data).0 <= max_data_len))
                    },
                );

                bm_group.bench_with_input(
                    BenchmarkId::new("data_len", max_bytes),
                    &encoded_streams,
                    |b, s| b.iter(|| assert!(coder.data_len(&s.tags) <= max_data_len)),
                );
            }
            bm_group.finish();
        }
    };
}

bm_group_trait!(bm_group32, u32, Group32);
bm_group_trait!(bm_group64, u64, Group64);

fn benchmark(c: &mut Criterion) {
    bm_group32::<Group1234>("Group1234", &[1, 2, 4], c);
    bm_group32::<Group0124>("Group0124", &[1, 2, 4], c);
    bm_group64::<Group1248>("Group1248", &[1, 4, 8], c);
}

criterion_group!(benches, benchmark);
criterion_main!(benches);
