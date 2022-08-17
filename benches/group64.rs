use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use rand::distributions::Uniform;
use rand::prelude::*;
use stream_vbyte_64::{Group1248, Group64};

fn generate_array(len: usize, max_bytes: usize) -> Vec<u64> {
    assert!(max_bytes <= 4);
    let seed = &[0xabu8; 32];
    let mut rng = StdRng::from_seed(*seed);
    let max_val = (0..max_bytes).fold(0u64, |acc, i| acc | (0xff << i * 8));
    let between = Uniform::from(0..=max_val);
    (0..len).map(|_| between.sample(&mut rng)).collect()
}

fn generate_cumulative_array(len: usize, max_bytes: usize, initial: u64) -> Vec<u64> {
    let mut values = generate_array(len, max_bytes);
    let mut cum = initial;
    for v in values.iter_mut() {
        cum = cum.wrapping_add(*v);
        *v = cum;
    }
    values
}

struct Streams {
    len: usize,
    tags: Vec<u8>,
    data: Vec<u8>,
}

fn encoded_stream<G: Group64>(coder: &G, values: &Vec<u64>, delta: bool) -> Streams {
    let mut tags = vec![0u8; values.len() / 4];
    let mut data = vec![0u8; values.len() * 8];
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

fn bm_group64<G: Group64>(name: &str, c: &mut Criterion) {
    let coder = G::new();
    const NUM_ELEM: usize = 1024;
    let mut bm_group = c.benchmark_group(name);
    bm_group.throughput(Throughput::Elements(NUM_ELEM as u64));
    for max_bytes in [1, 4] {
        let input_values = generate_array(NUM_ELEM, max_bytes);
        bm_group.bench_with_input(
            BenchmarkId::new("encode", max_bytes),
            &input_values,
            |b, v| {
                let mut tags = vec![0u8; v.len() / 4];
                let mut data = vec![0u8; v.len() * 8];
                b.iter(|| assert!(coder.encode(&v, &mut tags, &mut data) > 0))
            },
        );

        let input_delta_values = generate_cumulative_array(NUM_ELEM, max_bytes, 1);
        bm_group.bench_with_input(
            BenchmarkId::new("encode_deltas", max_bytes),
            &input_delta_values,
            |b, v| {
                let mut tags = vec![0u8; v.len() / 4];
                let mut data = vec![0u8; v.len() * 8];
                b.iter(|| assert!(coder.encode_deltas(1, &v, &mut tags, &mut data) > 0))
            },
        );

        let encoded_streams = encoded_stream(&coder, &input_values, false);
        bm_group.bench_with_input(
            BenchmarkId::new("decode", max_bytes),
            &encoded_streams,
            |b, s| {
                let mut values = vec![0u64; s.len];
                b.iter(|| assert!(coder.decode(&s.tags, &s.data, &mut values) > 0))
            },
        );
        bm_group.bench_with_input(
            BenchmarkId::new("data_len", max_bytes),
            &encoded_streams,
            |b, s| b.iter(|| assert!(coder.data_len(&s.tags) > 0)),
        );

        let encoded_delta_streams = encoded_stream(&coder, &input_values, true);
        bm_group.bench_with_input(
            BenchmarkId::new("decode_deltas", max_bytes),
            &encoded_delta_streams,
            |b, s| {
                let mut values = vec![0u64; s.len];
                b.iter(|| assert!(coder.decode_deltas(1, &s.tags, &s.data, &mut values) > 0))
            },
        );
        bm_group.bench_with_input(
            BenchmarkId::new("skip_deltas", max_bytes),
            &encoded_delta_streams,
            |b, s| b.iter(|| assert!(coder.skip_deltas(&s.tags, &s.data) > (0, 0))),
        );
    }
    bm_group.finish();
}

fn benchmark(c: &mut Criterion) {
    bm_group64::<Group1248>("Group1248", c);
}

criterion_group!(benches, benchmark);
criterion_main!(benches);
