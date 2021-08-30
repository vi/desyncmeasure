fn get_buckets() -> [[usize; 4]; 7] {
    let mut ret = [[0; 4]; 7];
    let mut freq = 830;
    for b in &mut ret {
        for x in b {
            *x = freq / 10;
            freq += 87;
        }
    }
    dbg!(freq);
    ret
}

fn encode_bucket(x: usize) -> [f32; 4] {
    assert!(x < 4);
    match x {
        0 => [1.0, 0.0, 0.0, 0.0],
        1 => [0.0, 1.0, 0.0, 0.0],
        2 => [0.0, 0.0, 1.0, 0.0],
        3 => [0.0, 0.0, 0.0, 1.0],
        _ => unreachable!(),
    }
}

fn encode_number(x: usize) -> [[f32; 4]; 7] {
    assert!(x < 4096);
    let mut cfs = [[0.0f32; 4]; 7];
    cfs[0] = encode_bucket((x & 0b000000000011) >> 0);
    cfs[1] = encode_bucket((x & 0b000000001100) >> 2);
    cfs[2] = encode_bucket((x & 0b000000110000) >> 4);
    cfs[3] = encode_bucket((x & 0b000011000000) >> 6);
    cfs[4] = encode_bucket((x & 0b001100000000) >> 8);
    cfs[5] = encode_bucket((x & 0b110000000000) >> 10);

    let parity = (x & 0b101010101010).count_ones() & 1 | ((x & 0b010101010101).count_ones() & 1 << 1);
    cfs[6] = encode_bucket(parity as usize);
    cfs
}

fn main() {
    let buckets = asyncmeasure::chirps::get_buckets();

    let zeroes = [0.0f32; 800];

    let mut audio_data = Vec::with_capacity(800 * 2 * 8192);

    let fft = rustfft::FftPlanner::<f32>::new().plan_fft_forward(800);


    for x in 0..4096 {
        let mut coefs = [num_complex::Complex32::new(0.0, 0.0); 800];
        let data = encode_number(x);
        for (bn, n) in IntoIterator::into_iter(data).enumerate() {
            for (sn, sv) in IntoIterator::into_iter(n).enumerate() {
                let f = buckets[bn][sn];
                //coefs[f].re = 1.0 * sv - (f as f32) / 4000.0;
                //coefs[f].im = (f as f32)*59.452;
                coefs[f] = num_complex::Complex32::from_polar(1.0 * sv - (f as f32) / 4000.0, (f as f32)*59.452);
            }
        }
        fft.process(&mut coefs);
        for i in 0..800 {
            let mut v = coefs[i].re * 0.01;
            if i < 10 { v *= ((i+1) as f32)/10.0}
            if i >= 790  { v *= ((800-i) as f32)/10.0}
            audio_data.push(v);
        }

        audio_data.extend_from_slice(&zeroes);
    }

    let mut audio_file = std::fs::File::create("chirps.wav").unwrap();
    wav::write(wav::Header {
        audio_format: wav::header::WAV_FORMAT_IEEE_FLOAT,
        channel_count: 1,
        sampling_rate: 8000,
        bytes_per_second: 4*8000,
        bytes_per_sample: 4,
        bits_per_sample: 32,
    }, &wav::BitDepth::ThirtyTwoFloat(audio_data), &mut audio_file).unwrap();
}
