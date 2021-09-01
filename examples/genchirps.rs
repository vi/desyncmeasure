fn main() {
    let buckets = desyncmeasure::chirps::get_buckets();

    let zeroes = [0.0f32; 800];

    let mut audio_data = Vec::with_capacity(800 * 2 * 8192);

    let fft = rustfft::FftPlanner::<f32>::new().plan_fft_forward(800);


    for x in 0..4096 {
        let mut coefs = [num_complex::Complex32::new(0.0, 0.0); 800];
        let data = desyncmeasure::chirps::encode_number(x);
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
            let mut v = coefs[i].re * 0.04;
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
