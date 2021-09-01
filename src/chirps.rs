use std::sync::Arc;

pub fn get_buckets() -> [[usize; 4]; 7] {
    let mut ret = [[0; 4]; 7];
    let mut freq = 830;
    for b in &mut ret {
        for x in b {
            *x = freq / 10;
            freq += 87;
        }
    }
    //dbg!(freq);
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

pub fn number_indexes(mut x: usize) -> [usize; 7] {
    assert!(x < 4096);
    let mut idxs = [0; 7];
    let mut parity = 0;
    for i in 0..=5 {
        idxs[i] = x & 0b11;
        parity ^= x & 0b11;
        x >>= 2;
    }
    idxs[6] = parity;
    idxs
}

pub fn encode_number(x: usize) -> [[f32; 4]; 7] {
    assert!(x < 4096);
    let idxs = number_indexes(x);
    let mut cfs = [[0.0f32; 4]; 7];
    for i in 0..=6 {
        cfs[i] = encode_bucket(idxs[i]);
    }
    cfs
}


pub struct ChirpAnalyzer {
    fft: Arc<dyn rustfft::Fft<f32>>,
    buckets : [[usize; 4]; 7],
    saved_num: usize,
    save_num_ctr: usize,
}

impl ChirpAnalyzer {
    pub fn new() -> ChirpAnalyzer {
        ChirpAnalyzer {
            fft: rustfft::FftPlanner::new().plan_fft_inverse(800),
            buckets : get_buckets(),
            saved_num: usize::MAX,
            save_num_ctr: 0,
        }
    }

    pub fn analyze_block(&mut self, _ts: f64, mut block: Vec<num_complex::Complex32>) -> Option<usize> {
        assert_eq!(block.len(), 800);

        self.fft.process(&mut block);

        let mut signal_quality = 100.0;

        let mut indexes = Vec::with_capacity(7);

        for bucket in self.buckets {
            let mut magnitutes = Vec::with_capacity(4);
            for string in bucket {
                let mut x = 0.0;
                x += 0.3*block[string].norm();
                x += 0.7*block[string].norm();
                x += 1.0*block[string].norm();
                x += 0.7*block[string].norm();
                x += 0.3*block[string].norm();
                magnitutes.push(x);
            }

            let mut sum : f32 = magnitutes.iter().sum();
            let best_idx = magnitutes.iter().enumerate().max_by_key(|(_,v)|ordered_float::OrderedFloat(**v)).unwrap().0;
            let mut losers_magnitutes = 0.0;

            for (n,x) in magnitutes.iter().enumerate() {
                if n == best_idx { continue }
                losers_magnitutes += *x;
            }
            sum += 0.0001;

            let mut qual = (magnitutes[best_idx] - losers_magnitutes)/sum;
            if qual < 0.05 { qual = 0.05; }
            signal_quality *= qual;

            indexes.push(best_idx);
        }

        let mut the_num = 0;
        let mut multiplier = 1;
        for i in 0..=5 {
            the_num += multiplier * indexes[i];
            multiplier *= 4;
        }
        //print!("QQQ {} {:?}  q {}", _ts, indexes, signal_quality);
        let canonical_indexes_for_this_number = number_indexes(the_num);

        //print!(" p {} the_num {}", parity, the_num);
        if indexes[6] != canonical_indexes_for_this_number[6] {
            signal_quality = 0.0;
        }
        //println!(" q {}", signal_quality);


        //println!("{:.3} AQ {}", ts, signal_quality);
        if signal_quality > 0.01 {
            //println!("{:.3} N {}", ts, the_num);
            if the_num == self.saved_num {
                self.save_num_ctr += 1;
                if self.save_num_ctr >= 3 {
                    Some(the_num)
                } else {
                    None
                }
            } else {
                self.saved_num = the_num;
                self.save_num_ctr = 1;
                None
            }
        } else {
            self.save_num_ctr = 0;
            None
        }
    }
}
