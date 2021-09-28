use mkv::elements::parser::Parser as _;
use std::{collections::VecDeque, convert::TryInto, io::Read};

#[derive(gumdrop::Options)]
struct Opts {
    #[options(no_help_flag)]
    help: bool,

    #[options(default = "1")]
    threads: usize,
}

struct MessageToVideoDecoder {
    pts: f64,
    buf: Vec<u8>,
    width: u32,
    height: u32,
}

struct VideoData {
    width: usize,
    heigth: usize,
    track: usize,
    // decoder: bardecoder::Decoder<image::DynamicImage,image::GrayImage>,
    // decoder: zbar_rust::ZBarImageScanner,
    decoder_tx: flume::Sender<MessageToVideoDecoder>,
}

const AUDIO_MINIBLOCK_SIZE : usize = 200*4;
const AUDIO_MINIBLOCK_COUNT: usize = (800*4) / AUDIO_MINIBLOCK_SIZE;

struct AudioData {
    track: usize,

    debt: Vec<u8>,
    miniblocks : VecDeque<(f64,Vec<u8>)>,

    analyzer: desyncmeasure::chirps::ChirpAnalyzer,
}

#[derive(Default)]
struct EncountreedStampData {
    video_ts: Option<f64>,
    audio_ts: Option<f64>,
    delta_reported: bool,
}

struct DataCollector {
    video_minimal_ots : u32,
    video_baseline_pts: f64,
    audio_minimal_ots : u32,
    audio_baseline_pts: f64,

    ots_rachet: u32,
    
    encountered_stamps: std::collections::HashMap<u32, EncountreedStampData>,

    video_threads_active: usize,
    audio_finished: bool,

    threads_for_video: usize,
    ots_rachet_spike_charge: f32,
    epoch: u32,
    first_ots_ever: bool,

    audio_timestamps_received: usize,
}

enum MessageToDataCollector {
    VideoTs{pts: f64, ots: u32},
    AudioTs{pts: f64, ots: u32},
    VideoThreadFinished,
    VideoThreadStarted,
    AudioFinished,
}

impl DataCollector {
    fn new(threads_for_video: usize) -> DataCollector {
        DataCollector {
            audio_minimal_ots : u32::MAX,
            audio_baseline_pts: f64::INFINITY,
            video_minimal_ots: u32::MAX,
            video_baseline_pts: f64::INFINITY,
            encountered_stamps: Default::default(),
            video_threads_active: 0,
            audio_finished: false,
            threads_for_video,
            ots_rachet: 0,
            epoch: 0,
            ots_rachet_spike_charge: 0.0,
            first_ots_ever: true,
            audio_timestamps_received: 0,
        }
    }

    fn process_ots_wraparound(&mut self, ots: u32) -> u32 {
        if self.first_ots_ever && ots > 4096 {
            self.ots_rachet = ots - 2048;
        }
        self.first_ots_ever = false;

        let mut ots = ots as i32;
        if ots < self.ots_rachet as i32 {
            ots += 8192;
        }

        if ots > self.ots_rachet as i32 + 4096 {
            ots -= 8192
        }

        if ots > 2048 &&  ots > self.ots_rachet as i32 + 2048  {
            let ots = ots as u32;
            if ots > self.ots_rachet + 2048 + 100 {
                // maybe outliner
                self.ots_rachet_spike_charge += 1.0;

                if self.ots_rachet_spike_charge > 9.0 {
                    self.ots_rachet_spike_charge = 0.0;
                    self.ots_rachet = ots - 2048;
                }
            } else {
                self.ots_rachet = ots - 2048;
                self.ots_rachet_spike_charge *= 0.7;
            }
        } else {
            self.ots_rachet_spike_charge *= 0.4;
        }

        ots += self.epoch as i32 * 8192;

        if self.ots_rachet >= 8192 {
            self.ots_rachet = 0;
            self.epoch += 1;
        }


        ots as u32
    }

    fn start_agent(mut self) -> (flume::Sender<MessageToDataCollector>, std::thread::JoinHandle<()>) {
        let (tx,rx) = flume::unbounded();
        let jh = std::thread::spawn(move || {
            let max_video_msg = if self.threads_for_video == 1 {
                1
            } else {
                2 * self.threads_for_video
            };
            let mut video_msg_sorter = std::collections::BinaryHeap::with_capacity(max_video_msg);
            for msg in rx {
                use MessageToDataCollector::*;
                match msg {
                    VideoThreadStarted => self.video_threads_active += 1,
                    VideoThreadFinished => self.video_threads_active -= 1,
                    AudioFinished => self.audio_finished = true,
                    VideoTs { pts, ots } => {
                        video_msg_sorter.push((std::cmp::Reverse(ordered_float::OrderedFloat(pts)),ots));
                    }
                    AudioTs { pts, ots } => {
                        //eprintln!("pre-wraparound audio ots: {}", ots);
                        let ots = self.process_ots_wraparound(ots);
                        let enc_tss = self.encountered_stamps.entry(ots).or_insert_with(Default::default);
                        if enc_tss.audio_ts.is_none() {
                            println!("{} aA {:.3}", (ots as f32) / 10.0, pts);
                            enc_tss.audio_ts = Some(pts);
                            self.audio_timestamps_received += 1;
                        

                            if ots < self.audio_minimal_ots && self.audio_timestamps_received < 10 {
                                self.audio_minimal_ots = ots;
                            }

                            if pts < self.audio_baseline_pts {
                                self.audio_baseline_pts = pts;
                            }

                            //dbg!(a.minimal_ots, self.baseline_tc, ots, tc_s);
                            println!(
                                "{} dA {:.3}",
                                (ots as f32)/10.0,
                                (pts - self.audio_baseline_pts) - ((ots - self.audio_minimal_ots) as f64)/10.0,
                            );
                            enc_tss.maybe_print_delta(ots);
                        }
                    }
                }
                let exiting = self.video_threads_active == 0 && self.audio_finished;

                while video_msg_sorter.len() >= max_video_msg || (video_msg_sorter.len() > 0 && exiting) {
                    let (std::cmp::Reverse(ordered_float::OrderedFloat( pts)), ots) = video_msg_sorter.pop().unwrap();
                    //eprintln!("pre-wraparound video ots: {}", ots);
                    let ots = self.process_ots_wraparound(ots);
                    let enc_tss = self.encountered_stamps.entry(ots).or_insert_with(Default::default);
                    if enc_tss.video_ts.is_none() {
                        println!("{} aV {:.3}", (ots as f32)/10.0, pts);
                        enc_tss.video_ts = Some(pts);
    
                        if ots < self.video_minimal_ots {
                            self.video_minimal_ots = ots;
                        }
    
                        if pts < self.video_baseline_pts {
                            self.video_baseline_pts = pts;
                        }
    
                        //dbg!(v.minimal_ots, self.baseline_tc, ots, tc_s);
                        println!(
                            "{} dV {:.3}",
                            (ots as f32)/10.0,
                            (pts - self.video_baseline_pts) - ((ots - self.video_minimal_ots) as f64)/10.0,
                        );
    
                        enc_tss.maybe_print_delta(ots);
                    }
                }

                if exiting {
                    break;
                }
            }
        });
        (tx, jh)
    }
}


impl EncountreedStampData {
    fn maybe_print_delta(&mut self, ots: u32) {
        if ! self.delta_reported && self.video_ts.is_some() && self.audio_ts.is_some() {
            self.delta_reported = true;
            println!("{} De {:.3}", ots as f32 / 10.0, self.audio_ts.unwrap() - self.video_ts.unwrap());
        }
    }
}

struct Handler {
    video: Option<VideoData>,
    audio: Option<AudioData>,
    data_collector: flume::Sender<MessageToDataCollector>,
    threads_for_video: usize,
}

impl Handler {
    pub fn new(dc: flume::Sender<MessageToDataCollector>, threads_for_video: usize) -> Self {
        Self {
            video: None,
            audio: None,
            data_collector: dc,
            threads_for_video,
        }
    }
}

fn video_decoders(num_threads: usize, data_collector: flume::Sender<MessageToDataCollector>) -> flume::Sender<MessageToVideoDecoder> {
    let (tx,rx) = flume::bounded(num_threads);

    for _ in 0..num_threads {
        let rx = rx.clone();
        let data_collector = data_collector.clone();
        std::thread::spawn(move || {
            #[cfg(feature = "zbar-rust")]
            let mut decoder = zbar_rust::ZBarImageScanner::new();
            let decoded = std::cell::Cell::new(false);

            data_collector.send(MessageToDataCollector::VideoThreadStarted).unwrap();
            'msgloop: for msg in rx {
                let msg : MessageToVideoDecoder = msg;
                if msg.buf.len() == 0 { break }

                let pts = msg.pts;

                let decoded_code_handler = |qr:&[u8]| {
                    if qr.len() != 4+4+1 { return; }
                    if ! qr[0..4].iter().all(|x| *x >= b'0' && *x <= b'9') { return; }
                    if qr[4] != b' ' { return; }
                    if ! qr[5..9].iter().all(|x| *x >= b'0' && *x <= b'9') { return; }

                    let ots : u32 = String::from_utf8_lossy(&qr[0..4]).parse().unwrap();
                    let ots2 : u32 = String::from_utf8_lossy(&qr[5..9]).parse().unwrap();

                    if ots+ots2 != 8192 { return; }
                    data_collector.send(MessageToDataCollector::VideoTs{pts: pts, ots}).unwrap();
                    decoded.set(true);
                };

                #[cfg(feature = "rqrr")] {
                    let mut pi = rqrr::PreparedImage::prepare_from_greyscale(msg.width as usize, msg.height as usize, |x,y| {
                        msg.buf[y*(msg.width as usize) + x]
                    });
                    for grid in pi.detect_grids() {
                        if let Ok((_, qr)) = grid.decode() {
                            let qr = qr.as_bytes();
                            decoded_code_handler(qr);
                            if decoded { continue 'msgloop; }
                        }
                    }
                }

                #[cfg(feature = "zbar-rust")] {
                    for qr in decoder.scan_y800(&msg.buf, msg.width, msg.height).unwrap() {
                        decoded_code_handler(&qr.data);
                        if decoded.get() { continue 'msgloop; }
                    }
                    let ib = image::ImageBuffer::<image::Luma<u8>,_>::from_vec(msg.width, msg.height, msg.buf).unwrap();
                    // Now also try downscaled variants
                    let halfscaled  = image::imageops::resize(
                        &ib,
                        msg.width / 2,
                        msg.height / 2,
                        image::imageops::FilterType::Triangle,
                    );
                    for layer in (4..=10).rev() {
                        let scaled_buf;
                        let scaled = match layer {
                            4 => &halfscaled,
                            3 | 5 => {
                                scaled_buf = image::imageops::resize(
                                    &ib,
                                    msg.width * 2 / layer,
                                    msg.height * 2 / layer,
                                    image::imageops::FilterType::Triangle,
                                );
                                &scaled_buf
                            },
                            _ => {
                                scaled_buf = image::imageops::resize(
                                    &halfscaled,
                                    msg.width  * 2 / layer,
                                    msg.height * 2 / layer,
                                    image::imageops::FilterType::Triangle,
                                );
                                &scaled_buf
                            },
                        };
                        assert_eq!(scaled.as_flat_samples().strides_cwh(), (1,1,scaled.width() as usize));
                        for qr in decoder.scan_y800(scaled.as_flat_samples().as_slice(), scaled.width(), scaled.height()).unwrap() {
                            decoded_code_handler(&qr.data);
                            if decoded.get() { continue 'msgloop; }
                        }
                    }
                }
            }
            data_collector.send(MessageToDataCollector::VideoThreadFinished).unwrap();
        });
    }

    tx
}

impl Handler {
    fn process_video(&mut self, f: mkv::events::MatroskaFrame) {
        let v = self.video.as_mut().unwrap();
        if f.buffers.len() != 1 {
            log::error!("Unexpected number of laced frames. Should be sole unlaced buffer.");
            return;
        }
        let buf = f.buffers.into_iter().next().unwrap();
        if buf.len() != v.width * v.heigth {
            log::error!("Unexpected byte size of video frame: should be exactly width * height");
            return;
        }
        let tc_s = f.timecode_nanoseconds as f64 / 1000_000_000.0;

        v.decoder_tx.send(MessageToVideoDecoder{
            pts: tc_s,
            buf,
            width: v.width as u32,
            height: v.heigth as u32,
        }).unwrap();

        //let img  = image::ImageBuffer::<image::Luma<u8>,_>::from_vec(v.width as u32, v.heigth as u32, buf).unwrap();
        //let img = image::DynamicImage::ImageLuma8(img);

        //for qr in v.decoder.decode(&img) {
    }

    fn process_audio(&mut self, f : mkv::events::MatroskaFrame) {
        let a = self.audio.as_mut().unwrap();

        if f.buffers.len() != 1 {
            log::error!("Unexpected number of laced frames. Should be sole unlaced buffer.");
            return;
        }
        let buf = f.buffers.into_iter().next().unwrap();
        let mut bufview = &buf[..];

        let mut tc_s = f.timecode_nanoseconds as f64 / 1000_000_000.0;
        //println!("{} len={} debt={}", tc_s, buf.len(), a.debt.len());

        loop {
            if bufview.len() + a.debt.len() >= AUDIO_MINIBLOCK_SIZE {
                let mut v = Vec::with_capacity(AUDIO_MINIBLOCK_SIZE);
                v.extend_from_slice(&a.debt);
                let takefrombufview = AUDIO_MINIBLOCK_SIZE-a.debt.len();
                v.extend_from_slice(&bufview[0..takefrombufview]);
                tc_s -= a.debt.len() as f64 / 8000.0 / 4.0;

                a.miniblocks.push_back((tc_s, v));

                tc_s += AUDIO_MINIBLOCK_SIZE as f64 / 8000.0 / 4.0;
                a.debt.clear();
                bufview = &bufview[takefrombufview..];
            } else {
                if bufview.is_empty() {
                    break;
                }
                a.debt.extend_from_slice(bufview);
                bufview = &bufview[0..0];
            }
        }

        while a.miniblocks.len() >= AUDIO_MINIBLOCK_COUNT {
            let ts = a.miniblocks[0].0;
            let mut block = Vec::with_capacity(800);
            for (_,miniblock) in a.miniblocks.iter().take(AUDIO_MINIBLOCK_COUNT) {
                for sample in itertools::Itertools::chunks(miniblock.iter(), 4).into_iter() {
                    let mut sample_bytes = Vec::with_capacity(4);
                    sample_bytes.extend(sample);
                    let sample = f32::from_le_bytes(sample_bytes.try_into().unwrap());
                    block.push(num_complex::Complex32::new(sample, 0.0));
                }
            }
            assert_eq!(block.len(), 800);

            if let Some(the_num) = a.analyzer.analyze_block(ts, block) {
                let ots = the_num as u32 * 2;

                self.data_collector.send(MessageToDataCollector::AudioTs{pts: ts, ots}).unwrap();
            }

            a.miniblocks.pop_front();
        }


        //println!("{} {}",tc_s, buf.len());
    }
}

impl mkv::events::MatroskaEventHandler for Handler {
    fn frame_encountered(&mut self,  f: mkv::events::MatroskaFrame) {
        if self.video.is_none() && self.audio.is_none() {
            log::error!("No suitable audio or video tracks found");
        }
        if let Some(ref mut v) = self.video {
            if v.track == f.track_number {
                self.process_video(f);
                return;
            }
        }
        if let Some(ref a) = self.audio {
            if a.track == f.track_number {
                self.process_audio(f);
                return;
            }
        }
    }

    fn segment_tracks(&mut self, e: &std::rc::Rc<mkv::elements::Element>) {

        use mkv::elements::database::Class;
        use mkv::elements::ElementContent;

        match &e.content {
            mkv::elements::ElementContent::Master(tracks) => {
                for track in tracks {
                    if track.class != Class::TrackEntry {
                        continue;
                    }

                    let mut tn = None;
                    let mut ci = None;
                    let mut tt = None;

                    let mut width = None;
                    let mut height = None;
                    let mut colsp = None;

                    let mut channels = None;
                    let mut samplerate = None;
                    let mut samplebits = None;

                    match track.content {
                        ElementContent::Master(ref v) => {

                            for x in v {
                                match x.class {
                                    Class::TrackNumber => match x.content {
                                        ElementContent::Unsigned(c) => tn = Some(c),
                                        _ => log::error!("Internal error 2"),
                                    }
                                    Class::CodecID => match &x.content {
                                        ElementContent::Text(c) => ci = Some(c),
                                        _ => log::error!("Internal error 2"),
                                    }
                                    Class::TrackType => match x.content {
                                        ElementContent::Unsigned(c) => tt = Some(c),
                                        _ => log::error!("Internal error 2"),
                                    }
                                    Class::Video => match x.content {
                                        ElementContent::Master(ref vv) => {
                                            for xx in vv {
                                                match xx.class {
                                                    Class::PixelWidth => match xx.content {
                                                        ElementContent::Unsigned(c) => width = Some(c),
                                                        _ => log::error!("Internal error 2"),
                                                    }
                                                    Class::PixelHeight => match xx.content {
                                                        ElementContent::Unsigned(c) => height = Some(c),
                                                        _ => log::error!("Internal error 2"),
                                                    }
                                                    Class::ColourSpace => match &xx.content {
                                                        ElementContent::Binary(c) => colsp = Some(c),
                                                        _ => log::error!("Internal error 2"),
                                                    }
                                                    _ => (),
                                                }
                                            }
                                        }
                                        _ => log::error!("Internal error 2"),
                                    }
                                    Class::Audio => match x.content {
                                        ElementContent::Master(ref vv) => {
                                            for xx in vv {
                                                match xx.class {
                                                    Class::SamplingFrequency => match xx.content {
                                                        ElementContent::Float(c) => samplerate = Some(c),
                                                        _ => log::error!("Internal error 2"),
                                                    }
                                                    Class::Channels => match xx.content {
                                                        ElementContent::Unsigned(c) => channels = Some(c),
                                                        _ => log::error!("Internal error 2"),
                                                    }
                                                    Class::BitDepth => match xx.content {
                                                        ElementContent::Unsigned(c) => samplebits = Some(c),
                                                        _ => log::error!("Internal error 2"),
                                                    }
                                                    _ => (),
                                                }
                                            }
                                        }
                                        _ => log::error!("Internal error 2"),
                                    }
                                    _ => (),
                                }
                            }
                        }
                        _ => log::error!("Internal error 3"),
                    }

                    if tn.is_none() {
                        log::error!("Malformed matroska file: no TrackNumber in TrackEntry");
                        continue;
                    }
                    let tn = tn.unwrap();

                    match tt {
                        None => log::error!("Malformed matroska file: no TrackType in a TrackEntry"),
                        Some(1) => {
                            match ci {
                                None => log::error!("No CodecID in a video track"),
                                Some(c) if **c == "V_UNCOMPRESSED" => {
                                    // OK
                                }
                                Some(_) => {
                                    log::error!("Only video tracks of type V_UNCOMPRESSED are supported");
                                    continue;
                                }
                            }
                            match colsp {
                                None => log::error!("Missing ColourSpace information in video track header"),
                                Some(c) if **c == [89, 56, 48, 48,] => {
                                    // OK
                                }
                                Some(_) => {
                                    log::error!("Only colour space Y800 (grayscale) is supported");
                                    continue;
                                }
                            }
                            match (width, height) {
                                (Some(w), Some(h)) => {
                                    log::info!("Video track {}x{} at track number {}", w,h,tn);
                                    if w > 1000 || h > 1000 {
                                        log::warn!("Too large video. Consider downscaling it");
                                    }
                                    if self.video.is_some() {
                                        log::error!("Multiple suitable video tracks found");
                                        continue;
                                    }
                                    self.video = Some(VideoData {
                                        track: tn as usize,
                                        width: w as usize,
                                        heigth: h as usize,
                                        //decoder : zbar_rust::ZBarImageScanner::new(), //bardecoder::default_decoder(),
                                        decoder_tx: video_decoders(self.threads_for_video, self.data_collector.clone()),
                                    });
                                }
                                _ => {
                                    log::error!("PixelWidth or PixelHeight is absend from video track header");
                                    continue;
                                }
                            }
                        }
                        Some(2) => {
                            match ci {
                                None => log::error!("No CodecID in audio track"),
                                Some(c) if **c == "A_PCM/FLOAT/IEEE" => {
                                    // OK
                                }
                                Some(_) => {
                                    log::error!("Only audio tracks of type A_PCM/FLOAT/IEEE, one channel, 32 bits, little endian are supported");
                                    continue;
                                }
                            }
                            match channels {
                                None => log::error!("No Channels info in audio track"),
                                Some(c) if c == 1 => {
                                    // OK
                                }
                                Some(_) => {
                                    log::error!("Audio should be mono");
                                    continue;
                                }
                            }
                            match samplerate {
                                None => log::error!("No Channels info in audio track"),
                                Some(c) if c > 7999.0 && c < 8001.0 => {
                                    // OK
                                }
                                Some(_) => {
                                    log::error!("Audio sample rate should be 8000 Hz.");
                                    continue;
                                }
                            }
                            match samplebits {
                                None => log::error!("No BitDepth info in audio track"),
                                Some(32) => (),
                                Some(_) => {
                                    log::error!("Audio sample size shuld be 32 bits");
                                    continue;
                                }
                            }

                            if self.audio.is_some() {
                                log::error!("Multiple audio video tracks found");
                                continue;
                            }

                            self.audio = Some(AudioData {
                                track: tn as usize,
                                debt: Vec::with_capacity(800),
                                miniblocks: VecDeque::with_capacity(AUDIO_MINIBLOCK_COUNT),
                                analyzer: desyncmeasure::chirps::ChirpAnalyzer::new(),
                            });

                        }
                        _ => log::error!("Non-video track encountered"),
                    }
                }
            }
            _ => log::error!("Internal error 1"),
        }

        //println!("{:#?}", *e);
    }
}


/* 
Element { class: Tracks, content: Master([
    Element { class: TrackEntry, content: Master([
        Element { class: TrackNumber, content: Unsigned( 1, ), },
        Element {  class: CodecID, content: Text( "V_UNCOMPRESSED", ), },
        Element {  class: TrackType, content: Unsigned( 1, ), },
        Element {  class: Video, content: Master([
            Element { class: PixelWidth, content: Unsigned( 320, ), },
            Element { class: PixelHeight, content: Unsigned( 240, ), },
            Element { class: ColourSpace, content: Binary( [ 89, 56, 48, 48, ], ), })] })]
    Element { class: TrackEntry, content: Master([
        Element { class: TrackNumber, content: Unsigned(1) },
        Element { class: CodecID, content: Text("A_PCM/FLOAT/IEEE") },
        Element { class: TrackType, content: Unsigned(2) },
        Element { class: Audio, content: Master([
            Element { class: Channels, content: Unsigned(1) },
            Element { class: SamplingFrequency, content: Float(8000.0) },
            Element { class: BitDepth, content: Unsigned(32) }]) }])
*/


fn main() -> anyhow::Result<()> {
    let opts : Opts = gumdrop::Options::parse_args_default_or_exit();
    if opts.help {
        println!("Analyse video for audio-video desynchronisation.");
        println!("This tool can only analyse one specific video (or its parts):");
        println!("    https://vi-server.org/pub/av_sync.mkv");
        println!("Recommended command line:");
        println!("    ffmpeg -v warning -i input_video.mp4 -pix_fmt gray -c:v rawvideo  -c:a pcm_f32le -ar 8000 -ac 1 -f matroska - | desyncmeasure");
        println!("Legend:");
        println!("  a{{V,A}} - receive timestamp against send {{video,audio}} timestamp. ");
        println!("  d{{V,A}} - Relative {{video,audio}} delay. ");
        println!("  De - A/V desyncronisation again send timestamp");
        println!("{}", <Opts as gumdrop::Options>::usage());
        return Ok(());
    }
    env_logger::init();
    let si = std::io::stdin();
    let mut si = si.lock();
    //let si = std::io::BufReader::with_capacity(80_000, si);

    let dc = DataCollector::new(opts.threads);
    let (dctx, dch) = dc.start_agent();
    let dctx2 = dctx.clone();

    {
        let h = Handler::new(dctx, opts.threads);
        let hl_p = mkv::events::MatroskaDemuxer::new(h);
        let mut ml_p = mkv::elements::midlevel::MidlevelParser::new(hl_p);
        let mut ll_p = mkv::elements::parser::new();
        
        let mut buf = vec![0; 79872];
        loop {
            let len = si.read(&mut buf[..])?;
            if len == 0 { break }
            let buf = &buf[0..len];
    
            ll_p.feed_bytes(buf, &mut ml_p);
        }
    }
    dctx2.send(MessageToDataCollector::AudioFinished).unwrap();
    
    dch.join().unwrap();

    Ok(())
}
