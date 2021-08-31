#![allow(unused)]

use mkv::elements::parser::Parser as _;
use std::{collections::VecDeque, convert::TryInto, io::Read};

#[derive(gumdrop::Options)]
struct Opts {
    #[options(no_help_flag)]
    help: bool,
}

struct VideoData {
    width: usize,
    heigth: usize,
    track: usize,
    // decoder: bardecoder::Decoder<image::DynamicImage,image::GrayImage>,
    decoder: zbar_rust::ZBarImageScanner,
    minimal_ots : u32,
}

const AUDIO_MINIBLOCK_SIZE : usize = 200*4;
const AUDIO_MINIBLOCK_COUNT: usize = (800*4) / AUDIO_MINIBLOCK_SIZE;

struct AudioData {
    track: usize,

    minimal_ots : u32,
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
    encountered_stamps: std::collections::HashMap<u32, EncountreedStampData>,
    baseline_tc: f64,
}

impl Handler {
    pub fn new() -> Self {
        Self {
            video: None,
            audio: None,
            baseline_tc: f64::INFINITY,
            encountered_stamps: std::collections::HashMap::with_capacity(1024),
        }
    }
}

impl Handler {
    fn process_video(&mut self, mut f: mkv::events::MatroskaFrame) {
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

        //let img  = image::ImageBuffer::<image::Luma<u8>,_>::from_vec(v.width as u32, v.heigth as u32, buf).unwrap();
        //let img = image::DynamicImage::ImageLuma8(img);

        //for qr in v.decoder.decode(&img) {
        for qr in v.decoder.scan_y800(&buf, v.width as u32, v.heigth as u32).unwrap() {
            if qr.data.len() == 4 && qr.data.iter().all(|x| *x >= b'0' && *x <= b'9') {
                let ots : u32 = String::from_utf8_lossy(&qr.data[..]).parse().unwrap();
                let enc_tss = self.encountered_stamps.entry(ots).or_insert_with(Default::default);
                if enc_tss.video_ts.is_none() {
                    println!("{} aV {:.3}", (ots as f32)/10.0, tc_s);
                    enc_tss.video_ts = Some(tc_s);

                    if ots < v.minimal_ots {
                        v.minimal_ots = ots;
                    }

                    if tc_s < self.baseline_tc {
                        self.baseline_tc = tc_s;
                    }

                    //dbg!(v.minimal_ots, self.baseline_tc, ots, tc_s);
                    println!(
                        "{} dV {:.3}",
                        (ots as f32)/10.0,
                        (tc_s - self.baseline_tc) - ((ots - v.minimal_ots) as f64)/10.0,
                    );

                    enc_tss.maybe_print_delta(ots);
                }
            }
        }
    }

    fn process_audio(&mut self, mut f : mkv::events::MatroskaFrame) {
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

            if let Some(mut the_num) = a.analyzer.analyze_block(ts, block) {
                let ots = the_num as u32 * 2;

                let enc_tss = self.encountered_stamps.entry(ots).or_insert_with(Default::default);
                if enc_tss.audio_ts.is_none() {
                    println!("{} aA {:.3}", (ots as f32) / 10.0, ts);
                    enc_tss.audio_ts = Some(ts);
                

                    if ots < a.minimal_ots {
                        a.minimal_ots = ots;
                    }

                    if ts < self.baseline_tc {
                        self.baseline_tc = ts;
                    }

                    //dbg!(a.minimal_ots, self.baseline_tc, ots, tc_s);
                    println!(
                        "{} dA {:.3}",
                        (ots as f32)/10.0,
                        (tc_s - self.baseline_tc) - ((ots - a.minimal_ots) as f64)/10.0,
                    );
                    enc_tss.maybe_print_delta(ots);
                }
            }

            a.miniblocks.pop_front();
        }


        //println!("{} {}",tc_s, buf.len());
    }
}

impl mkv::events::MatroskaEventHandler for Handler {
    fn frame_encountered(&mut self, mut f: mkv::events::MatroskaFrame) {
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
                                    if (w > 1000 || h > 1000) {
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
                                        decoder : zbar_rust::ZBarImageScanner::new(), //bardecoder::default_decoder(),
                                        minimal_ots: u32::MAX,
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
                                minimal_ots: u32::MAX,
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

    let h = Handler::new();
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
    

    Ok(())
}
