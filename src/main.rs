#![allow(unused)]

use mkv::elements::parser::Parser as _;
use std::io::Read;

struct VideoData {
    width: usize,
    heigth: usize,
    track: usize,
    // decoder: bardecoder::Decoder<image::DynamicImage,image::GrayImage>,
    decoder: zbar_rust::ZBarImageScanner,
    encountered_stamps: std::collections::HashSet<u32>,
    minimal_otc : u32,

}

struct Handler {
    video: Option<VideoData>,
    baseline_tc: f64,
}

impl Handler {
    pub fn new() -> Self {
        Self {
            video: None,
            baseline_tc: f64::INFINITY,
        }
    }
}

impl mkv::events::MatroskaEventHandler for Handler {
    fn frame_encountered(&mut self, mut f: mkv::events::MatroskaFrame) {
        if let Some(ref mut v) = self.video {
            if v.track == f.track_number {
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

                if tc_s < self.baseline_tc {
                    self.baseline_tc = tc_s;
                }

                //let img  = image::ImageBuffer::<image::Luma<u8>,_>::from_vec(v.width as u32, v.heigth as u32, buf).unwrap();
                //let img = image::DynamicImage::ImageLuma8(img);

                //for qr in v.decoder.decode(&img) {
                for qr in v.decoder.scan_y800(&buf, v.width as u32, v.heigth as u32).unwrap() {
                    if qr.data.iter().all(|x| *x >= b'0' && *x <= b'9') {
                        let ots : u32 = String::from_utf8_lossy(&qr.data[..]).parse().unwrap();
                        if  ! v.encountered_stamps.contains(&ots) {
                            println!("{} V {}", (ots as f32)/10.0, tc_s);
                            v.encountered_stamps.insert(ots);

                            if ots < v.minimal_otc {
                                v.minimal_otc = ots;
                            }

                            //dbg!(v.minimal_otc, self.baseline_tc, ots, tc_s);
                            println!(
                                "{} dV {}",
                                (ots as f32)/10.0,
                                (tc_s - self.baseline_tc) - ((ots - v.minimal_otc) as f64)/10.0,
                            );
                        }
                    }
                }

                return;
            }
        } else {
            log::error!("No video metadata available");
        }
        //println!("{}", f.timecode_nanoseconds);
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
                                        encountered_stamps: std::collections::HashSet::with_capacity(1024),
                                        minimal_otc: u32::MAX,
                                    });
                                }
                                _ => {
                                    log::error!("PixelWidth or PixelHeight is absend from video track header");
                                    continue;
                                }
                            }
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
            Element { class: ColourSpace, content: Binary( [ 89, 56, 48, 48, ], ), },
*/


fn main() -> anyhow::Result<()> {
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
