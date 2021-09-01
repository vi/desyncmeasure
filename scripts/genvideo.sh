#!/bin/bash

set -e

ffmpeg -v warning -f lavfi -i color=white:320x240:d=0.01 -y white.png

c() { convert white.png a.png -gravity center -composite -brightness-contrast "${1}"x00 "$2"; }

for i in {0..8192..2}; do fn=$(printf '%04d' "$i"); qrencode -s 8 -l H  -o a.png "$(printf "%04d %04d" "$i" "$((8192-i))" )"; c 0 "${fn}"a.png; c 30 "${fn}"b.png;  c 60 "${fn}"c.png;   done

test -f 8190c.png;

perl -E '$c=0; $n=0; for($c=0; $c<819.2*30;) { if ($c >= $n * 3) { $fn=sprintf("%04d", $n); say "file ${fn}a.png"; say "file ${fn}b.png"; say "file ${fn}c.png"; $c+=3; $n += 2; } else {  say "file white.png"; $c+=1; }  }' > filelist.txt

ffmpeg -v warning -r 30 -safe 0 -f concat -i filelist.txt -pix_fmt yuv420p -c:v libx264 -profile baseline -preset veryfast -crf 20 -y video.mkv

# ffmpeg -v warning -i chirps.wav -i video.mkv -c copy -c:a libopus -b:a 50k -f matroska av_sync.mkv
