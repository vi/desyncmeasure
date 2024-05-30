# desyncmeasure
A tool to measure audio/video synchronosation errors using special source video

# Steps

1. Play [av_sync.mkv](https://vi-server.org/pub/av_sync.mkv) to one device's camera and microphone.
2. Record video and audio from the other device into a file
3. Feed that video into this tool (using an FFmpeg command line in it's CLI help message)
4. Judge quality of audio and video by the number of recognized QR codes, audio chirps and discrepancy between timestamps encoded in the codes and chirps.

TODO: write a proper README.
