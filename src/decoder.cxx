#include "decoder.hxx"
#include <iostream>

bool openStream(Decoder &d, const char *url) {
  d.formatCtx = avformat_alloc_context();
  if (!d.formatCtx) {
    std::cout << "ERROR could not allocate Format Context" << std::endl;
    return false;
  }

  if (avformat_open_input(&d.formatCtx, url, nullptr, nullptr) != 0) {
    std::cout << "ERROR could not open the stream" << std::endl;
    return false;
  }

  if (avformat_find_stream_info(d.formatCtx, nullptr) < 0) {
    std::cout << "ERROR could not get stream info" << std::endl;
    return false;
  }

  const AVCodec *codec = nullptr;
  AVCodecParameters *codecParams = nullptr;
  for (unsigned i = 0; i < d.formatCtx->nb_streams; i++) {
    AVCodecParameters *p = d.formatCtx->streams[i]->codecpar;
    if (p->codec_type == AVMEDIA_TYPE_VIDEO && d.videoStreamIndex == -1) {
      d.videoStreamIndex = (int)i;
      codecParams = p;
      codec = avcodec_find_decoder(p->codec_id);
      std::cout << "Video Codec: " << (codec ? codec->name : "(unknown)")
                << " resolution " << p->width << "x" << p->height << std::endl;
      break;
    }
  }

  if (d.videoStreamIndex == -1 || !codec) {
    std::cout << "Stream does not contain a video stream" << std::endl;
    return false;
  }

  d.codecCtx = avcodec_alloc_context3(codec);
  if (!d.codecCtx ||
      avcodec_parameters_to_context(d.codecCtx, codecParams) < 0 ||
      avcodec_open2(d.codecCtx, codec, nullptr) < 0) {
    std::cout << "ERROR opening codec" << std::endl;
    return false;
  }

  d.frame = av_frame_alloc();
  d.packet = av_packet_alloc();
  return d.frame && d.packet;
}

bool nextFrame(Decoder &d) {
  int r = avcodec_receive_frame(d.codecCtx, d.frame);
  if (r == 0)
    return true;
  if (r != AVERROR(EAGAIN) && r != AVERROR_EOF)
    return false;

  while (av_read_frame(d.formatCtx, d.packet) >= 0) {
    if (d.packet->stream_index == d.videoStreamIndex) {
      int s = avcodec_send_packet(d.codecCtx, d.packet);
      av_packet_unref(d.packet);
      if (s < 0)
        return false;

      r = avcodec_receive_frame(d.codecCtx, d.frame);
      if (r == 0)
        return true;
      if (r != AVERROR(EAGAIN))
        return false;
    } else {
      av_packet_unref(d.packet);
    }
  }

  avcodec_send_packet(d.codecCtx, nullptr);
  return avcodec_receive_frame(d.codecCtx, d.frame) == 0;
}

void closeStream(Decoder &d) {
  if (d.formatCtx)
    avformat_close_input(&d.formatCtx);
  if (d.packet)
    av_packet_free(&d.packet);
  if (d.frame)
    av_frame_free(&d.frame);
  if (d.codecCtx)
    avcodec_free_context(&d.codecCtx);
}
