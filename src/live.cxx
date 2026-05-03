#include <iostream>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
}

int main() {
  AVFormatContext *pFormatContext = avformat_alloc_context();
  if (!pFormatContext) {
    std::cout << "ERROR could not allocate memory for Format Context"
              << std::endl;
    return -1;
  }
  if (avformat_open_input(
          &pFormatContext,
          "rtsp://stream.strba.sk:1935/strba/VYHLAD_JAZERO.stream", NULL,
          NULL) != 0) {
    std::cout << "ERROR could not open the file" << std::endl;
    return -1;
  }
  std::cout << "format " << pFormatContext->iformat->name << ", duration "
            << pFormatContext->duration << " us"
            << ", bit_rate " << pFormatContext->bit_rate << std::endl;
  if (avformat_find_stream_info(pFormatContext, NULL) < 0) {
    std::cout << "ERROR could not get the stream info";
    return -1;
  }

  const AVCodec *pCodec = NULL;
  // this component describes the properties of a codec used by the stream i
  // https://ffmpeg.org/doxygen/trunk/structAVCodecParameters.html
  AVCodecParameters *pCodecParameters = NULL;
  int video_stream_index = -1;
  for (int i = 0; i < pFormatContext->nb_streams; i++) {
    AVCodecParameters *pLocalCodecParameters = NULL;
    pLocalCodecParameters = pFormatContext->streams[i]->codecpar;
    std::cout << "AVStream->time_base before open coded "
              << pFormatContext->streams[i]->time_base.num << "/"
              << pFormatContext->streams[i]->time_base.den << std::endl;
    std::cout << "AVStream->r_frame_rate before open coded "
              << pFormatContext->streams[i]->r_frame_rate.num << "/"
              << pFormatContext->streams[i]->r_frame_rate.den << std::endl;
    std::cout << "AVStream->start_time "
              << pFormatContext->streams[i]->start_time << std::endl;
    std::cout << "AVStream->duration " << pFormatContext->streams[i]->duration
              << std::endl;

    std::cout << "finding the proper decoder (CODEC)" << std::endl;

    const AVCodec *pLocalCodec = NULL;

    pLocalCodec = avcodec_find_decoder(pLocalCodecParameters->codec_id);

    if (pLocalCodec == NULL) {
      std::cout << "ERROR unsupported codec!" << std::endl;
      continue;
    }

    if (pLocalCodecParameters->codec_type == AVMEDIA_TYPE_VIDEO) {
      if (video_stream_index == -1) {
        video_stream_index = i;
        pCodec = pLocalCodec;
        pCodecParameters = pLocalCodecParameters;
      }

      std::cout << "Video Codec: resolution " << pLocalCodecParameters->width
                << " x " << pLocalCodecParameters->height << std::endl;
    } else if (pLocalCodecParameters->codec_type == AVMEDIA_TYPE_AUDIO) {
      std::cout << "Audio Codec: "
                << pLocalCodecParameters->ch_layout.nb_channels
                << " channels, sample rate "
                << pLocalCodecParameters->sample_rate << std::endl;
    }

    std::cout << "\tCodec " << pLocalCodec->name << " ID " << pLocalCodec->id
              << " bit_rate " << pLocalCodecParameters->bit_rate << std::endl;
  }
  if (video_stream_index == -1) {
    std::cout << "Stream does not contain a video stream!" << std::endl;
    return -1;
  }

  AVCodecContext *pCodecContext = avcodec_alloc_context3(pCodec);
  if (!pCodecContext) {
    std::cout << "failed to allocated memory for AVCodecContext" << std::endl;
    return -1;
  }

  if (avcodec_parameters_to_context(pCodecContext, pCodecParameters) < 0) {
    std::cout << "failed to copy codec params to codec context" << std::endl;
    return -1;
  }

  if (avcodec_open2(pCodecContext, pCodec, NULL) < 0) {
    std::cout << "failed to open codec through avcodec_open2" << std::endl;
    return -1;
  }

  AVFrame *pFrame = av_frame_alloc();
  if (!pFrame) {
    std::cout << "failed to allocate memory for AVFrame" << std::endl;
    return -1;
  }
  AVPacket *pPacket = av_packet_alloc();
  if (!pPacket) {
    std::cout << "failed to allocate memory for AVPacket" << std::endl;
    return -1;
  }

  while (av_read_frame(pFormatContext, pPacket) >= 0) {
    if (pPacket->stream_index == video_stream_index) {
      std::cout << "AVPacket->pts " << pPacket->pts << std::endl;

      if (avcodec_send_packet(pCodecContext, pPacket) < 0) {
        std::cout << "Error sending packet to decoder" << std::endl;
        break;
      }

      while (true) {
        int r = avcodec_receive_frame(pCodecContext, pFrame);
        if (r == AVERROR(EAGAIN) || r == AVERROR_EOF)
          break;
        if (r < 0) {
          std::cout << "Error receiving frame from decoder" << std::endl;
          break;
        }

        std::cout << "Frame " << pCodecContext->frame_num << " "
                  << pFrame->width << "x" << pFrame->height
                  << " format=" << pFrame->format << " pts=" << pFrame->pts
                  << std::endl;
        // hand pFrame off to swscale → TRT here
      }
    }
    av_packet_unref(pPacket);
  }

  std::cout << "releasing all the resources" << std::endl;

  avformat_close_input(&pFormatContext);
  av_packet_free(&pPacket);
  av_frame_free(&pFrame);
  avcodec_free_context(&pCodecContext);

  return 0;
}
