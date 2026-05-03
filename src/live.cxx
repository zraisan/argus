#include <iostream>

extern "C" {
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

  return 0;
}
