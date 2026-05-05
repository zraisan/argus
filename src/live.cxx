#include "live.hxx"
#include <cstdint>
#include <cuda_runtime_api.h>
#include <iostream>
#include <memory>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libswscale/swscale.h>
}

using namespace nvinfer1;

void preprocess(const uint8_t *src, float *dst) {
  constexpr int nPixels = 640 * 640;
  for (int k = 0; k < nPixels; k++) {
    dst[k] = src[3 * k] / 255.0f;
    dst[nPixels + k] = src[3 * k + 1] / 255.0f;
    dst[2 * nPixels + k] = src[3 * k + 2] / 255.0f;
  }
}

void runLiveStream(IExecutionContext *trtContext, void *inputBuffer,
                   std::size_t inputBytes, void *outputBuffer,
                   std::size_t outputBytes, cudaStream_t stream) {
  std::unique_ptr<float[]> hostOutput{new float[outputBytes / sizeof(float)]};

  AVFormatContext *pFormatContext = avformat_alloc_context();
  if (!pFormatContext) {
    std::cout << "ERROR could not allocate memory for Format Context"
              << std::endl;
    return;
  }
  if (avformat_open_input(
          &pFormatContext,
          "rtsp://stream.strba.sk:1935/strba/VYHLAD_JAZERO.stream", NULL,
          NULL) != 0) {
    std::cout << "ERROR could not open the file" << std::endl;
    return;
  }
  std::cout << "format " << pFormatContext->iformat->name << ", duration "
            << pFormatContext->duration << " us"
            << ", bit_rate " << pFormatContext->bit_rate << std::endl;
  if (avformat_find_stream_info(pFormatContext, NULL) < 0) {
    std::cout << "ERROR could not get the stream info";
    return;
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
    return;
  }

  AVCodecContext *pCodecContext = avcodec_alloc_context3(pCodec);
  if (!pCodecContext) {
    std::cout << "failed to allocated memory for AVCodecContext" << std::endl;
    return;
  }

  if (avcodec_parameters_to_context(pCodecContext, pCodecParameters) < 0) {
    std::cout << "failed to copy codec params to codec context" << std::endl;
    return;
  }

  if (avcodec_open2(pCodecContext, pCodec, NULL) < 0) {
    std::cout << "failed to open codec through avcodec_open2" << std::endl;
    return;
  }

  AVFrame *pFrame = av_frame_alloc();
  if (!pFrame) {
    std::cout << "failed to allocate memory for AVFrame" << std::endl;
    return;
  }
  AVPacket *pPacket = av_packet_alloc();
  if (!pPacket) {
    std::cout << "failed to allocate memory for AVPacket" << std::endl;
    return;
  }

  while (av_read_frame(pFormatContext, pPacket) >= 0) {
    if (pPacket->stream_index == video_stream_index) {
      std::cout << "AVPacket->pts " << pPacket->pts << std::endl;

      if (avcodec_send_packet(pCodecContext, pPacket) < 0) {
        std::cout << "Error sending packet to decoder" << std::endl;
        break;
      }
      SwsContext *context = sws_getContext(
          pCodecContext->width, pCodecContext->height, pCodecContext->pix_fmt,
          640, 640, AV_PIX_FMT_RGB24, SWS_BILINEAR, NULL, NULL,
          NULL); // Make flags and params an input, same for all
                 // current NULL values

      std::unique_ptr<uint8_t[]> dstBuff{new uint8_t[640 * 640 * 3]};
      uint8_t *dstRGB[1] = {dstBuff.get()};
      int dstStride[1] = {640 * 3};
      std::unique_ptr<float[]> floatBuf{new float[3 * 640 * 640]};
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
        sws_scale(context, pFrame->data, pFrame->linesize, 0, pFrame->height,
                  dstRGB, dstStride);
        preprocess(dstBuff.get(), floatBuf.get());

        cudaMemcpyAsync(inputBuffer, floatBuf.get(), inputBytes,
                        cudaMemcpyHostToDevice, stream);
        trtContext->enqueueV3(stream);
        cudaMemcpyAsync(hostOutput.get(), outputBuffer, outputBytes,
                        cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
      }
      sws_freeContext(context);
    }
    av_packet_unref(pPacket);
  }

  std::cout << "releasing all the resources" << std::endl;

  avformat_close_input(&pFormatContext);
  av_packet_free(&pPacket);
  av_frame_free(&pFrame);
  avcodec_free_context(&pCodecContext);
}
