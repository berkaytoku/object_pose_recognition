#include <opencv2/core/core.hpp>

#include <fstream>  // NOLINT(readability/streams)
#include <iostream>  // NOLINT(readability/streams)
#include <string>
#include <utility>
#include <vector>

#include "caffe/data_layers.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"

namespace caffe {

template <typename Dtype>
TripleImageDataLayer<Dtype>::~TripleImageDataLayer<Dtype>() {
  this->JoinPrefetchThread();
}

template <typename Dtype>
void TripleImageDataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const int new_height = this->layer_param_.image_data_param().new_height();
  const int new_width  = this->layer_param_.image_data_param().new_width();
  const bool is_color  = this->layer_param_.image_data_param().is_color();
  string root_folder = this->layer_param_.image_data_param().root_folder();

  this->output_labels_ = false;
    
  CHECK((new_height == 0 && new_width == 0) ||
      (new_height > 0 && new_width > 0)) << "Current implementation requires "
      "new_height and new_width to be set at the same time.";
  // Read the file with filenames and labels
  const string& source = this->layer_param_.image_data_param().source();
  LOG(INFO) << "Opening file " << source;
  std::ifstream infile(source.c_str());
  string filename_a;
  string filename_b;
  string filename_c;
  while (infile >> filename_a >> filename_b >> filename_c) {
      std::vector<std::string> vec = std::vector<std::string>();
      vec.push_back(filename_a);
      vec.push_back(filename_b);
      vec.push_back(filename_c);
      lines_.push_back(vec);
  }

  if (this->layer_param_.image_data_param().shuffle()) {
    // randomly shuffle data
    LOG(INFO) << "Shuffling data";
    const unsigned int prefetch_rng_seed = caffe_rng_rand();
    prefetch_rng_.reset(new Caffe::RNG(prefetch_rng_seed));
    ShuffleImages();
  }
  LOG(INFO) << "A total of " << lines_.size() << " images.";

  lines_id_ = 0;
  // Check if we would need to randomly skip a few data points
  if (this->layer_param_.image_data_param().rand_skip()) {
    unsigned int skip = caffe_rng_rand() %
        this->layer_param_.image_data_param().rand_skip();
    LOG(INFO) << "Skipping first " << skip << " data points.";
    CHECK_GT(lines_.size(), skip) << "Not enough points to skip";
    lines_id_ = skip;
  }
  // Read an image, and use it to initialize the top blob.
  cv::Mat cv_img = ReadImageToCVMat(root_folder + lines_[lines_id_].at(0),
                                    new_height, new_width, is_color);
  // Use data_transformer to infer the expected blob shape from a cv_image.
  vector<int> top_shape = this->data_transformer_->InferBlobShape(cv_img);
  this->transformed_data_.Reshape(top_shape);
  // Reshape prefetch_data and top[0] according to the batch_size.
  const int batch_size = this->layer_param_.image_data_param().batch_size();
  top_shape[0] = batch_size;
  top_shape[1] = top_shape[1] * 3;
  this->prefetch_data_.Reshape(top_shape);
  top[0]->ReshapeLike(this->prefetch_data_);
  //top[1]->ReshapeLike(this->prefetch_data_);
  //top[2]->ReshapeLike(this->prefetch_data_);

  LOG(INFO) << "output data size: " << top[0]->num() << ","
      << top[0]->channels() << "," << top[0]->height() << ","
      << top[0]->width();
  // label
  //vector<int> label_shape(1, batch_size);
  //this->prefetch_label_.Reshape(label_shape);
}

template <typename Dtype>
void TripleImageDataLayer<Dtype>::ShuffleImages() {
  caffe::rng_t* prefetch_rng =
      static_cast<caffe::rng_t*>(prefetch_rng_->generator());
  shuffle(lines_.begin(), lines_.end(), prefetch_rng);
}

// This function is used to create a thread that prefetches the data.
template <typename Dtype>
void TripleImageDataLayer<Dtype>::InternalThreadEntry() {
  CPUTimer batch_timer;
  batch_timer.Start();
  double read_time = 0;
  double trans_time = 0;
  CPUTimer timer;
  CHECK(this->prefetch_data_.count());
  CHECK(this->transformed_data_.count());
  ImageDataParameter image_data_param = this->layer_param_.image_data_param();
  const int batch_size = image_data_param.batch_size();
  const int new_height = image_data_param.new_height();
  const int new_width = image_data_param.new_width();
  const bool is_color = image_data_param.is_color();
  string root_folder = image_data_param.root_folder();

  // Reshape according to the first image of each batch
  // on single input batches allows for inputs of varying dimension.
  cv::Mat cv_img = ReadImageToCVMat(root_folder + lines_[lines_id_].at(0),
      new_height, new_width, is_color);
  // Use data_transformer to infer the expected blob shape from a cv_img.
  vector<int> top_shape = this->data_transformer_->InferBlobShape(cv_img);
  this->transformed_data_.Reshape(top_shape);
  // Reshape prefetch_data according to the batch_size.
  top_shape[0] = batch_size;
  top_shape[1] = top_shape[1] * 3;
  this->prefetch_data_.Reshape(top_shape);

  Dtype* prefetch_data = this->prefetch_data_.mutable_cpu_data();
  //Dtype* prefetch_label = this->prefetch_label_.mutable_cpu_data();

  // datum scales
  const int lines_size = lines_.size();
  for (int item_id = 0; item_id < batch_size; ++item_id) {
      for (int i=0; i < 3; i++) {
          // get a blob
          timer.Start();
          CHECK_GT(lines_size, lines_id_);
          cv::Mat cv_img = ReadImageToCVMat(root_folder + lines_[lines_id_].at(i),
                                            new_height, new_width, is_color);
          CHECK(cv_img.data) << "Could not load " << lines_[lines_id_].at(i);
          read_time += timer.MicroSeconds();
          timer.Start();
          // Apply transformations (mirror, crop...) to the image
          int offset = this->prefetch_data_.offset(item_id);
          int imgOffset = this->transformed_data_.count();
          this->transformed_data_.set_cpu_data(prefetch_data + offset + imgOffset*i);
          this->data_transformer_->Transform(cv_img, &(this->transformed_data_));
          trans_time += timer.MicroSeconds();
       }   
       
       //prefetch_label[item_id] = (double)lines_[lines_id_].at(3);
       // go to the next iter
       lines_id_++;
       if (lines_id_ >= lines_size) {
          // We have reached the end. Restart from the first.
          DLOG(INFO) << "Restarting data prefetching from start.";
          lines_id_ = 0;
          if (this->layer_param_.image_data_param().shuffle()) {
              ShuffleImages();
          }
       }
  }
  batch_timer.Stop();
  DLOG(INFO) << "Prefetch batch: " << batch_timer.MilliSeconds() << " ms.";
  DLOG(INFO) << "     Read time: " << read_time / 1000 << " ms.";
  DLOG(INFO) << "Transform time: " << trans_time / 1000 << " ms.";
}

INSTANTIATE_CLASS(TripleImageDataLayer);
REGISTER_LAYER_CLASS(TripleImageData);

}  // namespace caffe
