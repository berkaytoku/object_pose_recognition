#include <algorithm>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void TriplePairEuclideanLossLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const int count = bottom[0]->count();
  vector<double> temp0, temp, pair;
  float m = 0.01;
  caffe_gpu_sub(
      count,
      bottom[0]->gpu_data(),  // a
      bottom[1]->gpu_data(),  // b
      xixj_diff_.mutable_gpu_data());  // a_i-b_i
  caffe_gpu_powx(
      count,
      xixj_diff_.mutable_gpu_data(),  // a_i-b_i
      Dtype(2),
      diff_sq_.mutable_gpu_data());  // (a_i-b_i)^2
  caffe_gpu_gemv(
      CblasNoTrans,
      bottom[0]->num(),
      bottom[0]->channels(),
      Dtype(1.0),
      diff_sq_.gpu_data(),  // (a_i-b_i)^2
      summer_vec_.gpu_data(),
      Dtype(0.0),
      xixj_dist_sq_.mutable_gpu_data());  // \Sum (a_i-b_i)^2
  Dtype margin = this->layer_param_.contrastive_loss_param().margin();
  bool legacy_version =
      this->layer_param_.contrastive_loss_param().legacy_version();
  Dtype loss(0.0);
  
  double tempDenominator = 0.0;
  
  for (int i = 0; i < bottom[0]->num(); ++i) {
      tempDenominator += diff_sq_.cpu_data()[i];
  }

  tempDenominator = sqrt(tempDenominator) + m;

  caffe_gpu_sub(
      count,
      bottom[0]->gpu_data(),  // a
      bottom[2]->gpu_data(),  // b
      xixk_diff_.mutable_gpu_data());  // a_i-b_i

  caffe_gpu_powx(
      count,
      xixk_diff_.mutable_gpu_data(),  // a_i-b_i
      Dtype(2),
      diff_sq_.mutable_gpu_data());
  caffe_gpu_gemv(
      CblasNoTrans,
      bottom[0]->num(),
      bottom[0]->channels(),
      Dtype(1.0),
      diff_sq_.gpu_data(),  // (a_i-b_i)^2
      summer_vec_.gpu_data(),
      Dtype(0.0),
      xixk_dist_sq_.mutable_gpu_data());  // \Sum (a_i-b_i)^2
  
  double tempNumerator = 0.0;
  for(int j=0; j<bottom[0]->num(); j++) {
	tempNumerator += diff_sq_.mutable_cpu_data()[j];  	
  }

  tempNumerator = sqrt(tempNumerator);
  Dtype dist = std::max(1-(tempNumerator/tempDenominator), 0.0);
  loss += dist;
  
  caffe_gpu_sub(
      count,
      bottom[3]->gpu_data(),  // a
      bottom[4]->gpu_data(),  // b
      xixj_p_diff_.mutable_gpu_data());  // a_i-b_i
  caffe_gpu_powx(
      count,
      diff_.mutable_gpu_data(),  // a_i-b_i
      Dtype(2),
      diff_sq_.mutable_gpu_data());
  caffe_gpu_gemv(
      CblasNoTrans,
      bottom[0]->num(),
      bottom[0]->channels(),
      Dtype(1.0),
      diff_sq_.gpu_data(),  // (a_i-b_i)^2
      summer_vec_.gpu_data(),
      Dtype(0.0),
      xixj_p_dist_sq_.mutable_gpu_data());  // \Sum (a_i-b_i)^2
  double denomForPair = 0.0;
  for(int k=0; k<bottom[0]->num(); k++) {
	denomForPair += diff_sq_.mutable_cpu_data()[k];  	
  }
  loss += denomForPair;
  
  //loss = loss / static_cast<Dtype>(bottom[0]->num()) / Dtype(2);
  top[0]->mutable_cpu_data()[0] = loss;
}

template <typename Dtype>
__global__ void CLLBackward(const int count, const int channels, int bottom_index, const Dtype *x,
    Dtype *bottom_diff, const Dtype *xixj_dist_sq_, const Dtype *xixk_dist_sq_) {
  Dtype bottom_diff_val(0.0);
  CUDA_KERNEL_LOOP(i, count) {
    //int n = i / channels;  // the num index, to access y and dist_sq
	 
		  if(bottom_index < 3){ //triple
				//gradient of loss equation
				if(bottom_index == 0){ //dLoss/dxi
					bottom_diff_val = sqrt(xixk_dist_sq_[i])/ sqrt(xixj_dist_sq_[i]);
				}
				else if (bottom_index == 1){ //dLoss/dxj
					bottom_diff_val = -(x[i] / sqrt(xixj_dist_sq_[i]));
				}			
				else if (bottom_index == 2){ //dLoss/dxk
					bottom_diff_val = -(sqrt(xixk_dist_sq_[i]) / x[i]);
				}
		  }
		  else if(bottom_index >= 3){ //pair
			  //gradient of loss equation
			  if (bottom_index == 3){ //dLoss/dxi_p
				  bottom_diff_val = 2 * x[i];
			  }
			  else if (bottom_index == 4){ //dLoss/dxj_p
				  bottom_diff_val = -(2 * x[i]);
			  }  
		  }
		  if (bottom_diff_val > 0.0){
			bottom_diff[i] = bottom_diff_val;
		  }
		  else{
			bottom_diff[i] = 0;
		  }
				
  }
}

template <typename Dtype>
void TriplePairEuclideanLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
	for(int i = 0; i < 5; ++i){
	  Dtype* bout = bottom[i]->mutable_cpu_diff();
	  //int num = bottom[i]->num();
	  int count = bottom[i]->count();
	  int channels = bottom[i]->channels();
	   if (propagate_down[i]) {
		  CLLBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
		  count, channels, i, bottom[i]->gpu_data(), bottom[i]->mutable_gpu_diff(), xixj_dist_sq_.gpu_data(), xixk_dist_sq_.gpu_data());
		  CUDA_POST_KERNEL_CHECK;
	  }
    }
}

INSTANTIATE_LAYER_GPU_FUNCS(TriplePairEuclideanLossLayer);

}  // namespace caffe
