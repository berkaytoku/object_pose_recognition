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

  /*
  for (int j=0; j < 10; j++) {
      LOG(INFO) << *(bottom[0]->cpu_data() + j);
  }
  */

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

    caffe_gpu_sub(
        count,
        bottom[3]->gpu_data(),  // a
        bottom[4]->gpu_data(),  // b
        xixj_p_diff_.mutable_gpu_data());  // a_i-b_i
    caffe_gpu_powx(
        count,
        xixj_p_diff_.mutable_gpu_data(),  // a_i-b_i
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

  Dtype loss(0.0);
  for (int i = 0; i < bottom[0]->num(); ++i) {
        loss += std::max(Dtype(0.0), static_cast<Dtype>(1-(sqrt(xixk_dist_sq_.cpu_data()[i])/(sqrt(xixj_dist_sq_.cpu_data()[i]) + Dtype(1e-2)))));
        loss += xixj_p_dist_sq_.cpu_data()[i];
  }


    top[0]->mutable_cpu_data()[0] = loss / static_cast<Dtype>(bottom[0]->num());
}

template <typename Dtype>
__global__ void CLLBackward(const int count, const int channels, int bottom_index,
    Dtype *bottom_diff, const Dtype *xixj_diff_, const Dtype *xixk_diff_, const Dtype *xixj_p_diff_, const Dtype *xixj_dist_sq_, const Dtype *xixk_dist_sq_, const Dtype alpha) {
  CUDA_KERNEL_LOOP(i, count) {
	//printf("bottom_index = %f \n", bottom_index);
	//printf("channel = %f \n", channels);
    
	int n = i / channels;  // the num index, to access y and dist_sq
		if(bottom_index < 3){ //triple
			//derivative of max function
			//printf("Loss = %f \n", 1-(sqrt(xixk_dist_sq_[n]) / (sqrt(xixj_dist_sq_[n]) + Dtype(1e-2))));
			//LOG(INFO) << "Loss : " << 1-(sqrt(xixk_dist_sq_[n]) / (sqrt(xixj_dist_sq_[n]) + Dtype(1e-2)));
			if(sqrt(xixk_dist_sq_[n]) / (sqrt(xixj_dist_sq_[n]) + Dtype(1e-2)) < 1){
				//gradient of loss equation
				if(bottom_index == 0){ //dLoss/dxi
					bottom_diff[i] = -((xixk_diff_[i]/(sqrt(xixk_dist_sq_[n]) + Dtype(1e-3))) * (sqrt(xixj_dist_sq_[n]) + Dtype(1e-2)) - (sqrt(xixk_dist_sq_[n]) * (xixj_diff_[i] / (sqrt(xixj_dist_sq_[n]) + Dtype(1e-3)))));
					bottom_diff[i] /= powf(sqrt(xixj_dist_sq_[n]) + Dtype(1e-2), 2);
					//printf("dLoss/dxi = %f \n", bottom_diff[i]);
				}
				else if (bottom_index == 1){ //dLoss/dxj
					bottom_diff[i] = -(sqrt(xixk_dist_sq_[n]) * (xixj_diff_[i] / (sqrt(xixj_dist_sq_[n]) + Dtype(1e-3))));
					bottom_diff[i] /= powf(sqrt(xixj_dist_sq_[n]) + Dtype(1e-2), 2);
				}			
				else if (bottom_index == 2){ //dLoss/dxk
					bottom_diff[i] = xixk_diff_[i] / (sqrt(xixk_dist_sq_[n]) + Dtype(1e-3)) ;
					bottom_diff[i] /= sqrt(xixj_dist_sq_[n]) + Dtype(1e-2);
				}
			}
			else{
				bottom_diff[i] = 0;
			}
		}
		else if(bottom_index >= 3){ //pair 
			//gradient of loss equation
			if (bottom_index == 3){ //dLoss/dxi_p
				bottom_diff[i] = 2 * xixj_p_diff_[i];
			}
			else if (bottom_index == 4){ //dLoss/dxj_p
				bottom_diff[i] = -(2 * xixj_p_diff_[i]);
			}  
		}
bottom_diff[i] *= alpha;
  }
}

template <typename Dtype>
void TriplePairEuclideanLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
	for(int i = 0; i < 5; ++i){
	   if (propagate_down[i]) {
            int count = bottom[0]->count();
            int channels = bottom[0]->channels();
          const Dtype alpha = top[0]->cpu_diff()[0] / static_cast<Dtype>(bottom[0]->num());
		  CLLBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
		  count, channels, i, bottom[i]->mutable_gpu_diff(), 
		  xixj_diff_.gpu_data(), xixk_diff_.gpu_data(), xixj_p_diff_.gpu_data(),
		  xixj_dist_sq_.gpu_data(), xixk_dist_sq_.gpu_data(), alpha);
		  CUDA_POST_KERNEL_CHECK;
		  //LOG(INFO) << channels<< "   bindex: " << i;
	  }
	  
	  
    }
}

INSTANTIATE_LAYER_GPU_FUNCS(TriplePairEuclideanLossLayer);

}  // namespace caffe
