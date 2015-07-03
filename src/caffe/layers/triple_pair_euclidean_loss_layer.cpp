#include <algorithm>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/loss_layers.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void TriplePairEuclideanLossLayer<Dtype>::LayerSetUp(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);
  CHECK_EQ(bottom[0]->channels(), bottom[1]->channels());
  CHECK_EQ(bottom[0]->height(), 1);
  CHECK_EQ(bottom[0]->width(), 1);
  CHECK_EQ(bottom[1]->height(), 1);
  CHECK_EQ(bottom[1]->width(), 1);
  CHECK_EQ(bottom[2]->height(), 1);
  CHECK_EQ(bottom[2]->width(), 1);
  diff_.Reshape(bottom[0]->num(), bottom[0]->channels(), 1, 1);
  xixj_diff_.Reshape(bottom[0]->num(), bottom[0]->channels(), 1, 1);
  xixk_diff_.Reshape(bottom[0]->num(), bottom[0]->channels(), 1, 1);
  xixj_p_diff_.Reshape(bottom[0]->num(), bottom[0]->channels(), 1, 1);
  diff_sq_.Reshape(bottom[0]->num(), bottom[0]->channels(), 1, 1);
  dist_sq_.Reshape(bottom[0]->num(), 1, 1, 1);
  xixj_dist_sq_.Reshape(bottom[0]->num(), bottom[0]->channels(), 1, 1);
  xixk_dist_sq_.Reshape(bottom[0]->num(), bottom[0]->channels(), 1, 1);
  xixj_p_dist_sq_.Reshape(bottom[0]->num(), bottom[0]->channels(), 1, 1);
  // vector of ones used to sum along channels
  summer_vec_.Reshape(bottom[0]->channels(), 1, 1, 1);
  for (int i = 0; i < bottom[0]->channels(); ++i)
    summer_vec_.mutable_cpu_data()[i] = Dtype(1);
}

template <typename Dtype>
void TriplePairEuclideanLossLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  int count = bottom[0]->count();

  caffe_sub(
      count,
      bottom[0]->cpu_data(),  // a
      bottom[1]->cpu_data(),  // b
      xixj_diff_.mutable_cpu_data());  // a_i-b_i
  const int channels = bottom[0]->channels();
  Dtype loss(0.0);
  
  for (int i = 0; i < bottom[0]->num(); ++i) {
    xixj_dist_sq_.mutable_cpu_data()[i] = caffe_cpu_dot(channels,
        xixj_diff_.cpu_data() + (i*channels), xixj_diff_.cpu_data() + (i*channels));    
  }
  caffe_sub(
      count,
      bottom[0]->cpu_data(),  // a
      bottom[2]->cpu_data(),  // b
      xixk_diff_.mutable_cpu_data());  // a_i-b_i
  
  for(int j=0; j<bottom[0]->num(); j++) {
	xixk_dist_sq_.mutable_cpu_data()[j] = caffe_cpu_dot(channels,
        xixk_diff_.cpu_data() + (j*channels), xixk_diff_.cpu_data() + (j*channels));
  }

  caffe_sub(
      count,
      bottom[3]->cpu_data(),  // a
      bottom[4]->cpu_data(),  // b
      xixj_p_diff_.mutable_cpu_data());  // a_i-b_i
	  
  for(int k=0; k<bottom[0]->num(); k++) {
	xixj_p_dist_sq_.mutable_cpu_data()[k] = caffe_cpu_dot(channels,
        xixj_p_diff_.cpu_data() + (k*channels), xixj_p_diff_.cpu_data() + (k*channels));
  }

  for (int i = 0; i < bottom[0]->num(); ++i) {
        loss += std::max(Dtype(0.0), static_cast<Dtype>(1-(sqrt(xixk_dist_sq_.cpu_data()[i])/(sqrt(xixj_dist_sq_.cpu_data()[i]) + Dtype(1e-2)))));
        loss += xixj_p_dist_sq_.cpu_data()[i];
  }

  top[0]->mutable_cpu_data()[0] = loss / static_cast<Dtype>(bottom[0]->num());
}

template <typename Dtype>
void TriplePairEuclideanLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
	
	for(int i = 0; i < 5; ++i){
	  Dtype* bout = bottom[i]->mutable_cpu_diff();
	  const Dtype alpha = top[0]->cpu_diff()[0] / static_cast<Dtype>(bottom[i]->num());
	  int channels = bottom[i]->channels();
	  for (int j = 0; j < bottom[0]->count(); ++j) {
		  int n = j / channels;
		  if(propagate_down[i] && i < 3){ //triple
				//gradient of loss equation
			if(sqrt(xixk_dist_sq_.mutable_cpu_data()[n]) / (sqrt(xixj_dist_sq_.mutable_cpu_data()[n]) + Dtype(1e-2)) < 1){ //derivative of max function 
				if(i == 0){ //dLoss/dxi
					bout[j] = -((xixk_diff_.mutable_cpu_data()[j]/(sqrt(xixk_dist_sq_.mutable_cpu_data()[n]) + Dtype(1e-3))) * (sqrt(xixj_dist_sq_.mutable_cpu_data()[n]) + Dtype(1e-2)) - (sqrt(xixk_dist_sq_.mutable_cpu_data()[n]) * (xixj_diff_.mutable_cpu_data()[j] / (sqrt(xixj_dist_sq_.mutable_cpu_data()[n]) + Dtype(1e-3)))));
					bout[j] /= powf(sqrt(xixj_dist_sq_.mutable_cpu_data()[n]) + Dtype(1e-2), 2);
				}
				else if (i == 1){ //dLoss/dxj
					bout[j] = -(sqrt(xixk_dist_sq_.mutable_cpu_data()[n]) * (xixj_diff_.mutable_cpu_data()[j] / (sqrt(xixj_dist_sq_.mutable_cpu_data()[n]) + Dtype(1e-3))));
					bout[j] /= powf(sqrt(xixj_dist_sq_.mutable_cpu_data()[n]) + Dtype(1e-2), 2);
				}			
				else if (i == 2){ //dLoss/dxk
					bout[j] = xixk_diff_.mutable_cpu_data()[j] / (sqrt(xixk_dist_sq_.mutable_cpu_data()[n]) + Dtype(1e-3)) ;
					bout[j] /= sqrt(xixj_dist_sq_.mutable_cpu_data()[n]) + Dtype(1e-2);
				}
			}
			else { 
				bout[j] = 0.0;
			}
		  }
		  else if(propagate_down[i] && i >= 3){ //pair
			  //gradient of loss equation
			if (i == 3){ //dLoss/dxi_p
				bout[j] = 2 * xixj_p_diff_.mutable_cpu_data()[j];
			}
			else if (i == 4){ //dLoss/dxj_p
				bout[j] = -(2 * xixj_p_diff_.mutable_cpu_data()[j]);
			}
		  }
		  bout[j] *= alpha;
	  }
    }
}

#ifdef CPU_ONLY
STUB_GPU(TriplePairEuclideanLossLayer);
#endif

INSTANTIATE_CLASS(TriplePairEuclideanLossLayer);
REGISTER_LAYER_CLASS(TriplePairEuclideanLoss);

}  // namespace caffe
