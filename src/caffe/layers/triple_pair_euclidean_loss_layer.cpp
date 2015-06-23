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

  double m = 0.01;
  caffe_sub(
      count,
      bottom[0]->cpu_data(),  // a
      bottom[1]->cpu_data(),  // b
      xixj_diff_.mutable_cpu_data());  // a_i-b_i
  const int channels = bottom[0]->channels();
  //Dtype margin = this->layer_param_.contrastive_loss_param().margin();
  //bool legacy_version = this->layer_param_.contrastive_loss_param().legacy_version();
  Dtype loss(0.0);
  double tempDenominator = 0.0;
  double tempNumerator = 0.0;
  
  for (int i = 0; i < bottom[0]->num(); ++i) {
    xixj_dist_sq_.mutable_cpu_data()[i] = caffe_cpu_dot(channels,
        xixj_diff_.cpu_data() + (i*channels), xixj_diff_.cpu_data() + (i*channels));
    tempDenominator += xixj_dist_sq_.mutable_cpu_data()[i];      
  }
  tempDenominator = sqrt(tempDenominator) + m;  
  tempDenominator += m;
  caffe_sub(
      count,
      bottom[0]->cpu_data(),  // a
      bottom[2]->cpu_data(),  // b
      xixk_diff_.mutable_cpu_data());  // a_i-b_i
  
  for(int j=0; j<bottom[0]->num(); j++) {
	xixk_dist_sq_.mutable_cpu_data()[j] = caffe_cpu_dot(channels,
        xixk_diff_.cpu_data() + (j*channels), xixk_diff_.cpu_data() + (j*channels));
      tempNumerator += xixk_dist_sq_.mutable_cpu_data()[j];
  }
  tempNumerator = sqrt(tempNumerator);
  Dtype dist = std::max(1-(tempNumerator/tempDenominator), 0.0);
  loss += dist;

  double tempPairDenominator = 0.0;//tempDenominator;
  caffe_sub(
      count,
      bottom[3]->cpu_data(),  // a
      bottom[4]->cpu_data(),  // b
      xixj_p_diff_.mutable_cpu_data());  // a_i-b_i
	  
  for(int k=0; k<bottom[0]->num(); k++) {
	xixj_p_dist_sq_.mutable_cpu_data()[k] = caffe_cpu_dot(channels,
        xixj_p_diff_.cpu_data() + (k*channels), xixj_p_diff_.cpu_data() + (k*channels));
  	tempPairDenominator += xixj_p_dist_sq_.mutable_cpu_data()[k];
  }
  loss += tempPairDenominator;
  
  //loss = loss / static_cast<Dtype>(bottom[0]->num()) / Dtype(2);
  top[0]->mutable_cpu_data()[0] = loss;
}

template <typename Dtype>
void TriplePairEuclideanLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  //Dtype margin = this->layer_param_.contrastive_loss_param().margin();
  //bool legacy_version = this->layer_param_.contrastive_loss_param().legacy_version();


    for(int i = 0; i < 5; ++i){
	  Dtype* bout = bottom[i]->mutable_cpu_diff();
	  Dtype bottom_diff_val(0.0);
	  int num = bottom[i]->num();
	  int channels = bottom[i]->channels();
	  for (int j = 0; j < num; ++j) {
		  if(propagate_down[i] && i < 3){ //triple
				//gradient of loss equation
			if(i == 0){ //dLoss/dxi
				bottom_diff_val = sqrt(xixk_dist_sq_.mutable_cpu_data()[j])/ sqrt(xixj_dist_sq_.mutable_cpu_data()[j]);
			}
			else if (i == 1){ //dLoss/dxj
				bottom_diff_val = -(bottom[i]->mutable_cpu_data()[j] / sqrt(xixj_dist_sq_.mutable_cpu_data()[j]));
			}			
			else if (i == 2){ //dLoss/dxk
				bottom_diff_val = -(sqrt(xixk_dist_sq_.mutable_cpu_data()[j]) / bottom[i]->mutable_cpu_data()[j]);
			}
		  }
		  else if(propagate_down[i] && i >= 3){ //pair
			  //gradient of loss equation
			if (i == 3){ //dLoss/dxi_p
				bottom_diff_val = 2 * bottom[i]->mutable_cpu_data()[j];
			}
			else if (i == 4){ //dLoss/dxj_p
				bottom_diff_val = -(2 * bottom[i]->mutable_cpu_data()[j]);
			}
		  }
		  if (bottom_diff_val > 0.0){
			bout[j] = bottom_diff_val;
		  }
		  else{
			bout[j] = 0;
		  }
	  }
    }
/*  for (int i = 0; i < 2; ++i) {
    if (propagate_down[i]) {
      const Dtype sign = (i == 0) ? 1 : -1;
      const Dtype alpha = sign * top[0]->cpu_diff()[0] /
          static_cast<Dtype>(bottom[i]->num());
      int num = bottom[i]->num();
      int channels = bottom[i]->channels();
      for (int j = 0; j < num; ++j) {
        Dtype* bout = bottom[i]->mutable_cpu_diff();
        if (static_cast<int>(bottom[2]->cpu_data()[j])) {  // similar pairs
          caffe_cpu_axpby(
              channels,
              alpha,
              diff_.cpu_data() + (j*channels),
              Dtype(0.0),
              bout + (j*channels));
        } else {  // dissimilar pairs
          Dtype mdist(0.0);
          Dtype beta(0.0);
          if (legacy_version) {
            mdist = margin - dist_sq_.cpu_data()[j];
            beta = -alpha;
          } else {
            Dtype dist = sqrt(dist_sq_.cpu_data()[j]);
            mdist = margin - dist;
            beta = -alpha * mdist / (dist + Dtype(1e-4));
          }
          if (mdist > Dtype(0.0)) {
            caffe_cpu_axpby(
                channels,
                beta,
                diff_.cpu_data() + (j*channels),
                Dtype(0.0),
                bout + (j*channels));
          } else {
            caffe_set(channels, Dtype(0), bout + (j*channels));
          }
        }
      }
    }
  }*/
}

#ifdef CPU_ONLY
STUB_GPU(TriplePairEuclideanLossLayer);
#endif

INSTANTIATE_CLASS(TriplePairEuclideanLossLayer);
REGISTER_LAYER_CLASS(TriplePairEuclideanLoss);

}  // namespace caffe