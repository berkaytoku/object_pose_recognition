#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/vision_layers.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

template <typename TypeParam>
class TriplePairEuclideanLossLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  TriplePairEuclideanLossLayerTest()
      : blob_bottom_data_xi_(new Blob<Dtype>(16, 1, 1, 1)),
        blob_bottom_data_xj_(new Blob<Dtype>(16, 1, 1, 1)),
		blob_bottom_data_xk_(new Blob<Dtype>(16, 1, 1, 1)),
		blob_bottom_data_xi_p_(new Blob<Dtype>(16, 1, 1, 1)),
		blob_bottom_data_xj_p_(new Blob<Dtype>(16, 1, 1, 1)),
        blob_top_loss_(new Blob<Dtype>()) {
    // fill the values
    FillerParameter filler_param;
    filler_param.set_min(-1.0);
    filler_param.set_max(1.0);  // distances~=1.0 to test both sides of margin
    UniformFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_data_xi_);
    blob_bottom_vec_.push_back(blob_bottom_data_xi_);
    filler.Fill(this->blob_bottom_data_xj_);
    blob_bottom_vec_.push_back(blob_bottom_data_xj_);
	filler.Fill(this->blob_bottom_data_xk_);
    blob_bottom_vec_.push_back(blob_bottom_data_xk_);
    filler.Fill(this->blob_bottom_data_xi_p_);
    blob_bottom_vec_.push_back(blob_bottom_data_xi_p_);
	filler.Fill(this->blob_bottom_data_xj_p_);
    blob_bottom_vec_.push_back(blob_bottom_data_xj_p_);
    blob_top_vec_.push_back(blob_top_loss_);
  }
  virtual ~TriplePairEuclideanLossLayerTest() {
    delete blob_bottom_data_xi_;
    delete blob_bottom_data_xj_;
	delete blob_bottom_data_xk_;
	delete blob_bottom_data_xi_p_;
    delete blob_bottom_data_xj_p_;
    delete blob_top_loss_;
  }

  Blob<Dtype>* const blob_bottom_data_xi_;
  Blob<Dtype>* const blob_bottom_data_xj_;
  Blob<Dtype>* const blob_bottom_data_xk_;
  Blob<Dtype>* const blob_bottom_data_xi_p_;
  Blob<Dtype>* const blob_bottom_data_xj_p_;
  Blob<Dtype>* const blob_top_loss_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(TriplePairEuclideanLossLayerTest, TestDtypesAndDevices);

TYPED_TEST(TriplePairEuclideanLossLayerTest, TestForward) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  TriplePairEuclideanLossLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  // manually compute to compare
  //const Dtype margin = layer_param.contrastive_loss_param().margin();
  const int num = this->blob_bottom_data_xi_->num();
  const int channels = this->blob_bottom_data_xi_->channels();
  Dtype loss(0);

  for (int i = 0; i < num; ++i) {
	  Dtype xixj_dist_sq(0);
	  Dtype xixk_dist_sq(0);
	  Dtype xixj_p_dist_sq(0);
	  for(int j = 0; j < channels; ++j){
		  Dtype xixj_diff = this->blob_bottom_data_xi_->cpu_data()[i*channels + j] - this->blob_bottom_data_xj_->cpu_data()[i*channels + j];
		  xixj_dist_sq += xixj_diff * xixj_diff;
		  Dtype xixk_diff = this->blob_bottom_data_xi_->cpu_data()[i*channels + j] - this->blob_bottom_data_xk_->cpu_data()[i*channels + j];
		  xixk_dist_sq += xixk_diff * xixk_diff;
		  Dtype xixj_p_diff = this->blob_bottom_data_xi_p_->cpu_data()[i*channels + j] - this->blob_bottom_data_xj_p_->cpu_data()[i*channels + j];
		  xixj_p_dist_sq += xixj_p_diff * xixj_p_diff;
	  }
	  loss += std::max(Dtype(0.0), static_cast<Dtype>(1-(sqrt(xixk_dist_sq)/(sqrt(xixj_dist_sq) + Dtype(1e-2)))));
      loss += xixj_p_dist_sq;
  }
  loss /= static_cast<Dtype>(num);
  EXPECT_NEAR(this->blob_top_loss_->cpu_data()[0], loss, 1e-6); 
}

TYPED_TEST(TriplePairEuclideanLossLayerTest, TestGradient) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  TriplePairEuclideanLossLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  GradientChecker<Dtype> checker(1e-2, 1e-2, 1701);
  // check the gradient for the 5 bottom layers
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_, 0);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_, 1);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_, 2);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_, 3);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_, 4);
}

}  // namespace caffe
