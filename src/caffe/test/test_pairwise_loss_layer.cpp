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
class PairwiseLossLayerTest: public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  PairwiseLossLayerTest()
      : blob_bottom_data_(new Blob<Dtype>(256, 20, 1, 1)),
        blob_bottom_label_(new Blob<Dtype>(256, 5, 1, 1)),
        blob_bottom_word2vec_(new Blob<Dtype>(5,20, 1,1)),
        blob_top_loss_(new Blob<Dtype>()) {
    // fill the values
    Caffe::set_random_seed(1701);
    FillerParameter filler_param;
    filler_param.set_std(1);
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_data_);
    filler_param.set_std(1);
    filler.Fill(this->blob_bottom_word2vec_);
    blob_bottom_vec_.push_back(blob_bottom_data_);

    for (int i = 0; i < blob_bottom_label_->count(); ++i) {
      if ((caffe_rng_rand() % 2) == 1)
        this->blob_bottom_label_->mutable_cpu_data()[i] = 1;
      else
        this->blob_bottom_label_->mutable_cpu_data()[i] = -1;
    }
    blob_bottom_vec_.push_back(blob_bottom_label_);
    blob_bottom_vec_.push_back(blob_bottom_word2vec_);
    blob_top_vec_.push_back(blob_top_loss_);
  }
 public:
  virtual ~PairwiseLossLayerTest() {
    delete blob_bottom_data_;
    delete blob_bottom_label_;
    delete blob_bottom_word2vec_;
    delete blob_top_loss_;
  }
  Blob<Dtype>* const blob_bottom_data_;
  Blob<Dtype>* const blob_bottom_label_;
  Blob<Dtype>* const blob_bottom_word2vec_;
  Blob<Dtype>* const blob_top_loss_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(PairwiseLossLayerTest, TestDtypesAndDevices);

TYPED_TEST(PairwiseLossLayerTest, TestForward) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  PairwiseLossLayer<Dtype> layer(layer_param);

  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);  
  const Dtype margin = layer_param.pairwise_loss_param().margin();
  const int num = this->blob_bottom_data_->num();  
  const int count = this->blob_bottom_data_->count();
  const int dim = count / num;
  const int classes = this->blob_bottom_label_->count()/num;
  Dtype loss(0);
  for(int i=0; i<num; i++)
  {
    Dtype im_loss(0.0);
    int pair = 0;
    for (int j = 0; j < classes; j++)
      for (int k = j; k < classes; k++)
      {
        Dtype t = 0.0;
        if( (this->blob_bottom_label_->cpu_data()[i*classes+ j]) > 0 && (this->blob_bottom_label_->cpu_data()[i*classes+ k]) < 0)
        {
          pair++;
          for(int m = 0; m < dim; m++)
          {
            t += -(this->blob_bottom_data_->cpu_data()[i * dim + m]) * (this->blob_bottom_word2vec_->cpu_data()[j * dim + m]);
            t += (this->blob_bottom_data_->cpu_data()[i * dim + m]) * (this->blob_bottom_word2vec_->cpu_data()[k * dim + m]);
          }
          if ( t + margin > 0)
          {
              im_loss += t + margin;
          }
        }
      }
    if( pair > 0)
      loss += im_loss / pair;
  }
  loss /= num;
  EXPECT_NEAR(this->blob_top_loss_->cpu_data()[0], loss, 1e-4); 
}

TYPED_TEST(PairwiseLossLayerTest, TestGradient) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  PairwiseLossLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 1e-2, 1701);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_, 0);
}

}
