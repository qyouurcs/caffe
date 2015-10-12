#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/loss_layers.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {
    template <typename Dtype>
    void PairwiseLossLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*> &bottom,
            const vector <Blob<Dtype>*> &top){
        // v, label, and word2vec_fea
        CHECK_EQ(bottom[0] -> num(), bottom[1] -> num());
        CHECK_EQ(bottom[0] -> channels(), bottom[2] -> channels());

        int num = bottom[0]->num();
        int count = bottom[0]->count();
        int dim = count / num;
        im_diff.Reshape(dim, 1,1,1);
    }
    template <typename Dtype>
    void PairwiseLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
            const vector<Blob<Dtype>*> &top){
        const Dtype* bottom_data = bottom[0]->cpu_data();
        const Dtype* label = bottom[1]->cpu_data();
        const Dtype* word2vec = bottom[2]->cpu_data();

        // get the margin.
        Dtype margin = this->layer_param_.pairwise_loss_param().margin();

        Dtype* bottom_diff = bottom[0] -> mutable_cpu_diff();
        Dtype* im_diff_p = im_diff.mutable_cpu_diff();
        
        Dtype* loss = top[0]->mutable_cpu_data();
        loss[0] = 0;
        int num = bottom[0]->num();
        int count = bottom[0]->count();
        int dim = count / num;
        int classes = bottom[1]->count() / num;

        caffe_set(count, (Dtype) 0.0, bottom_diff);
        for (int i = 0; i < num; i++){
            int pair = 0;
            Dtype im_loss = 0.0;
            caffe_set( dim, (Dtype) 0.0, im_diff_p);
            for (int j = 0; j < classes; j++){
                for (int k = j; k < classes; k++){
                    //int lbl_j = static_cast<int>(label[i*classes + j]);
                    //int lbl_k = static_cast<int>(label[i*classes + k]);
                    Dtype lbl_j = label[i * classes + j];
                    Dtype lbl_k = label[i * classes + k];
                    if (lbl_j > 0 && lbl_k < 0){
                        // different category.
                        pair ++;
                        Dtype inner_j = caffe_cpu_dot(dim, bottom_data + i * dim, word2vec + j * dim);
                        Dtype inner_k = caffe_cpu_dot(dim, bottom_data + i * dim, word2vec + k * dim);
                        
                        Dtype t = margin - inner_j + inner_k;
                        if( t > 0 ){
                            im_loss += t;
                            for (int m = 0; m < dim; m++){
                                im_diff_p[m] += -(*(word2vec + j*dim + m));
                                im_diff_p[m] += *(word2vec + k*dim + m);
                            }
                        }
                    }
                }
            }
            if(pair > 0){ // There is possibility that all labels are correctly calculated.
                caffe_scal(dim, (Dtype)1.0 /pair, im_diff_p);
                loss[0] += im_loss / pair;
                caffe_copy(dim, im_diff_p, bottom_diff +  i * dim);
            }
        }
        loss[0] /= num;
    }
    template <typename Dtype>
    void PairwiseLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
        const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom){
        if(propagate_down[1]){
            LOG(FATAL) << this->type()
                       << " Layer cannot bp to label inputs.";
        }
        if(propagate_down[2]){
            LOG(FATAL) << this->type()
                       << " Layer cannot bp to inpput txt feature inputs.";
        }
        const Dtype loss_weight = top[0] -> cpu_diff()[0];
        int num = bottom[0] -> num();
        int count = bottom[0] -> count();
        Dtype* bottom_diff = bottom[0] -> mutable_cpu_diff();
        caffe_scal(count, loss_weight/(num), bottom_diff);
    }
    INSTANTIATE_CLASS(PairwiseLossLayer);
    REGISTER_LAYER_CLASS(PairwiseLoss);
}
