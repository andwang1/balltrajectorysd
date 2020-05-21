//
// Created by Luca Grillotti on 31/10/2019.
//

#ifndef SFERES2_PREPROCESSOR_HPP
#define SFERES2_PREPROCESSOR_HPP

#include <iostream>

class RescaleFeature {
public:
    RescaleFeature() : no_prep(true) {}

    typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> Mat;

    void init() {
        no_prep = true;
    }

    void init(const Mat &data) {
        no_prep = false;
        m_mean_dataset = data.colwise().mean();
        m_std_dataset = (data.array().rowwise() - m_mean_dataset.transpose().array()).pow(2).colwise().mean().sqrt();
    }

    void apply(const Mat &data, Mat &res) const {
        if (no_prep) {
            res = data;
        } else {
            // res = a + (data.array() - _min) * (b - a) / (_max - _min);
            res = data.array().rowwise() - m_mean_dataset.transpose().array();
            res = res.array().rowwise() / (m_std_dataset.transpose().array() + 0.001f);
        }
    }

    void deapply(const Mat &data, Mat &res) const {
        if (no_prep) {
            res = data;
        } else {
            res = data.array().rowwise() * m_std_dataset.transpose().array();
            res = res.array().rowwise() + m_mean_dataset.transpose().array();
            res = res.cwiseMax(0.).cwiseMin(1.);
        }
    }

private:
    bool no_prep;
    Eigen::VectorXf m_mean_dataset;
    Eigen::VectorXf m_std_dataset;
};

#endif //SFERES2_PREPROCESSOR_HPP
