//| This file is a part of the sferes2 framework.
//| Copyright 2009, ISIR / Universite Pierre et Marie Curie (UPMC)
//| Main contributor(s): Jean-Baptiste Mouret, mouret@isir.fr
//|
//| This software is a computer program whose purpose is to facilitate
//| experiments in evolutionary computation and evolutionary robotics.
//|
//| This software is governed by the CeCILL license under French law
//| and abiding by the rules of distribution of free software.  You
//| can use, modify and/ or redistribute the software under the terms
//| of the CeCILL license as circulated by CEA, CNRS and INRIA at the
//| following URL "http://www.cecill.info".
//|
//| As a counterpart to the access to the source code and rights to
//| copy, modify and redistribute granted by the license, users are
//| provided only with a limited warranty and the software's author,
//| the holder of the economic rights, and the successive licensors
//| have only limited liability.
//|
//| In this respect, the user's attention is drawn to the risks
//| associated with loading, using, modifying and/or developing or
//| reproducing the software by the user in light of its specific
//| status of free software, that may mean that it is complicated to
//| manipulate, and that also therefore means that it is reserved for
//| developers and experienced professionals having in-depth computer
//| knowledge. Users are therefore encouraged to load and test the
//| software's suitability as regards their requirements in conditions
//| enabling the security of their systems and/or data to be ensured
//| and, more generally, to use and operate it in the same conditions
//| as regards security.
//|
//| The fact that you are presently reading this means that you have
//| had knowledge of the CeCILL license and that you accept its terms.




// doing this to control the angle and dpf seperately, might no longer be necessary once the simulation is in place and can
// then likely go back to old parameters only



#ifndef PHEN_HPP_
#define PHEN_HPP_

#include <vector>
#include <sferes/phen/indiv.hpp>
#include <boost/foreach.hpp>

namespace sferes {
  namespace phen {
    SFERES_INDIV(Custom_Phen, Indiv) {

      template<typename G, typename F, typename P, typename E>
      friend std::ostream& operator<<(std::ostream& output, const Custom_Phen< G, F, P, E >& e);
    public:
#ifdef EIGEN_CORE_H
      EIGEN_MAKE_ALIGNED_OPERATOR_NEW
#endif
      Custom_Phen() : _params(Params::qd::gen_dim), split(static_cast<int>((*this)._gen.size() / 2))  
      {}
      typedef float type_t;
      void develop() {
        // need to do this in here bcs of the reevaluation possiblity
        std::fill(_params.begin(), _params.end(), 0);
        for (unsigned i = 0; i < _params.size(); ++i)
          _params[i] = this->_gen.data(i) * (Params::parameters::max_angle - Params::parameters::min_angle) + Params::parameters::min_angle;
      }

      void simple_summation() {
        for (int i{0}; i < split; ++i)
        {_params[0] += this->_gen.data(i);}

        for (int i{split}; i < (*this)._gen.size(); ++i)
        {_params[1] += this->_gen.data(i);}

        // normalise to 0 - 1 and then scale to max min range
        _params[0] /= split * Params::parameters::max_angle;
        _params[1] = _params[1] / ((*this)._gen.size() - split) * (Params::parameters::max_dpf - Params::parameters::min_dpf) + Params::parameters::min_dpf;
      }

      void non_linear() {
        float tmp[2] = {0, 0};
        float res[2] = {1, 1};
        int input_counter, output_counter, temp_index;

        for (int i{0}; i < (*this)._gen.size(); ++i)
        {
            if (((i % 4) == 0) && i > 0)
            {
                res[0] = tmp[0];
                res[1] = tmp[1];
                tmp[0] = 0;
            }
            else if ((i % 4) == 2)
            {tmp[1] = 0;}

            input_counter = i % 2;
            output_counter = i % 4;
            temp_index = output_counter / 2;

            // gen is scaled to -0.5, 2
            tmp[temp_index] += res[input_counter] * (this->_gen.data(i) * 2.5 - 0.5);
        }

        // flip distribution by making the negative be + 2 -> tried, very sparse so not good
        _params[0] = std::max(tmp[0], 0.f);
        _params[1] = std::max(tmp[1], 0.f);

        // floor division, 4.001 so that multiples of 4 are still in the same exponent group
        int exponent = static_cast<int>((*this)._gen.size() / 4.001) * 2 + 2;

        if ((*this)._gen.size() % 4 == 1)
        {
            _params[0] /= std::pow(2, exponent - 1);
            _params[1] /= std::pow(2, exponent - 2);
        }
        else if ((*this)._gen.size() % 4 == 2)
        {
            _params[0] /= std::pow(2, exponent);
            _params[1] /= std::pow(2, exponent - 2);
        }
        else if ((*this)._gen.size() % 4 == 3)
        {
            _params[0] /= std::pow(2, exponent);
            _params[1] /= std::pow(2, exponent - 1);
        }
        else
        {
            _params[0] /= std::pow(2, exponent);
            _params[1] /= std::pow(2, exponent);
        }

        assert(_params[0] < 1.1);
        assert(_params[1] < 1.1);

        _params[0] *= Params::parameters::max_angle;
        _params[1] *= (Params::parameters::max_dpf - Params::parameters::min_dpf) + Params::parameters::min_dpf;        

      }

      void uniform() {
          // uniform distribution in output
          for (int i{0}; i < split; ++i)
          {_params[0] += this->_gen.data(i) / std::pow(10.f, i);}

          for (int i{split}; i < (*this)._gen.size(); ++i)
          {_params[1] += this->_gen.data(i) / std::pow(10.f, i - split);}

          // making sure angle is bounded, wrapping gives better distribution
          if (_params[0] > 1)
          {_params[0] -= 1;}

          assert(_params[0] < 1.1);
          assert(_params[1] < 1.12);

          _params[0] *= Params::parameters::max_angle;
          _params[1] = _params[1] * (Params::parameters::max_dpf - Params::parameters::min_dpf) + Params::parameters::min_dpf;
      }

      float data(size_t i) const {
        assert(i < size());
        return _params[i];
      }

      size_t size() const {
        return _params.size();
      }
      const std::vector<float>& data() const {
        return _params;
      }
      // squared Euclidean distance
      float dist(const Custom_Phen& params) const {
        assert(params.size() == size());
        float d = 0.0f;
        for (size_t i = 0; i < _params.size(); ++i) {
          float x = _params[i] - params._params[i];
          d += x * x;
        }
        return d;
      }
      void show(std::ostream& os) const {
        BOOST_FOREACH(float p, _params)
        os<<p<<" ";
        os<<std::endl;
      }
    protected:
      std::vector<float> _params;
      int split;
    };
    template<typename G, typename F, typename P, typename E>
    std::ostream& operator<<(std::ostream& output, const Custom_Phen< G, F, P, E >& e) {
      for (size_t i = 0; i < e.size(); ++i)
        output <<" "<<e.data(i) ;
      return output;
    }
  }
}

#endif
