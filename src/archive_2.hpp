#ifndef ARCHIVE_2_HPP
#define ARCHIVE_2_HPP


namespace sferes {
    namespace qd {

        namespace container {
            // quick hack to have "write" access to storage
            template <typename Phen, typename Storage, typename Params> 
            class Archive_2 : public sferes::qd::container::Archive<Phen, Storage, Params>
            {
                public:
                typedef boost::shared_ptr<Phen> indiv_t;
                typedef typename std::vector<indiv_t> pop_t;
                typedef typename pop_t::iterator it_t;
                typedef Eigen::Map<const Eigen::VectorXd> point_t;
                typedef Storage storage_t;
                
                using knn_iterator_t = typename std::vector<typename Storage::data_t>::iterator;
                using const_knn_iterator_t = typename std::vector<typename Storage::data_t>::const_iterator;
                
                storage_t& archive() {return this->_archive;}

                void replace(const indiv_t& toberemoved, const indiv_t& tobeinserted)
                    {this->_replace(toberemoved, tobeinserted);}

                bool add(indiv_t i1, int container_idx = 0)
                {
                    // p_i1 is the behavioral coordinate of i1
                    point_t p_i1(i1->fit().desc().data(), i1->fit().desc().size());

                    if (i1->fit().dead())
                        return false;
                    if (archive().size() == 0 || (archive().nearest(p_i1).first - p_i1).norm() > Params::nov::l[container_idx]) // ADD because new
                    {
                        // this is novel enough, we add
                        this->direct_add(i1);
                        return true;
                    }
                    else if (archive().size() == 1) {
                        // there is only one indiv in the archive and the current one is too close
                        return false;
                    }
                    else // archive size >= 2
                    {
                        auto neigh_cur = archive().knn(p_i1, 2);
                        if ((p_i1 - neigh_cur[1].first).norm() < (1 - Params::nov::eps) * Params::nov::l[container_idx]) // too close the second NN -- this works better
                        {
                            // too close the second nearest neighbor, we skip
                            return false;
                        }
                        auto nn = neigh_cur[0].second;
                        std::vector<double> score_cur(2, 0), score_nn(2, 0);
                        score_cur[0] = i1->fit().value();
                        score_nn[0] = nn->fit().value();

                        // Compute the Novelty
                        neigh_cur.clear();
                        auto neigh_nn=neigh_cur;
                        point_t p_nn(nn->fit().desc().data(), nn->fit().desc().size());

                        // we look for the k+1 nearest neighbours as the first one is "nn" which might be or not replaced.
                        neigh_cur = archive().knn(p_i1, Params::nov::k + 1);
                        neigh_nn = archive().knn(p_nn, Params::nov::k + 1);
                
                        score_cur[1] = this->get_novelty(i1, ++neigh_cur.begin(), neigh_cur.end());
                        score_nn[1] = this->get_novelty(nn, ++neigh_nn.begin(), neigh_nn.end());

                        // TEST
                        if ((score_cur[0] >= (1 - sign(score_nn[0]) * Params::nov::eps) * score_nn[0] &&
                            score_cur[1] >= (1 - sign(score_nn[1]) * Params::nov::eps) * score_nn[1]) &&
                            ((score_cur[0] - score_nn[0]) * std::abs(score_nn[1]) > -(score_cur[1] - score_nn[1]) * std::abs(score_nn[0])))
                        {
                            // add if significatively better on one objective
                            this->_replace(nn, i1);
                            return true;
                        }
                        else
                        {
                            return false;
                        }
                    }
                }
            };
        }
    }
}
#endif