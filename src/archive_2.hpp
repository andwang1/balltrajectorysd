#ifndef ARCHIVE_2_HPP
#define ARCHIVE_2_HPP

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
};

#endif