#ifndef QD_2_HPP
#define QD_2_HPP

// quick hack to have "write" access to the container, this need to be added to the main API later.
template<typename Phen, typename Eval, typename Stat, typename FitModifier, typename Select, typename Container, typename Params, typename Exact = stc::Itself>
class QualityDiversity_2
        : public sferes::qd::QualityDiversity<Phen, Eval, Stat, FitModifier, Select, Container, Params, typename stc::FindExact<QualityDiversity_2<Phen, Eval, Stat, FitModifier, Select, Container, Params, Exact>, Exact>::ret> {

public:
    typedef Phen phen_t;
    typedef boost::shared_ptr <Phen> indiv_t;
    typedef Eigen::Map<const Eigen::VectorXd> point_t;
    typedef typename std::vector<indiv_t> pop_t;
    typedef typename pop_t::iterator it_t;

    pop_t pop_advers;
    // pop_t& get_pop_advers() { return this->pop_advers; }

    Container &container() { return this->_container; }

    void add(pop_t &pop_off, std::vector<bool> &added, pop_t &pop_parents) {
        _add(pop_off, added, pop_parents);
    }

    // Same function, but without the need of parent.
    void add(pop_t &pop_off, std::vector<bool> &added) {
        std::cout << "adding with l: " << Params::nov::l[0] << std::endl;
        _add(pop_off, added);
    }

    void _add(pop_t& pop_off, std::vector<bool>& added, pop_t& pop_parents)
    {
        added.resize(pop_off.size());
        for (size_t i = 0; i < pop_off.size(); ++i)
            added[i] = this->_add_to_container(pop_off[i], pop_parents[i]);
        container().update(pop_off, pop_parents);
        
        // updating the parents is not needed as that is just for the curiosity score, so dont need to do anything with pop_parents
        if (Params::qd::num_train_archives > 0)
        {
            std::vector<bool> bool_added = added;
            pop_t copy_pop_off = pop_off;
            add_to_train_archives(bool_added, copy_pop_off);
        }
    }

    // Same function, but without the need of parent.
    void _add(pop_t& pop_off, std::vector<bool>& added)
    {
        added.resize(pop_off.size());
        for (size_t i = 0; i < pop_off.size(); ++i)
            added[i] = container().add(pop_off[i]);
        pop_t empty;
        container().update(pop_off, empty);

        if (Params::qd::num_train_archives > 0)
        {
            std::vector<bool> bool_added = added;
            pop_t copy_pop_off = pop_off;
            add_to_train_archives(bool_added, copy_pop_off);
        }
    }

    void add_to_train_archives(std::vector<bool> &bool_added, pop_t &copy_pop_off)
    {
        std::vector<bool> remaining;
        // loop through archives and add remaining phenotypes each time
        for (int i{0}; i < Params::qd::num_train_archives; ++i)
        {
            // extract the phens that were not added before
            pop_t phen_to_be_added;
            for (size_t j{0}; j < bool_added.size(); ++j)
            {
                if (!bool_added[j])
                    {phen_to_be_added.push_back(copy_pop_off[j]);}
            }
            
            remaining.resize(phen_to_be_added.size());

            // add into current archive
            for (size_t j{0}; j < remaining.size(); ++j)
                {remaining[j] = train_archives[i].add(phen_to_be_added[j], i + 1);}

            // if all added can break early
            if (std::find(remaining.begin(), remaining.end(), false) == remaining.end())
                {break;}

            bool_added = remaining;
            copy_pop_off = phen_to_be_added;
        }

        // if after the last iteration there are still phenotypes to be added
        if (std::find(remaining.begin(), remaining.end(), false) != remaining.end())
        {   
            // extract the phens that were not added before
            pop_t phen_to_be_added;
            for (size_t j{0}; j < bool_added.size(); ++j)
            {
                if (!bool_added[j])
                    {phen_to_be_added.push_back(copy_pop_off[j]);}
            }
            std::cout << "Still to be added after looping through all containers: " << phen_to_be_added.size() << "\n";

            for (size_t i{0}; i < phen_to_be_added.size(); ++i)
            {
                // pick random container
                int container_idx = rand() % Params::qd::num_train_archives;
                // std::cout << "Container index chosen for replacement: " << container_idx << "\n";

                Eigen::VectorXd phen_desc(Params::qd::behav_dim);
                for (int l{0}; l < phen_desc.size(); ++l)
                    {phen_desc[l] = phen_to_be_added[i]->fit().desc()[l];}
                
                indiv_t to_be_removed = train_archives[container_idx].archive().nearest(phen_desc).second;
                // assert((to_be_removed) != (phen_to_be_added[i]));
                train_archives[container_idx].replace(to_be_removed, phen_to_be_added[i]);
                
                // std::cout << "Replaced \n" << phen_desc << "\nwith \n"; 
                // for (float k : to_be_removed->fit().desc())
                //     {std::cout << k << "\n";}
            }
        }
    }

    void get_full_content_train_archives(std::vector<indiv_t> &content)
    {
        for (int i{0}; i < Params::qd::num_train_archives; ++i)
            {train_archives[i].get_full_content(content);}
    }

    Container &train_container(int i)
        {return train_archives[i];}

    // Main Iteration of the QD algorithm, overriding here so it chooses the new _add methods
    void epoch()
    {
        this->_parents.resize(Params::pop::size);

        // Selection of the parents (will fill the _parents vector)
        this->_selector(this->_parents, *this); // not a nice API

        // CLEAR _offspring ONLY after selection, as it can be
        // used by the selector (via this->_offspring)
        this->_offspring.clear();
        this->_offspring.resize(Params::pop::size);

        // Generation of the offspring
        std::vector<size_t> a;
        sferes::misc::rand_ind(a, this->_parents.size());
        assert(this->_parents.size() == Params::pop::size);
        for (size_t i = 0; i < Params::pop::size; i += 2) {
            boost::shared_ptr<Phen> i1, i2;
            this->_parents[a[i]]->cross(this->_parents[a[i + 1]], i1, i2);
            i1->mutate();
            i2->mutate();
            i1->develop();
            i2->develop();
            this->_offspring[a[i]] = i1;
            this->_offspring[a[i + 1]] = i2;
        }

        // Evaluation of the offspring
        this->_eval_pop(this->_offspring, 0, this->_offspring.size());
        this->apply_modifier();

        // Addition of the offspring to the container
        _add(this->_offspring, this->_added, this->_parents);

        assert(this->_offspring.size() == this->_parents.size());

        this->_pop.clear();

        // Copy of the containt of the container into the _pop object.
        container().get_full_content(this->_pop);

        for (size_t j{0}; j < train_archives.size(); ++j)
        {std::cout << "Training Container " << j << " holds " << train_archives[j].archive().size() << " individuals. L: "<< Params::nov::l[j + 1] << "\n";}
    }

    std::array<Container, Params::qd::num_train_archives> train_archives;
};

#endif
