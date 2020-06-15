//
// Created by Andy Wang
//

#ifndef TRAJECTORY_HPP
#define TRAJECTORY_HPP

#include <cmath>
#include <cassert>
#include <array>
#include <bitset>
#include <fstream>
#include <random>

// pseudo random number generator
namespace rng {
    // Will be used to obtain a seed for the random number engine
    std::random_device rd;  
    std::mt19937 gen(rd()); 
    std::uniform_real_distribution<> rng(0, 1);
}

FIT_QD(Trajectory)
{
    public:
    Trajectory(){}

    template <typename Indiv> 
    void eval(Indiv & ind){
        // get the data from ind which is the phenotype
        float angle = ind.data(0);
        float dpf = ind.data(1);

        _angle = angle;
        _dpf = dpf;

        // generate actual true trajectory from phenotype
        Eigen::VectorXf single_traj;
        single_traj.resize(Params::sim::num_trajectory_elements);
        generate_traj(single_traj, _wall_impacts, angle, dpf);
        _trajectories[0] = single_traj;
        
        // track number of random trajectories
        _m_num_trajectories = 0;

        // generate random trajectories
        for (int i{1}; i < Params::random::max_num_random + 1; ++i)
        {
            float prob = rng::rng(rng::gen);
            if (prob <= Params::random::pct_random)
            {
                angle = rng::rng(rng::gen) * Params::parameters::max_angle;
                if (Params::random::is_random_dpf)
                {dpf = rng::rng(rng::gen) * Params::parameters::max_dpf;}
                // generate a random trajectory
                generate_traj(single_traj, angle, dpf);
                // 1 means it is a trajectory
                _is_random_trajectories[i] = 1;
                ++_m_num_trajectories;
            }
            else 
                {_is_random_trajectories[i] = 0;}
                
            _trajectories[i] = single_traj;
        }
        // FITNESS: constant because we're interested in exploration
        this->_value = -1;
    }
    
    // generate trajectories
    void generate_traj(Eigen::VectorXf &traj, double angle, double dist_per_frame)
    {
        assert (dist_per_frame < Params::sim::ROOM_H);
        assert (dist_per_frame < Params::sim::ROOM_W);

        float current_x = Params::sim::start_x;
        float current_y = Params::sim::start_y;
        float current_angle = angle;

        // put starting position as first observation
        traj(0) = Params::sim::start_x;
        traj(1) = Params::sim::start_y;

        // start at 1 since start point is the first frame
        for (int i {1}; i < Params::sim::trajectory_length; ++i)
        {
            float x_delta = dist_per_frame * cos(current_angle);
            float y_delta = dist_per_frame * sin(current_angle);
            
            // save these for impact point calculations
            float previous_x = current_x;
            float previous_y = current_y;
            float previous_angle = current_angle;
            bool impact{false};
            
            current_x += x_delta;
            current_y += y_delta;

            if (current_x < 0)
            {
                impact = true;
                current_x *= -1;
                // coming from below
                if (current_angle < M_PI) {current_angle = (M_PI - current_angle);}
                else {current_angle = (3 * M_PI - current_angle);}
            }
            if (current_y < 0)
            {
                impact = true;
                current_y *= -1;
                current_angle = (2 * M_PI - current_angle);
            }
            if (current_x > Params::sim::ROOM_W)
            {
                impact = true;
                current_x = 2 * Params::sim::ROOM_W - current_x;
                if (current_angle < M_PI) {current_angle = (M_PI - current_angle);}
                else {current_angle = (3 * M_PI - current_angle);}
            }
            if (current_y > Params::sim::ROOM_H)
            {
                impact = true;
                current_y = 2 * Params::sim::ROOM_H - current_y;
                current_angle = (2 * M_PI - current_angle);
            }
            traj(i * 2) = current_x;
            traj(i * 2 + 1) = current_y;
        }
    }

    // generates trajectories and the impact points
    void generate_traj(Eigen::VectorXf &traj, std::vector<float> &wall_impacts, double angle, double dist_per_frame)
    {
        wall_impacts.clear();

        assert (dist_per_frame < Params::sim::ROOM_H);
        assert (dist_per_frame < Params::sim::ROOM_W);

        float current_x = Params::sim::start_x;
        float current_y = Params::sim::start_y;
        float current_angle = angle;

        // put starting position as first observation
        traj(0) = Params::sim::start_x;
        traj(1) = Params::sim::start_y;

        // start at 1 since start point is the first frame
        for (int i {1}; i < Params::sim::trajectory_length; ++i)
        {
            float x_delta = dist_per_frame * cos(current_angle);
            float y_delta = dist_per_frame * sin(current_angle);
            
            // save these for impact point calculations
            float previous_x = current_x;
            float previous_y = current_y;
            float previous_angle = current_angle;
            bool impact{false};
            
            current_x += x_delta;
            current_y += y_delta;

            if (current_x < 0)
            {
                impact = true;
                current_x *= -1;
                // coming from below
                if (current_angle < M_PI) {current_angle = (M_PI - current_angle);}
                else {current_angle = (3 * M_PI - current_angle);}
            }
            if (current_y < 0)
            {
                impact = true;
                current_y *= -1;
                current_angle = (2 * M_PI - current_angle);
            }
            if (current_x > Params::sim::ROOM_W)
            {
                impact = true;
                current_x = 2 * Params::sim::ROOM_W - current_x;
                if (current_angle < M_PI) {current_angle = (M_PI - current_angle);}
                else {current_angle = (3 * M_PI - current_angle);}
            }
            if (current_y > Params::sim::ROOM_H)
            {
                impact = true;
                current_y = 2 * Params::sim::ROOM_H - current_y;
                current_angle = (2 * M_PI - current_angle);
            }
            traj(i * 2) = current_x;
            traj(i * 2 + 1) = current_y;
            
            if (impact)
            {generate_impact_points(wall_impacts, previous_x, previous_y, previous_angle, x_delta, y_delta, dist_per_frame);}
        }
    }

    void generate_impact_points(std::vector<float> &wall_impacts, float previous_x, float previous_y, float previous_angle, 
                            float x_delta, float y_delta, float dist_per_frame)
    {
        float epsilon = 1e-5;

        float projected_x = previous_x + x_delta;
        float projected_y = previous_y + y_delta;

        bool double_impact{false};
        // check if there will be two impacts
        if (((projected_x > Params::sim::ROOM_W - epsilon) && (projected_y > Params::sim::ROOM_H - epsilon)) || ((projected_x < 0 + epsilon) && (projected_y < 0 + epsilon)) ||
            ((projected_x > Params::sim::ROOM_W - epsilon) && (projected_y < 0 + epsilon)) || ((projected_x < 0 + epsilon) && (projected_y > Params::sim::ROOM_H - epsilon)))
        {double_impact = true;}

        float impact_1_x, impact_1_y, impact_2_x, impact_2_y, new_angle;

        if (previous_x + x_delta > Params::sim::ROOM_W - epsilon)
        {
            impact_1_x = Params::sim::ROOM_W;
            impact_1_y = previous_y + tan(previous_angle) * (Params::sim::ROOM_W - previous_x);
            if (double_impact)
            // one of the two conditions below must hold if there is a double impact
            {
                bool reorder{false};
                if (impact_1_y > Params::sim::ROOM_H - epsilon)
                // ball hits the upper y axis limit first, so recalculate
                {
                    reorder = true;
                    impact_1_x = previous_x + (Params::sim::ROOM_H - previous_y) / tan(previous_angle);
                    impact_1_y = Params::sim::ROOM_H;
                }
                else if (impact_1_y < 0 + epsilon)
                // ball hits the lower y axis limit first
                {
                    reorder = true;
                    impact_1_x = previous_x - previous_y / tan(previous_angle);
                    impact_1_y = 0;
                }
                else
                // no reordering required
                {
                    impact_2_y = Params::sim::ROOM_H;
                    if (previous_angle < M_PI)
                    {
                        new_angle = (M_PI - previous_angle);
                        impact_2_x = Params::sim::ROOM_W + (Params::sim::ROOM_H - impact_1_y) / tan(new_angle);
                    }
                    else 
                    {
                        new_angle = (3 * M_PI - previous_angle);
                        impact_2_x = Params::sim::ROOM_W - impact_1_y / tan(new_angle);
                    }
                }
                if (reorder)
                // calculate second impact point after reordering
                {
                    new_angle = (2 * M_PI - previous_angle);
                    impact_2_x = Params::sim::ROOM_W;
                    impact_2_y = impact_1_y + tan(new_angle) * (Params::sim::ROOM_W - impact_1_x);
                }
            }
        }
        else if (previous_x + x_delta < 0 + epsilon)
        {
            impact_1_x = 0;
            impact_1_y = previous_y - tan(previous_angle) * previous_x;
            if (double_impact)
            {
                bool reorder{false};
                if (impact_1_y > Params::sim::ROOM_H - epsilon)
                {
                    reorder = true;
                    impact_1_x = previous_x + (Params::sim::ROOM_H - previous_y) / tan(previous_angle);
                    impact_1_y = Params::sim::ROOM_H;
                }
                else if (impact_1_y < 0 + epsilon)
                {
                    reorder = true;
                    impact_1_x = previous_x - previous_y / tan(previous_angle);
                    impact_1_y = 0;
                }
                else
                {
                    impact_2_y = 0;
                    if (previous_angle < M_PI) 
                    {
                        new_angle = (M_PI - previous_angle);
                        impact_2_x = (Params::sim::ROOM_H - impact_1_y) / tan(new_angle);
                    }
                    else 
                    {
                        new_angle = (3 * M_PI - previous_angle);
                        impact_2_x = impact_1_y / -tan(new_angle);
                    }
                }
                if (reorder)
                {
                    new_angle = (2 * M_PI - previous_angle);
                    impact_2_x = 0;
                    impact_2_y = impact_1_y + tan(new_angle) * -impact_1_x;
                }
            }
        }
        // if there is no double impact and the x limits are not exceeded, then the impact is on y    
        if ((previous_y + y_delta > Params::sim::ROOM_H - epsilon) && !double_impact)
        {
            impact_1_x = previous_x + (Params::sim::ROOM_H - previous_y) / tan(previous_angle);
            impact_1_y = Params::sim::ROOM_H;
        }
        if ((previous_y + y_delta < 0 + epsilon) && !double_impact)
        {
            impact_1_x = previous_x - previous_y / tan(previous_angle);
            impact_1_y = 0;
        }

        wall_impacts.push_back(impact_1_x);
        wall_impacts.push_back(impact_1_y);

        // this will give two impact if the points are the same, e.g. 10,10 can add a norm comparison to make it only be one
        if (double_impact)
        {
            wall_impacts.push_back(impact_2_x);
            wall_impacts.push_back(impact_2_y);
        }

        assert(impact_1_x < Params::sim::ROOM_W + 1);
        assert(impact_1_y < Params::sim::ROOM_H + 1);
        assert(impact_1_x > -1);
        assert(impact_1_y > -1);
        if (double_impact)
        {
            assert(impact_2_x < Params::sim::ROOM_W + 1);
            assert(impact_2_y < Params::sim::ROOM_H + 1);
            assert(impact_2_x > -1);
            assert(impact_2_y > -1);
        }
    }


    template<typename block_t>
    void get_flat_observations(block_t &data) const 
    {
        for (size_t row {0}; row < (Params::random::max_num_random + 1); ++row)
        {   
            for (size_t i{0}; i < Params::sim::num_trajectory_elements; ++i)
                {data(row, i) = _trajectories[row](i);}
        }
    }

    float &entropy() { return _m_entropy; }

    size_t &num_trajectories() { return _m_num_trajectories; }    

    bool is_random(int index)
    {return _is_random_trajectories.at(index);}

    int calculate_diversity_bins(std::bitset<Params::nov::discretisation * Params::nov::discretisation> &crossed_buckets)
    {
        Eigen::VectorXf impact_pts = Eigen::Map<Eigen::VectorXf> (_wall_impacts.data(), _wall_impacts.size());
        Eigen::VectorXf traj = _trajectories[0];
        Eigen::Vector2f start = traj.head<2>();
        Eigen::Vector2f end = traj.tail<2>();
        Eigen::Vector2f impact_pt;
        Eigen::Vector2f slope;
        Eigen::Vector2f current;

        double discrete_length_x {Params::sim::ROOM_W / Params::nov::discretisation};
        double discrete_length_y {Params::sim::ROOM_H / Params::nov::discretisation};

        // how many sub steps do we want to make
        double factor_divider {10};

        double factor;
        double max_factor;

        for (int i{0}; i < impact_pts.size() + 2; i += 2)
        {
            // not a nice way, includes the end point of the trajectory into the for loop with the above +2
            if (i >= impact_pts.size()) 
            {impact_pt = end;}
            else
            {impact_pt = impact_pts.segment(i, 2);}
            
            slope = impact_pt - start;
            slope.normalize();

            // create factor such that we move from bucket to bucket on one axis
            // pick the axis where there is most change to restrict
            // max factor calculation so we know when to stop
            if (abs(slope(0)) > abs(slope(1)))
            {
                factor = abs((discrete_length_x / factor_divider) / slope(0));
                max_factor = abs((impact_pt(0) - start(0)) / slope(0));
            }
            else
            {
                factor = abs((discrete_length_y / factor_divider) / slope(1));
                max_factor = abs((impact_pt(1) - start(1)) / slope(1));
            }

            for (int j{0}; j * factor < max_factor; ++j)
            // the end point of one line is included as the start point in the next line
            {
                current = start + j * factor * slope;
                
                int bucket_x = current(0) / discrete_length_x;
                int bucket_y = current(1) / discrete_length_y;

                // at the edge of the image, the bucket would be for the box outside the image, so subtract 1 to keep it inside
                if (bucket_x == Params::nov::discretisation) {bucket_x -= 1;}
                if (bucket_y == Params::nov::discretisation) {bucket_y -= 1;}
                int bucket_number = bucket_y * Params::nov::discretisation + bucket_x;
                crossed_buckets.set(bucket_number);
            }
            start = impact_pt;
        }

        // end point of trajectory needs to be added
        int bucket_x = end(0) / discrete_length_x;
        int bucket_y = end(1) / discrete_length_y;
        if (bucket_x == Params::nov::discretisation) {bucket_x -= 1;}
        if (bucket_y == Params::nov::discretisation) {bucket_y -= 1;}
        int bucket_number = bucket_y * Params::nov::discretisation + bucket_x;
        crossed_buckets.set(bucket_number);
        return bucket_number;
    }

    private:
    // random trajectories + 1 real one
    std::array<Eigen::VectorXf, Params::random::max_num_random + 1> _trajectories;
    std::array<int, Params::random::max_num_random + 1> _is_random_trajectories {1};
    size_t _m_num_trajectories;
    std::vector<float> _wall_impacts;
    float _m_entropy;

    // for debugging
    float _angle;
    float _dpf;
};

#endif //TRAJECTORY_HPP
