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

#include <box2d/box2d.h>
#include <robox2d/simu.hpp>
#include <robox2d/robot.hpp>
#include <robox2d/common.hpp>
#include <robox2d/gui/magnum/graphics.hpp>

// print statements for debugging
bool VERBOSE = false;

// pseudo random number generator
namespace rng {
    // Will be used to obtain a seed for the random number engine
    std::random_device rd;  
    std::mt19937 gen(rd()); 
    std::uniform_real_distribution<> rng(0, 1);
}

class Arm : public robox2d::Robot {
public:
  
    Arm(std::shared_ptr<b2World> world, std::array<int, Params::random::max_num_random + 1> &is_traj)
    {
        size_t nb_joints = Params::qd::gen_dim;
        float arm_length = 1.5f;
        float seg_length = arm_length / (float) nb_joints;

        // create walls one by one as GUI needs seperate bodies, so cannot use one body for multiple fixtures
        // specify width and height to each side from a central location
        if (Params::sim::enable_graphics)
        {
            b2Body* ceiling = robox2d::common::createBox(world, {Params::sim::ROOM_W / 2, 0.01}, b2_staticBody, {Params::sim::ROOM_W / 2, Params::sim::ROOM_H, 0.f});
            b2Body* floor = robox2d::common::createBox(world, {Params::sim::ROOM_W / 2, 0.01}, b2_staticBody, {Params::sim::ROOM_W / 2, 0.f, 0.f});
            b2Body* right = robox2d::common::createBox(world, {0.01, Params::sim::ROOM_H / 2}, b2_staticBody, {Params::sim::ROOM_W, Params::sim::ROOM_H / 2, 0.f});
            b2Body* left = robox2d::common::createBox(world, {0.01, Params::sim::ROOM_H / 2}, b2_staticBody, {0, Params::sim::ROOM_H / 2, 0.f});
        }
        else // if not using the GUI, create one body with 4 walls for faster sim
            {b2Body* room = robox2d::common::createRoom(world, {Params::sim::ROOM_W, Params::sim::ROOM_H});}

        // base in the center of the room
        b2Body* body = robox2d::common::createBox(world, {arm_length*0.025f, arm_length*0.025f}, b2_staticBody, {Params::sim::ROOM_W / 2, Params::sim::ROOM_H / 2,0.f});
        b2Vec2 anchor = body->GetWorldCenter();
        
        for(size_t i{0}; i < nb_joints; ++i)
        {
            float density = 1.0f/std::pow(1.5,i);
            _end_effector = robox2d::common::createBox(world, {seg_length*0.5f, arm_length*0.01f}, b2_dynamicBody, {(0.5f+i)*seg_length + Params::sim::ROOM_W / 2, Params::sim::ROOM_H / 2,0.0f}, density);
            this->_servos.push_back(std::make_shared<robox2d::common::Servo>(world,body, _end_effector, anchor));

            body = _end_effector;
            anchor = _end_effector->GetWorldCenter() + b2Vec2(seg_length*0.5, 0.0f);
        }

        // add random balls first so that first observation retrieved in simu.run is the actual trajectory
        // start at 1 because index 0 is the actual ball
        for (int i{1}; i < is_traj.size(); ++i)
        {
            if (is_traj[i])
            {
                // use rng to generate position and force
                float pos_x = rng::rng(rng::gen) * (Params::sim::start_x - 2 * Params::sim::radius) + Params::sim::radius;
                float pos_y = rng::rng(rng::gen) * (Params::sim::start_y - 2 * Params::sim::radius) + Params::sim::radius;
                float force_x = rng::rng(rng::gen) * Params::sim::max_force;
                float force_y = rng::rng(rng::gen) * Params::sim::max_force;
                
                b2Body* random_ball = robox2d::common::createCircle(world, Params::sim::radius, b2_dynamicBody, {pos_x, pos_y, 0.f}, 0.2f);
                b2Vec2 force{force_x, force_y};
                random_ball->ApplyForce(force, random_ball->GetWorldCenter(), true);
            }
        }
        b2Body* ball = robox2d::common::createCircle(world, Params::sim::radius, b2_dynamicBody, {Params::sim::start_x, Params::sim::start_y, 0.f}, 0.2f);
    }
  
b2Vec2 get_end_effector_pos(){return _end_effector->GetWorldCenter();}
  
private:
b2Body* _end_effector;
};

FIT_QD(Trajectory)
{
    public:
    Trajectory(): _params(Params::qd::gen_dim), _full_trajectory(Params::sim::full_trajectory_length){
        for (Eigen::VectorXf &traj : _trajectories)
            {traj.resize(Params::sim::num_trajectory_elements);}
    
        for (Eigen::VectorXf &traj : _undisturbed_trajectories)
            {traj.resize(Params::sim::num_trajectory_elements);}

        std::fill(_is_trajectory.begin(), _is_trajectory.end(), 0);
    }

    template <typename Indiv> 
    void eval(Indiv & ind){

        for (size_t i = 0; i < ind.size(); ++i)
            _params[i] = ind.data(i);

        // track number of random trajectories
        _m_num_trajectories = 0;

        // first trajectory is actual trajectory, 1 = is trajectory
        _is_trajectory[0] = 1;

        // generate random trajectories
        for (int i{1}; i < Params::random::max_num_random + 1; ++i)
        {
            float prob = rng::rng(rng::gen);
            if (prob <= Params::random::pct_random)
            {
                // 1 means it is a trajectory
                _is_trajectory[i] = 1;
                ++_m_num_trajectories;
            }
            else 
                {_is_trajectory[i] = 0;}
        }
        // create trajectories
        simulate(_params, _is_trajectory);

        // for diversity and loss tracking generate only the real trajectory without any randomness 
        if (_m_num_trajectories > 0)
            {simulate(_params);}
        else // if no other balls present in simulation already
            {_undisturbed_trajectories[0] = _trajectories[0];}

        // FITNESS: constant because we're interested in exploration
        this->_value = -1;
    }
    
    // generate trajectories during the algorithm
    void simulate(Eigen::VectorXd &ctrl_pos, std::array<int, Params::random::max_num_random + 1> &is_traj){
        robox2d::Simu simu;
        simu.add_floor();
        
        auto rob = std::make_shared<Arm>(simu.world(), is_traj);
        auto ctrl = std::make_shared<robox2d::control::ConstantPos>(ctrl_pos);
        rob->add_controller(ctrl);
        simu.add_robot(rob);

        if (Params::sim::enable_graphics)
        {
            auto graphics = std::make_shared<robox2d::gui::Graphics<>>(simu.world());
            simu.set_graphics(graphics);
        }
        simu.run(Params::sim::sim_duration, _trajectories, _full_trajectory, Params::sim::trajectory_length);
    }

    // generate full trajectory for diversity calc and loss tracking
    void simulate(Eigen::VectorXd &ctrl_pos){
    // simulating
        robox2d::Simu simu;
        simu.add_floor();

        // no random trajectories
        std::array<int, Params::random::max_num_random + 1> is_traj;
        std::fill(is_traj.begin(), is_traj.end(), 0);
        
        auto rob = std::make_shared<Arm>(simu.world(), is_traj);
        auto ctrl = std::make_shared<robox2d::control::ConstantPos>(ctrl_pos);
        rob->add_controller(ctrl);
        simu.add_robot(rob);

        // if (Params::sim::enable_graphics)
        // {
        //     auto graphics = std::make_shared<robox2d::gui::Graphics<>>(simu.world());
        //     simu.set_graphics(graphics);
        // }

        simu.run(Params::sim::sim_duration, _undisturbed_trajectories, _full_trajectory, Params::sim::trajectory_length);
    }
    
    int calculate_diversity_bins(std::bitset<Params::nov::discretisation * Params::nov::discretisation> &crossed_buckets) const
    {
        int bucket_number{0};

        for (int j{0}; j < _full_trajectory.size(); j += 2)
        {
            int bucket_x = _full_trajectory[j] / Params::nov::discrete_length_x;
            int bucket_y = _full_trajectory[j+1] / Params::nov::discrete_length_y;
            bucket_number = bucket_y * Params::nov::discretisation + bucket_x;
            
            if (VERBOSE)
                {std::cout << "Bx " << bucket_x << "By " << bucket_y << "Bnum " << bucket_number << std::endl;}
            crossed_buckets.set(bucket_number);
        }

        if (VERBOSE)
        {
            std::cout << "BITS PRINTED IN REV ORDER" << std::endl;
            std::cout << crossed_buckets;
        }
        return bucket_number;
    }

    int calculate_distance(float &distance, bool &moved)
    {
        Eigen::VectorXf manhattan_dist = _full_trajectory.segment<Params::sim::full_trajectory_length - 2>(0) - 
                                         _full_trajectory.segment<Params::sim::full_trajectory_length - 2>(2);

        distance = 0;
        for (int i{0}; i < manhattan_dist.size(); i+=2)
            {distance += manhattan_dist.segment<2>(i).norm();}

        _moved = distance > 1e-6;
        moved = _moved;
        return get_bucket_index(Params::nov::discrete_length_x, Params::nov::discrete_length_y, Params::nov::discretisation);
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

    Eigen::VectorXf get_undisturbed_trajectory() const
    {return _undisturbed_trajectories[0];}

    float &entropy() 
    {return _m_entropy;}

    size_t num_trajectories() const
    {return _m_num_trajectories;}

    bool is_random(int index) const
    {return _is_trajectory.at(index);}

    Eigen::VectorXd &params()
    {return _params;}

    bool moved() const
    {return _moved;}

    int get_bucket_index(double discrete_length_x, double discrete_length_y, int discretisation) const
    {
        int bucket_x = _full_trajectory[Params::sim::full_trajectory_length - 2] / discrete_length_x;
        int bucket_y = _full_trajectory[Params::sim::full_trajectory_length - 1] / discrete_length_y;
        return bucket_y * discretisation + bucket_x;
    }

    private:
    // using matrix directly does not work, see above comment at generate_traj, will not stay in mem after assigining
    // Eigen::Matrix<double, Params::random::max_num_random + 1, Params::sim::trajectory_length> trajectories;
    
    Eigen::VectorXd _params;
    // random trajectories + 1 real one
    std::array<Eigen::VectorXf, Params::random::max_num_random + 1> _trajectories;
    std::array<Eigen::VectorXf, Params::random::max_num_random + 1> _undisturbed_trajectories;
    Eigen::VectorXf _full_trajectory;
    std::array<int, Params::random::max_num_random + 1> _is_trajectory;
    size_t _m_num_trajectories;
    bool _moved;
    float _m_entropy;
};

#endif //TRAJECTORY_HPP
