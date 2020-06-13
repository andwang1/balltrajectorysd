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

namespace rng {
    // Will be used to obtain a seed for the random number engine
    std::random_device rd;  
    std::mt19937 gen(rd()); 
    std::uniform_real_distribution<> rng(0, 1);
}


class Arm : public robox2d::Robot {
public:
  
  Arm(std::shared_ptr<b2World> world, std::array<int, Params::random::max_num_random + 1> &is_random_traj){

    size_t nb_joints=Params::qd::gen_dim;
    float arm_length=1.5f;
    float seg_length = arm_length / (float) nb_joints;

    // create walls one by one as GUI needs seperate bodies, so cannot use one body for multiple fixtures (TODO create diff variant for non GUI run?)
    // specify width and height to each side from a central location
    if (Params::sim::enable_graphics)
    {
    b2Body* ceiling = robox2d::common::createBox(world, {Params::sim::ROOM_W / 2, 0.01}, b2_staticBody, {Params::sim::ROOM_W / 2, Params::sim::ROOM_H, 0.f});
    b2Body* floor = robox2d::common::createBox(world, {Params::sim::ROOM_W / 2, 0.01}, b2_staticBody, {Params::sim::ROOM_W / 2, 0.f, 0.f});
    b2Body* right = robox2d::common::createBox(world, {0.01, Params::sim::ROOM_H / 2}, b2_staticBody, {Params::sim::ROOM_W, Params::sim::ROOM_H / 2, 0.f});
    b2Body* left = robox2d::common::createBox(world, {0.01, Params::sim::ROOM_H / 2}, b2_staticBody, {0, Params::sim::ROOM_H / 2, 0.f});
    }
    else
    {b2Body* room = robox2d::common::createRoom(world, {Params::sim::ROOM_W, Params::sim::ROOM_H});}

    // base in the center of the room
    b2Body* body = robox2d::common::createBox( world,{arm_length*0.025f, arm_length*0.025f}, b2_staticBody,  {Params::sim::ROOM_W / 2, Params::sim::ROOM_H / 2,0.f} );
    b2Vec2 anchor = body->GetWorldCenter();
    // body will always represent the body created in the previous iteration
    
    for(size_t i =0; i < nb_joints; i++)
    {
      float density = 1.0f/std::pow(1.5,i);
	    _end_effector = robox2d::common::createBox( world,{seg_length*0.5f , arm_length*0.01f }, b2_dynamicBody, {(0.5f+i)*seg_length + Params::sim::ROOM_W / 2, Params::sim::ROOM_H / 2,0.0f}, density );
      this->_servos.push_back(std::make_shared<robox2d::common::Servo>(world,body, _end_effector, anchor));

      body=_end_effector;
      anchor = _end_effector->GetWorldCenter() + b2Vec2(seg_length*0.5 , 0.0f);
    }

    // in this order so that first observation retrieved in simu.run is the actual trajectory
    // start at 1 because first one is the actual ball
    for (int i{1}; i < is_random_traj.size(); ++i)
    {
        if (is_random_traj[i])
        {
            // use rng to generate position and force
            float pos_x = rng::rng(rng::gen) * (Params::sim::start_x - 2 * Params::sim::radius) + Params::sim::radius;
            float pos_y = rng::rng(rng::gen) * (Params::sim::start_y - 2 * Params::sim::radius) + Params::sim::radius;
            float force_x = rng::rng(rng::gen) * Params::sim::max_force;
            float force_y = rng::rng(rng::gen) * Params::sim::max_force;
            
            b2Body* random_ball = robox2d::common::createCircle( world, Params::sim::radius, b2_dynamicBody,  {pos_x, pos_y, 0.f}, 0.2f);
            b2Vec2 force{force_x, force_y};
            random_ball->ApplyForce(force, random_ball->GetWorldCenter(), true);
        }
    }
    b2Body* ball = robox2d::common::createCircle( world, Params::sim::radius, b2_dynamicBody,  {Params::sim::start_x, Params::sim::start_y, 0.f}, 0.2f );
  }
  
  b2Vec2 get_end_effector_pos(){return _end_effector->GetWorldCenter(); }
  
private:
  b2Body* _end_effector;
};

FIT_QD(Trajectory)
{
    public:
    Trajectory(){
        _params.resize(Params::qd::gen_dim);

        for (Eigen::VectorXf &traj : trajectories)
            {traj.resize(Params::sim::num_trajectory_elements);}

        std::fill(is_random_trajectories.begin(), is_random_trajectories.end(), 0);

        // *100 timesteps in simulation, *2 two coordinates for each timestep
        full_trajectory.resize(static_cast<int>(Params::sim::sim_duration * 100 * 2));
    }

    template <typename Indiv> 
    void eval(Indiv & ind){

        for (size_t i = 0; i < ind.size(); ++i)
        _params[i] = ind.data(i);
        // track number of random trajectories
        m_num_trajectories = 0;

        // first trajectory is actual trajectory
        is_random_trajectories[0] = 1;

        // generate random trajectories
        // pass in boolean vector to the simulate function
        for (int i{1}; i < Params::random::max_num_random + 1; ++i)
        {
            float prob = rng::rng(rng::gen);
            if (prob <= Params::random::pct_random)
            {
                // 1 means it is a trajectory
                is_random_trajectories[i] = 1;
                ++m_num_trajectories;
            }
            else 
            {
                is_random_trajectories[i] = 0;
            }
        }
        // populates member vars
        simulate(_params, is_random_trajectories);

        // FITNESS: constant because we're interested in exploration
        this->_value = -1;
    }
    
    // generate trajectories during the algorithm
    void simulate(Eigen::VectorXd &ctrl_pos, std::array<int, Params::random::max_num_random + 1> &is_random_traj){
        robox2d::Simu simu;
        simu.add_floor();
        
        auto rob = std::make_shared<Arm>(simu.world(), is_random_traj);
        auto ctrl = std::make_shared<robox2d::control::ConstantPos>(ctrl_pos);
        rob->add_controller(ctrl);
        simu.add_robot(rob);

        if (Params::sim::enable_graphics)
        {
            auto graphics = std::make_shared<robox2d::gui::Graphics<>>(simu.world());
            simu.set_graphics(graphics);
        }
        simu.run(Params::sim::sim_duration, trajectories, Params::sim::trajectory_length);
    }

    // generate full trajectory for diversity calc
    void simulate(Eigen::VectorXd &ctrl_pos){
    // simulating
        robox2d::Simu simu;
        simu.add_floor();

        // no random trajectories
        std::array<int, Params::random::max_num_random + 1> is_random_traj;
        std::fill(is_random_traj.begin(), is_random_traj.end(), 0);
        
        auto rob = std::make_shared<Arm>(simu.world(), is_random_traj);
        auto ctrl = std::make_shared<robox2d::control::ConstantPos>(ctrl_pos);
        rob->add_controller(ctrl);
        simu.add_robot(rob);

        if (Params::sim::enable_graphics)
        {
            auto graphics = std::make_shared<robox2d::gui::Graphics<>>(simu.world());
            simu.set_graphics(graphics);
        }
    
        // get full trajectory
        simu.run(Params::sim::sim_duration, full_trajectory);
    }
    
    template<typename block_t>
    void get_flat_observations(block_t &data) const 
    {
        for (size_t row {0}; row < (Params::random::max_num_random + 1); ++row)
        {   
            for (size_t i{0}; i < Params::sim::num_trajectory_elements; ++i)
                {data(row, i) = trajectories[row](i);}
        }
    }

    float &entropy() { return m_entropy; }

    size_t &num_trajectories() { return m_num_trajectories; }    

    bool is_random(int index)
    {
        return is_random_trajectories.at(index);
    }
    
    int calculate_diversity_bins(std::bitset<Params::nov::discretisation * Params::nov::discretisation> &crossed_buckets)
    {
        // generate trajectory again but without additional random balls
        simulate(_params);


        double ROOM_H = Params::sim::ROOM_H;
        double ROOM_W = Params::sim::ROOM_W;
        double discretisation = Params::nov::discretisation;

        double discrete_length_x {ROOM_W / discretisation};
        double discrete_length_y {ROOM_H / discretisation};

        int bucket_number{-1};

        for (int j{0}; j < full_trajectory.size(); j+=2)
        {
            int bucket_x = full_trajectory[j] / discrete_length_x;
            int bucket_y = full_trajectory[j+1] / discrete_length_y;
            bucket_number = bucket_y * discretisation + bucket_x;
            
            if (VERBOSE)
            {
                std::cout << "Bx " << bucket_x << "By " << bucket_y << "Bnum " << bucket_number << std::endl;
            }
            crossed_buckets.set(bucket_number);
        }

        if (VERBOSE)
        {
            std::cout << "BITS PRINTED IN REV ORDER" << std::endl;
            std::cout << crossed_buckets;
        }
        return bucket_number;
    }

    // // generates images from the trajectories fed into the function
    // void generate_image(std::array<Eigen::Matrix<double, discretisation, discretisation>, trajectory_length> &image_frames,
    //                     const std::vector<Eigen::VectorXd> &trajectories)
    // {
    //     double discrete_length_x {double(ROOM_W) / discretisation};
    //     double discrete_length_y {double(ROOM_H) / discretisation};

    //     for (int i {0}; i < 2 * trajectory_length; i += 2)
    //     {
    //         // initialise image
    //         int image_num {i / 2};
    //         image_frames[image_num] = Eigen::Matrix<double, discretisation, discretisation>::Zero();
    //         for (auto &traj : trajectories)
    //         {
    //             double x = traj(i);
    //             double y = traj(i + 1);

    //             // flip rows and columns so x = horizontal axis
    //             int index_y = x / discrete_length_x;
    //             int index_x = y / discrete_length_y;

    //             image_frames[image_num](index_x, index_y) = 1;
    //         }
    //         if (VERBOSE)
    //         std::cout << image_frames[image_num] << std::endl;
    //     }
    // }

    private:
    // using matrix directly does not work, see above comment at generate_traj, will not stay in mem after assigining
    // Eigen::Matrix<double, Params::random::max_num_random + 1, Params::sim::trajectory_length> trajectories;
    
    // random trajectories + 1 real one
    std::array<Eigen::VectorXf, Params::random::max_num_random + 1> trajectories;
    Eigen::VectorXf full_trajectory;
    std::array<int, Params::random::max_num_random + 1> is_random_trajectories;
    size_t m_num_trajectories;
    Eigen::VectorXd _params;
    float m_entropy;
};

#endif //TRAJECTORY_HPP
