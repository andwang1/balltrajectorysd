//
// Created by Luca Grillotti on 04/12/2019.
//

#ifndef SFERES2_FIT_MAZE_HPP
#define SFERES2_FIT_MAZE_HPP


#include <SDL/SDL.h>
#include <SDL/SDL_thread.h>

#include <tbb/mutex.h>

#include <modules/libfastsim/src/display.hpp>
#include <modules/libfastsim/src/map.hpp>
#include <modules/libfastsim/src/settings.hpp>


namespace global {
    boost::shared_ptr<fastsim::Settings> settings;
    boost::shared_ptr<fastsim::Map> map;
    boost::shared_ptr<fastsim::DisplaySurface> display;
    tbb::mutex sdl_mutex;
}

void init_fastsim_settings() {

    global::settings = boost::make_shared<fastsim::Settings>(
            "/git/sferes2/exp/example-pytorch-sferes/resources/LS_maze_hard.xml");

    global::map = global::settings->map();
    global::display = boost::make_shared<fastsim::DisplaySurface>(global::map, *(global::settings->robot()));
}

FIT_QD(HardMaze)
{
public:
    HardMaze() {}

    typedef std::vector<std::vector<std::vector<uint8_t>>> image_t

    template<typename Indiv>
    void eval(Indiv &ind) {

        boost::shared_ptr<fastsim::Robot> robot = boost::make_shared<fastsim::Robot>(
                *global::settings->robot());

        constexpr size_t c_size_input = 5;
        constexpr size_t c_number_iterations = 2000;
        constexpr size_t c_period_add_successive_sensor_measure = 200;
        constexpr size_t c_period_add_successive_action_measure = 50;

        std::vector<float> in(c_size_input);

        for (size_t i = 0; i < c_number_iterations; ++i) {
            // NN indiv policies stay the same across different environments
            for (size_t index_laser = 0; index_laser < 3; ++index_laser) {
                in[index_laser] = robot->get_lasers()[index_laser].get_dist();
            }

            in[3] = static_cast<float>(robot->get_left_bumper());
            in[4] = static_cast<float>(robot->get_right_bumper());

            ind.nn().step(in);
            robot->move(ind.nn().get_outf(0), ind.nn().get_outf(1), global::map);
        }
        const int c_size(global::display->screen()->w * global::display->screen()->h * 4);
        uint8_t array_pixels[c_size];
        global::sdl_mutex.lock();

        global::display->update(*robot);
        //fastsim::DisplaySurface d(map, *robot);
        //d.update();

        memcpy(array_pixels, global::display->screen()->pixels, c_size);
        global::sdl_mutex.unlock();

        for (const auto &laser : robot->get_lasers()) {
            m_lasers_dists.push_back(laser.get_dist());
        }

        this->_update_gt(robot);

        this->_create_image(array_pixels, global::display->screen()->h, global::display->screen()->w);

        this->_value = -1; // FITNESS: constant because we're interested in exploration

    }

    std::vector<float> &gt() { return m_gt; }

    const image_t &get_rgb_image() const { return m_rgb_image; }

    float &entropy() { return m_entropy; }

    const std::vector<float> &observations() const {
        return m_image_float;
    }

    const std::vector<float> &lasers_dists() const {
        return m_lasers_dists;
    }

    size_t get_flat_obs_size() const {
        return observations().size();
    }

    template<typename block_t>
    void get_flat_observations(block_t &data) const {
        // std::cout << _image.size() << std::endl;
        for (size_t i = 0; i < m_image_float.size(); i++) {
            data(0, i) = m_image_float[i];
        }
    }

protected:
    float m_entropy;
    std::vector<float> m_image_float;
    image_t m_rgb_image;
    std::vector<float> m_gt;
    std::vector<float> m_lasers_dists;

    void _update_gt(const boost::shared_ptr<fastsim::Robot> &robot) {
        constexpr size_t c_size_gt = 3;
        m_gt.clear();
        m_gt.reserve(c_size_gt);
        m_gt.push_back(robot->get_pos().get_x());
        m_gt.push_back(robot->get_pos().get_y());
        m_gt.push_back(robot->get_pos().theta());
    }

    void _create_image(uint8_t array_pixels[], const int w, const int h) {
        _store_image(array_pixels, w, h);

        if (Params::use_colors) {
            _store_rgb_vector();
        } else {
            _store_grayscale_vector();
        }
    }

    void _store_image(uint8_t array_pixels[], const int w, const int h) {
        m_rgb_image.clear();
        for (int index_row = 0; index_row < h; ++index_row) {
            std::vector<std::vector<uint8_t>> v_row;
            for (int index_col = 0; index_col < w; ++index_col) {
                std::vector<uint8_t> v_pixel;
                v_pixel.assign(&array_pixels[4 * (index_row + index_col * w)], &array_pixels[4 * (index_row + index_col * w)] + 3);
                v_row.push_back(v_pixel);
            }
            m_rgb_image.push_back(v_row);
        }
    }

    void _store_rgb_vector() {
        std::vector<float> img_red_float;
        std::vector<float> img_green_float;
        std::vector<float> img_blue_float;

        std::vector<uint8_t> img_red_uint8;
        std::vector<uint8_t> img_green_uint8;
        std::vector<uint8_t> img_blue_uint8;

        for (const std::vector<std::vector<uint8_t>> &row: m_rgb_image) {
            for (const std::vector<uint8_t> &rgb_color: row) {
                img_red_uint8.push_back(rgb_color[0]);
                img_green_uint8.push_back(rgb_color[1]);
                img_blue_uint8.push_back(rgb_color[2]);
            }
        }

        for (size_t i = 0; i < img_red_uint8.size(); ++i) {
            img_red_float.push_back(img_red_uint8[i] / 255.0);
            img_green_float.push_back(img_green_uint8[i] / 255.0);
            img_blue_float.push_back(img_blue_uint8[i] / 255.0);
        }

        for (int k = Params::times_downsample; k > 1; k /= 2) {
            _downsample(img_red_float, k);
            _downsample(img_green_float, k);
            _downsample(img_blue_float, k);
        }

        m_image_float.insert(m_image_float.end(), img_red_float.begin(), img_red_float.end());
        m_image_float.insert(m_image_float.end(), img_green_float.begin(), img_green_float.end());
        m_image_float.insert(m_image_float.end(), img_blue_float.begin(), img_blue_float.end());
    }

    void _store_grayscale_vector() {
        std::vector<uint8_t> img_uint8;

        for (const std::vector<std::vector<uint8_t>> &row: m_rgb_image) {
            for (const std::vector<uint8_t> &rgb_color: row) {
                img_uint8.insert(img_uint8.end(), rgb_color.begin(), rgb_color.end());
            }
        }

        float pixel;

        for (size_t i = 0; i < img_uint8.size(); i += 3) {
            pixel = R * (float) img_uint8[i] + G * (float) img_uint8[i + 1] + B * (float) img_uint8[i + 2];
            m_image_float.push_back(pixel / 255.0);
        }

        for (int k = Params::times_downsample; k > 1; k /= 2) {
            _downsample(m_image_float, k);
        }
    }

    void _downsample(std::vector<float> &img, int k) const {
        // std::cout << "downsample: img size " << img.size() << "k: " << k << std::endl;
        int image_height = Params::image_height * k;
        int image_width = Params::image_width * k;

        Mat image;
        image.resize(image_height, image_width);

        // std::cout << "image in matrix form resized: " << image.rows() << "x" << image.cols() << std::endl;

        size_t count = 0;
        for (int i = 0; i < image_height; i++) {
            for (int j = 0; j < image_width; j++) {
                image(i, j) = img[count++];
            }
        }

        img.clear();

        for (int i = 0; i < image_height; i += 2) {
            for (int j = 0; j < image_width; j += 2) {
                // Using Average Pooling to reduce dimensionality
//                        img.push_back(std::max({image(i, j), image(i + 1, j), image(i, j + 1), image(i + 1, j + 1)}));
                img.push_back((image(i, j) + image(i + 1, j) + image(i, j + 1) + image(i + 1, j + 1)) / 4);
            }
        }
    }
};

#endif //SFERES2_FIT_MAZE_HPP
