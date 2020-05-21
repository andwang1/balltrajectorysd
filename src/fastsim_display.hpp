//
// Created by Luca Grillotti on 21/05/2020.
//

#ifndef EXAMPLE_PYTORCH_SFERES_FASTSIM_DISPLAY_HPP
#define EXAMPLE_PYTORCH_SFERES_FASTSIM_DISPLAY_HPP

#include <SDL/SDL.h>
#include <SDL/SDL_thread.h>

#include <tbb/mutex.h>

#include <modules/libfastsim/src/display.hpp>
#include <modules/libfastsim/src/map.hpp>
#include <modules/libfastsim/src/settings.hpp>

namespace fastsim {
    class DisplaySurface : public Display {
    public:
        DisplaySurface(const boost::shared_ptr<Map> &m, Robot &r) : Display(m, r) {};

        SDL_Surface *screen() const {
            return _screen;
        }

        void update(const Robot &robot) {
            _events();
            // convert to pixel
            unsigned x = _map->real_to_pixel(robot.get_pos().x());
            unsigned y = _map->real_to_pixel(robot.get_pos().y());
            unsigned r = _map->real_to_pixel(robot.get_radius());
            float theta = robot.get_pos().theta();

            // erase robot
            SDL_BlitSurface(_map_bmp, &_prev_bb, _screen, &_prev_bb);
            // erase all
            SDL_BlitSurface(_map_bmp, 0, _screen, 0);

            // lasers
//           _disp_lasers(robot);

            // goals
//           _disp_goals(robot);


            // light sensor
//           _disp_light_sensors(robot);

            // radars
//           _disp_radars(robot);

            // camera
//           _disp_camera(robot);

            // draw the circle again (robot)
            unsigned int col = robot.color();
            _disc(_screen, x, y, r, _color_from_id(_screen, col));
            _circle(_screen, x, y, r, 255, 0, 0);
            // direction
            Uint32 color = SDL_MapRGB(_screen->format, 0, 255, 0);
            _line(_screen, x, y, (int) (r * cosf(theta) + x), (int) (r * sinf(theta) + y), color);

            // bumpers
//           _disp_bumpers(robot);

            // illuminated switches
//           _disp_switches();


            SDL_Rect rect;
            _bb_to_sdl(robot.get_bb(), &rect);
            rect.x = std::max(0, std::min((int) rect.x, (int) _prev_bb.x));
            rect.y = std::max(0, std::min((int) rect.y, (int) _prev_bb.y));
            rect.w = std::max(rect.w, _prev_bb.w);
            rect.h = std::max(rect.h, _prev_bb.h);

            if (rect.x + rect.w > _w) rect.w = _w;
            if (rect.y + rect.h > _h) rect.h = _h;


        }

    protected:
        void _disp_bb(const Robot &robot) {
            unsigned x = _map->real_to_pixel(robot.get_bb().x);
            unsigned y = _map->real_to_pixel(robot.get_bb().y);
            unsigned w = _map->real_to_pixel(robot.get_bb().w);
            unsigned h = _map->real_to_pixel(robot.get_bb().h);

            assert(x >= 0);
            assert(y >= 0);
            assert(x + w < (unsigned) _screen->w);
            assert(y + h < (unsigned) _screen->h);
            _line(_screen, x, y, x + w, y, 0);
            _line(_screen, x + w, y, x + w, y + h, 0);
            _line(_screen, x + w, y + h, x, y + h, 0);
            _line(_screen, x, y + h, x, y, 0);
        }

        void _disp_goals(const Robot &robot) {
            for (size_t i = 0; i < _map->get_goals().size(); ++i) {
                const Goal &goal = _map->get_goals()[i];
                unsigned x = _map->real_to_pixel(goal.get_x());
                unsigned y = _map->real_to_pixel(goal.get_y());
                unsigned diam = _map->real_to_pixel(goal.get_diam());
                Uint8 r = 0, g = 0, b = 0;
                switch (goal.get_color()) {
                    case 0:
                        r = 255;
                        break;
                    case 1:
                        g = 255;
                        break;
                    case 2:
                        b = 255;
                        break;
                    default:
                        assert(0);
                }
                _circle(_screen, x, y, diam, r, g, b);
            }
        }

        void _disp_radars(const Robot &robot) {
            unsigned r = _map->real_to_pixel(robot.get_radius()) / 2;
            unsigned x = _map->real_to_pixel(robot.get_pos().x());
            unsigned y = _map->real_to_pixel(robot.get_pos().y());

            for (size_t i = 0; i < robot.get_radars().size(); ++i) {
                const Radar &radar = robot.get_radars()[i];
                if (radar.get_activated_slice() != -1) {
                    float a1 = robot.get_pos().theta() + radar.get_inc() * radar.get_activated_slice();
                    float a2 = robot.get_pos().theta() + radar.get_inc() * (radar.get_activated_slice() + 1);
                    _line(_screen,
                          cos(a1) * r + x, sin(a1) * r + y,
                          cos(a2) * r + x, sin(a2) * r + y,
                          0x0000FF);
                    assert(radar.get_color() < (int) _map->get_goals().size());
                    const Goal &g = _map->get_goals()[radar.get_color()];
                    unsigned gx = _map->real_to_pixel(g.get_x());
                    unsigned gy = _map->real_to_pixel(g.get_y());
                    _line(_screen, x, y, gx, gy, 0x0000FF);
                }

            }

        }

        void _disp_bumpers(const Robot &robot) {
            // convert to pixel
            unsigned x = _map->real_to_pixel(robot.get_pos().x());
            unsigned y = _map->real_to_pixel(robot.get_pos().y());
            unsigned r = _map->real_to_pixel(robot.get_radius());
            float theta = robot.get_pos().theta();
            Uint32 cb_left = SDL_MapRGB(_screen->format, robot.get_left_bumper() ? 255 : 0, 0, 0);
            Uint32 cb_right = SDL_MapRGB(_screen->format, robot.get_right_bumper() ? 255 : 0, 0, 0);
            _line(_screen,
                  (int) (r * cosf(theta + M_PI / 2.0f) + x),
                  (int) (r * sinf(theta + M_PI / 2.0f) + y),
                  (int) (r * cosf(theta) + x),
                  (int) (r * sinf(theta) + y),
                  cb_right);
            _line(_screen,
                  (int) (r * cosf(theta - M_PI / 2.0f) + x),
                  (int) (r * sinf(theta - M_PI / 2.0f) + y),
                  (int) (r * cosf(theta) + x),
                  (int) (r * sinf(theta) + y),
                  cb_left);
        }

        void _disp_lasers(const Robot &robot) {
            _disp_lasers(robot.get_lasers(), robot);
            for (size_t i = 0; i < robot.get_laser_scanners().size(); ++i)
                _disp_lasers(robot.get_laser_scanners()[i].get_lasers(), robot);
        }

        void _disp_lasers(const std::vector<Laser> &lasers, const Robot &robot) {
            for (size_t i = 0; i < lasers.size(); ++i) {
                unsigned x_laser = _map->real_to_pixel(robot.get_pos().x()
                                                       + lasers[i].get_gap_dist() * cosf(robot.get_pos().theta()
                                                                                         + lasers[i].get_gap_angle()));
                unsigned y_laser = _map->real_to_pixel(robot.get_pos().y()
                                                       + lasers[i].get_gap_dist() * sinf(robot.get_pos().theta()
                                                                                         + lasers[i].get_gap_angle()));
                _line(_screen, x_laser, y_laser,
                      lasers[i].get_x_pixel(),
                      lasers[i].get_y_pixel(),
                      0xFF00000);
            }
        }

        void _disp_light_sensors(const Robot &robot) {
            for (size_t i = 0; i < robot.get_light_sensors().size(); ++i) {
                const LightSensor &ls = robot.get_light_sensors()[i];
                unsigned x_ls = _map->real_to_pixel(robot.get_pos().x());
                unsigned y_ls = _map->real_to_pixel(robot.get_pos().y());
                unsigned x_ls1 = _map->real_to_pixel(robot.get_pos().x()
                                                     +
                                                     200. / ((float) ls.get_color() + 1) * cosf(robot.get_pos().theta()
                                                                                                + ls.get_angle() -
                                                                                                ls.get_range() / 2.0));
                unsigned y_ls1 = _map->real_to_pixel(robot.get_pos().y()
                                                     +
                                                     200. / ((float) ls.get_color() + 1) * sinf(robot.get_pos().theta()
                                                                                                + ls.get_angle() -
                                                                                                ls.get_range() / 2.0));
                _line(_screen, x_ls, y_ls, x_ls1, y_ls1, _color_from_id(_screen, ls.get_color()));
                unsigned x_ls2 = _map->real_to_pixel(robot.get_pos().x()
                                                     +
                                                     200. / ((float) ls.get_color() + 1) * cosf(robot.get_pos().theta()
                                                                                                + ls.get_angle() +
                                                                                                ls.get_range() / 2.0));
                unsigned y_ls2 = _map->real_to_pixel(robot.get_pos().y()
                                                     +
                                                     200. / ((float) ls.get_color() + 1) * sinf(robot.get_pos().theta()
                                                                                                + ls.get_angle() +
                                                                                                ls.get_range() / 2.0));
                _line(_screen, x_ls, y_ls, x_ls2, y_ls2, _color_from_id(_screen, ls.get_color()));
                _line(_screen, x_ls1, y_ls1, x_ls2, y_ls2, _color_from_id(_screen, ls.get_color()));

                if (ls.get_activated()) {
                    const IlluminatedSwitch &is = *_map->get_illuminated_switches()[ls.get_num()];
                    unsigned x_is = _map->real_to_pixel(is.get_x());
                    unsigned y_is = _map->real_to_pixel(is.get_y());
                    _line(_screen, x_ls, y_ls, x_is, y_is, _color_from_id(_screen, is.get_color()));
                }
            }
        }

        void _disp_camera(const Robot &robot) {
            static const int pw = 20;
            if (!robot.use_camera())
                return;
            unsigned x_ls = _map->real_to_pixel(robot.get_pos().x());
            unsigned y_ls = _map->real_to_pixel(robot.get_pos().y());
            float a1 = robot.get_pos().theta() + robot.get_camera().get_angular_range() / 2.0;
            _line(_screen, x_ls, y_ls, cos(a1) * 200 + x_ls,
                  sin(a1) * 200 + y_ls, 0x0000ff);
            float a2 = robot.get_pos().theta() - robot.get_camera().get_angular_range() / 2.0;
            _line(_screen, x_ls, y_ls, cos(a2) * 200 + x_ls,
                  sin(a2) * 200 + y_ls, 0x0000ff);

            for (size_t i = 0; i < robot.get_camera().pixels().size(); ++i) {
                int pix = robot.get_camera().pixels()[i];
                Uint32 color = pix == -1 ? 0xffffff : _color_from_id(_screen, pix);
                SDL_Rect r;
                r.x = i * pw;
                r.y = 0;
                r.w = pw;
                r.h = pw;
                SDL_FillRect(_screen, &r, color);
            }

        }

    };

}

#endif //EXAMPLE_PYTORCH_SFERES_FASTSIM_DISPLAY_HPP
