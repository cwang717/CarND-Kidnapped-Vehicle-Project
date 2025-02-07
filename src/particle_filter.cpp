/**
 * particle_filter.cpp
 *
 * Created on: Dec 12, 2016
 * Author: Tiffany Huang
 */

#include "particle_filter.h"

#include <math.h>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "helper_functions.h"

using std::string;
using std::vector;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  /**
   * TODO: Set the number of particles. Initialize all particles to 
   *   first position (based on estimates of x, y, theta and their uncertainties
   *   from GPS) and all weights to 1. 
   * TODO: Add random Gaussian noise to each particle.
   * NOTE: Consult particle_filter.h for more information about this method 
   *   (and others in this file).
   */
  num_particles = 200;  // TODO: Set the number of particles
  std::default_random_engine gen;
  std::normal_distribution<double> dist_x(x, std[0]);
  std::normal_distribution<double> dist_y(y, std[1]);
  std::normal_distribution<double> dist_theta(theta, std[2]);

  for (int i = 0; i < num_particles; ++i) {
    Particle p;
    p.id = i;
    p.x = dist_x(gen);
    p.y = dist_y(gen);
    p.theta = dist_theta(gen);
    p.weight = 1.0;
    particles.emplace_back(p);
  }

  weights.resize(num_particles);
  is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], 
                                double velocity, double yaw_rate) {
  /**
   * TODO: Add measurements to each particle and add random Gaussian noise.
   * NOTE: When adding noise you may find std::normal_distribution 
   *   and std::default_random_engine useful.
   *  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
   *  http://www.cplusplus.com/reference/random/default_random_engine/
   */
  double x_f = 0.0;
  double y_f = 0.0;
  double theta_f = 0.0;
  std::default_random_engine motion_gen;

  for (int i = 0; i < num_particles; ++i) {
    if (yaw_rate) {
      particles[i].x += (velocity / yaw_rate) * (sin(particles[i].theta + yaw_rate * delta_t) - sin(particles[i].theta));
      particles[i].y += (velocity / yaw_rate) * (cos(particles[i].theta) - cos(particles[i].theta + yaw_rate * delta_t));
      particles[i].theta += yaw_rate * delta_t;
    } else {
      particles[i].x += velocity * delta_t * cos(particles[i].theta);
      particles[i].y += velocity * delta_t * sin(particles[i].theta);
    }
    
    // Create normal (Gaussian) distributions for x_f, y_f and theta_f
    std::normal_distribution<double> dist_x(particles[i].x, std_pos[0]);
    std::normal_distribution<double> dist_y(particles[i].y, std_pos[1]);
    std::normal_distribution<double> dist_theta(particles[i].theta, std_pos[2]);
    
    particles[i].x = dist_x(motion_gen);
    particles[i].y = dist_y(motion_gen);
    particles[i].theta = dist_theta(motion_gen);
  }

}

void ParticleFilter::dataAssociation(vector<LandmarkObs> predicted, 
                                     vector<LandmarkObs>& observations) {
  /**
   * TODO: Find the predicted measurement that is closest to each 
   *   observed measurement and assign the observed measurement to this 
   *   particular landmark.
   * NOTE: this method will NOT be called by the grading code. But you will 
   *   probably find it useful to implement this method and use it as a helper 
   *   during the updateWeights phase.
   */
  for (int i = 0; i < observations.size(); ++i) {
    double min_distance = std::numeric_limits<double>::infinity();
    int temp_id = predicted.size();
    for (int j = 0; j < predicted.size(); ++j) {
      double distance = dist(predicted[j].x, predicted[j].y, observations[i].x, observations[i].y);
      if (distance < min_distance) {
        min_distance = distance;
        temp_id = predicted[j].id;
      }
    }
    observations[i].id = temp_id;
  }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
                                   const vector<LandmarkObs> &observations, 
                                   const Map &map_landmarks) {
  /**
   * TODO: Update the weights of each particle using a mult-variate Gaussian 
   *   distribution. You can read more about this distribution here: 
   *   https://en.wikipedia.org/wiki/Multivariate_normal_distribution
   * NOTE: The observations are given in the VEHICLE'S coordinate system. 
   *   Your particles are located according to the MAP'S coordinate system. 
   *   You will need to transform between the two systems. Keep in mind that
   *   this transformation requires both rotation AND translation (but no scaling).
   *   The following is a good resource for the theory:
   *   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
   *   and the following is a good resource for the actual equation to implement
   *   (look at equation 3.33) http://planning.cs.uiuc.edu/node99.html
   */
  double total_weight = 0.0;
  for (int i = 0; i < particles.size(); ++i) {
    // coordinate transformation from vehicle coordinate to map coordinate
    vector<LandmarkObs> obs_map(observations.size());
    for (int j = 0; j < observations.size(); ++j) {
      obs_map[j] = LandmarkObs{observations[j].id,
                                                  observations[j].x * cos(particles[i].theta) - observations[j].y * sin(particles[i].theta)+                                                   particles[i].x,
                                                  observations[j].x * sin(particles[i].theta) + observations[j].y * cos(particles[i].theta)+                                                   particles[i].y};
    }
    // find landmarks in sensor range
    vector<LandmarkObs> local_landmarks;
    local_landmarks.reserve(map_landmarks.landmark_list.size());
    for (int k = 0; k < map_landmarks.landmark_list.size(); ++k) {
      if (sensor_range > dist(particles[i].x, particles[i].y, 
                             map_landmarks.landmark_list[k].x_f, map_landmarks.landmark_list[k].y_f)) {
       local_landmarks.push_back(LandmarkObs{map_landmarks.landmark_list[k].id_i,
                                               map_landmarks.landmark_list[k].x_f,
                                               map_landmarks.landmark_list[k].y_f});
      }
    }
    dataAssociation(local_landmarks, obs_map);
    // multivariate gaussian distribution to find weights
    for (int l = 0; l < local_landmarks.size(); ++l) {
      for (int m = 0; m < obs_map.size(); ++m) {
        if (local_landmarks[l].id == obs_map[m].id) {
          particles[i].weight *= (1.0 / (2.0 * M_PI * std_landmark[0] * std_landmark[1])) *
                                 exp(-1.0 * ((pow((obs_map[m].x - local_landmarks[l].x), 2) /
                                 (2.0 * std_landmark[0] * std_landmark[0])) + 
                                 (pow((obs_map[m].y - local_landmarks[l].y), 2) /
                                 (2.0 * std_landmark[1] * std_landmark[1]))));
        }
      }
    }
    total_weight += particles[i].weight;
  }
  // normalize weights
  for (int n = 0; n < particles.size(); ++n) {
    particles[n].weight = particles[n].weight / total_weight;
  }
}

void ParticleFilter::resample() {
  /**
   * TODO: Resample particles with replacement with probability proportional 
   *   to their weight. 
   * NOTE: You may find std::discrete_distribution helpful here.
   *   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
   */
  vector<Particle> particles_sampled (particles.size());
  // create weights vector to use in discrete_distribution
  vector<double> weights(particles.size());
  for (int i = 0; i < particles.size(); ++i) {
    weights[i] = particles[i].weight;
  }
  // resample
  std::random_device rd;
  std::mt19937 gen(rd());
  std::discrete_distribution<> d(weights.begin(),weights.end());
  for (int j = 0; j < particles.size(); ++j) {      
    particles_sampled[j] = particles[d(gen)];
  }
  particles = particles_sampled;
}

void ParticleFilter::SetAssociations(Particle& particle, 
                                     const vector<int>& associations, 
                                     const vector<double>& sense_x, 
                                     const vector<double>& sense_y) {
  // particle: the particle to which assign each listed association, 
  //   and association's (x,y) world coordinates mapping
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates
  particle.associations= associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best) {
  vector<int> v = best.associations;
  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseCoord(Particle best, string coord) {
  vector<double> v;

  if (coord == "X") {
    v = best.sense_x;
  } else {
    v = best.sense_y;
  }

  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}