#include <iostream>
#include <opencv2/opencv.hpp>
#include <unordered_map>
#include <math.h>

using namespace std;
using namespace cv;

// Load image.
Mat load_image(std::string input_file) {
  Mat image = cv::imread(input_file);
  return image;
}


// Save image.
void save_image(Mat image, std::string output_file) {
  cv::namedWindow ("Show Image");
  cv::imshow("Show Image", image);
  cv::waitKey(0);
  // imwrite(output_file, image);
}


// Copy image.
Mat copy_image(Mat input_image, int kernel_size, int window_size) {
  // Information of input image
  int width_input = input_image.cols;
  int height_input = input_image.rows;
  int width_step_input = input_image.step[0];
  int n_channels_input = input_image.step[1];

  // Create output image
  Mat output_image = Mat(height_input - kernel_size - window_size + 1, width_input - kernel_size - window_size + 1, CV_8UC3);
  int width_step_output = output_image.step[0];
  int n_channels_output = output_image.step[1];

  // Loop
  int padding_col = kernel_size / 2 * n_channels_input + window_size / 2 * n_channels_input;
  int padding_row = width_step_input * (kernel_size/2) + width_step_input * (window_size/2);
  uchar* p_data_input = (uchar*)input_image.data + padding_col + padding_row;
  uchar* p_data_output = (uchar*)output_image.data;
  for (int y = 0; y < height_input - kernel_size - window_size + 1; y++, p_data_input += width_step_input, p_data_output += width_step_output) {
    uchar* p_row_input = p_data_input;
    uchar* p_row_output = p_data_output;
    for (int x = 0; x < width_input - kernel_size - window_size + 1; x++, p_row_input += n_channels_input, p_row_output += n_channels_output) {
      p_row_output[0] = p_row_input[0];
      p_row_output[1] = p_row_input[1];
      p_row_output[2] = p_row_input[2];
    }
  }

  return output_image;
}


// Convert a color image to the gray image.
Mat rgb2gray(Mat input_image) {
  // Information of input image
  int width_input = input_image.cols;
  int height_input = input_image.rows;
  int width_step_input = input_image.step[0];
  int n_channels_input = input_image.step[1];

  // Create output image
  Mat output_image = Mat(height_input, width_input, CV_8UC1);
  int width_step_output = output_image.step[0];
  int n_channels_output = output_image.step[1];

  // Loop
  uchar* p_data_input = (uchar*)input_image.data;
  uchar* p_data_output = (uchar*)output_image.data;
  for (int y = 0; y < height_input; y++, p_data_input += width_step_input, p_data_output += width_step_output) {
    uchar* p_row_input = p_data_input;
    uchar* p_row_output = p_data_output;
    for (int x = 0; x < width_input; x++, p_row_input += n_channels_input, p_row_output += n_channels_output) {
      p_row_output[0] = 0.114 * p_row_input[0] + 0.587 * p_row_input[1] + 0.299 * p_row_input[2];
    }
  }

  return output_image;
}



// Filter image with gaussian to reduce noise.
Mat filter_gau(Mat input_image, float sigma, int kernel_size) {
  // Create gaussian kernel
  vector<vector<double>> gau_kernel(kernel_size, vector<double>(kernel_size, 0.0));
  double sum = 0.0;
  for (int x = -kernel_size/2; x <= kernel_size/2; ++x) {
    for (int y = -kernel_size/2; y <= kernel_size/2; ++y) {
      double exponent = -(x * x + y * y) / (2 * sigma * sigma);
      gau_kernel[x + kernel_size/2][y + kernel_size/2] = (1 / (2 * 3.14 * sigma * sigma)) * exp(exponent);
      sum += gau_kernel[x + kernel_size/2][y + kernel_size/2];
    }
  }
  for (int i = 0; i < kernel_size; ++i) {
    for (int j = 0; j < kernel_size; ++j) {
      gau_kernel[i][j] /= sum;
    }
  }


  // Information of input image
  int width_input = input_image.cols;
  int height_input = input_image.rows;
  int width_step_input = input_image.step[0];
  int n_channels_input = input_image.step[1];

  // Create output image
  Mat output_image;
  if (n_channels_input == 1) {
    output_image = Mat(height_input - kernel_size + 1, width_input - kernel_size + 1, CV_8UC1);
  } else if (n_channels_input == 3) {
    output_image = Mat(height_input - kernel_size + 1, width_input - kernel_size + 1, CV_8UC3);
  }
  int width_step_output = output_image.step[0];
  int n_channels_output = output_image.step[1];

  // Loop
  int padding_col = kernel_size / 2 * n_channels_input;
  int padding_row = width_step_input * (kernel_size/2);
  uchar* p_data_input = (uchar*)input_image.data + padding_row + padding_col;
  uchar* p_data_output = (uchar*)output_image.data;
  for (int y = 0; y < height_input - kernel_size + 1; y++, p_data_input += width_step_input, p_data_output += width_step_output) {
    uchar* p_row_input = p_data_input;
    uchar* p_row_output = p_data_output;
    for (int x = 0; x < width_input - kernel_size + 1; x++, p_row_input += n_channels_input, p_row_output += n_channels_output) {
      // Loop through kernel
      vector<float> gau_sum(n_channels_input, 0.0);
      uchar* p_pixel = p_row_input - padding_col - padding_row;
      for (int ky = 0; ky < kernel_size; ky++, p_pixel += width_step_input) {
        uchar* p_row_pixel = p_pixel;
        for (int kx = 0; kx < kernel_size; kx++, p_row_pixel += n_channels_input) {
          for (int c = 0; c < n_channels_input; c++) {
            gau_sum[c] += p_row_pixel[c] * gau_kernel[ky][kx];
          }
        }
      }
      for (int c = 0; c < n_channels_input; c++) {
        p_row_output[c] = gau_sum[c];
      }
    }
  }

  return output_image;
}


// Compute magnitude of the x at each pixel.
Mat compute_gradient_dx(Mat input_image) {
  // Information of input image
  int width_input = input_image.cols;
  int height_input = input_image.rows;
  int width_step_input = input_image.step[0];
  int n_channels_input = input_image.step[1];

  // Create output image
  Mat output_image = Mat(height_input - 1, width_input - 1, CV_8UC1);
  int width_step_output = output_image.step[0];
  int n_channels_output = output_image.step[1];

  // Loop
  uchar* p_data_input = (uchar*)input_image.data;
  uchar* p_data_output = (uchar*)output_image.data;
  for (int y = 0; y < height_input - 1; y++, p_data_input += width_step_input, p_data_output += width_step_output) {
    uchar* p_row_input = p_data_input;
    uchar* p_row_output = p_data_output;
    for (int x = 0; x < width_input - 1; x++, p_row_input += n_channels_input, p_row_output += n_channels_output) {
      int number = p_row_input[1] - p_row_input[0];
      if (number < 0) {
        number *= -1;
      }
      p_row_output[0] = number;
    }
  }
  return output_image;
}


// Compute magnitude of the y at each pixel.
Mat compute_gradient_dy(Mat input_image) {
  // Information of input image
  int width_input = input_image.cols;
  int height_input = input_image.rows;
  int width_step_input = input_image.step[0];
  int n_channels_input = input_image.step[1];

  // Create output image
  Mat output_image = Mat(height_input - 1, width_input - 1, CV_8UC1);
  int width_step_output = output_image.step[0];
  int n_channels_output = output_image.step[1];

  // Loop
  uchar* p_data_input = (uchar*)input_image.data;
  uchar* p_data_output = (uchar*)output_image.data;
  for (int y = 0; y < height_input - 1; y++, p_data_input += width_step_input, p_data_output += width_step_output) {
    uchar* p_row_input = p_data_input;
    uchar* p_row_output = p_data_output;
    for (int x = 0; x < width_input - 1; x++, p_row_input += n_channels_input, p_row_output += n_channels_output) {
      int number = p_row_input[0] - p_row_input[width_step_input];
      if (number < 0) {
        number *= -1;
      }
      p_row_output[0] = number;
    }
  }
  return output_image;
}


// Harris key point detection algorithm.
std::vector<std::vector<double>> harris(Mat dx, Mat dy, int window_size, double k) {
  // Information of input image
  int width_input = dx.cols;
  int height_input = dx.rows;
  int width_step_input = dx.step[0];
  int n_channels_input = dx.step[1];

  // Output 
  std::vector<std::vector<double>> R_list;

  // Loop
  int padding_col = window_size / 2 * n_channels_input;
  int padding_row = width_step_input * (window_size/2);
  uchar* p_data_input_dx = (uchar*)dx.data + padding_col + padding_row;
  uchar* p_data_input_dy = (uchar*)dy.data + padding_col + padding_row;
  for (int y = 0; y < height_input - window_size + 1; y++, p_data_input_dx += width_step_input, p_data_input_dy += width_step_input) {
    uchar* p_row_input_dx = p_data_input_dx;
    uchar* p_row_input_dy = p_data_input_dy;
    std::vector<double> R_list_row;
    for (int x = 0; x < width_input - window_size + 1; x++, p_row_input_dx += n_channels_input, p_row_input_dy += n_channels_input) {
      // Construct M in a window around each pixel (Harris uses a Gaussian window)
      double Mxx = 0.0, Myy = 0.0, Mxy = 0.0;
      uchar* p_kernel_dx = p_row_input_dx - padding_col - padding_row;
      uchar* p_kernel_dy = p_row_input_dy - padding_col - padding_row;
      for (int ky = 0; ky < window_size; ky++, p_kernel_dx += width_step_input, p_kernel_dy += width_step_input) {
        uchar* p_row_kernel_dx = p_kernel_dx;
        uchar* p_row_kernel_dy = p_kernel_dy;
        for (int kx = 0; kx < window_size; kx++, p_row_kernel_dx += n_channels_input, p_row_kernel_dy += n_channels_input) {
          // Gaussian window
          double weight = exp(-(kx * kx + ky * ky) / (2 * (window_size * window_size)));

          Mxx += p_row_kernel_dx[0] * p_row_kernel_dx[0] * weight;
          Myy += p_row_kernel_dy[0] * p_row_kernel_dy[0] * weight;
          Mxy += p_row_kernel_dx[0] * p_row_kernel_dy[0] * weight;
        }
      }

      // Compute R
      double det = Mxx * Myy - Mxy * Mxy;
      double trace = Mxx + Myy;
      double R = det - k * (trace * trace);
      R_list_row.push_back(R);
    }
    R_list.push_back(R_list_row);
  }
  return R_list;
}


// Draw corners
Mat draw (Mat output, std::vector<std::vector<double>> R_list, double threshold) {
  // Information of input image
  int width_input = output.cols;
  int height_input = output.rows;
  int width_step_input = output.step[0];
  int n_channels_input = output.step[1];

  for (int y = 0; y < height_input; y++) {
    for (int x = 0; x < width_input; x++) {
      if (R_list[y][x] > threshold) {
        double max = R_list[y][x];

        // Local non-maxima suppression
        bool skip = false;
        for (int nrow = -2; nrow <= 2; ++nrow) {
          for (int ncol = -2; ncol <= 2; ++ncol) {
            int new_row = y + nrow;
            int new_col = x + ncol;
            if (new_row >= 0 && new_row < height_input && new_col >= 0 && new_col < width_input) {
              if (R_list[new_row][new_col] > max) {
                skip = true;
                break;
              }
            }
          }
          
          if (skip) {
            break; // Break out of the inner loop
          }
        }

        if (!skip) {
          cv::circle(output, cv::Point(x, y), 2, cv::Scalar(0, 0, 255), 2);
        }
      }
    }
  }
  return output;
}


int main(int argc, char* argv[]) {
  // Information of command line arguments
  std::string command = argv[1];
  std::string input_file = argv[2];
  std::string output_file = argv[3];

  // Load image with grayscale
  Mat image = load_image(input_file);
  if (image.empty()) {
    cout << "Could not open the image!" << endl;
    return -1;
  }

  if (command == "-harris") {
    // Constant
    float sigma = 1.0;
    int kernel_size = 5;
    int window_size = 5;
    double k = 0.06;
    double threshold = 100000;

    // Grayscale
    Mat process_image = rgb2gray(image);

    // Filter image
    process_image = filter_gau(process_image, sigma, kernel_size);

    // Gradient dx
    Mat dx = compute_gradient_dx(process_image);

    // Gradient dy
    Mat dy = compute_gradient_dy(process_image);

    // Harris
    std::vector<std::vector<double>> R_list = harris(dx, dy, window_size, k);

    // Copy output image from input image
    Mat output = copy_image(image, kernel_size, window_size);

    // Draw corners on the image
    Mat output_draw = draw(output, R_list, threshold);

    // Save image
    save_image(output_draw, output_file);
  } else {
    cout << "We do not have this command!" << endl;
  }

  return 0;
}