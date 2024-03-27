__kernel void get_dark_channel(const int startThreadNum,
                               __global const float *r, __global const float *g,
                               __global const float *b, const int x_height,
                               const int x_width, int wnd,
                               __global float *darkChannel) {
  const int idx = get_global_id(0) + startThreadNum;
  int i = idx / x_width;
  int j = idx % x_width;
  int off = idx; 

  wnd = min(wnd, min(x_height, x_width));
  int rmin = max(i - wnd / 2, 0);
  int rmax = min(i + wnd / 2, x_height - 1);
  int cmin = max(j - wnd / 2, 0);
  int cmax = min(j + wnd / 2, x_width - 1);

  // Region of interest (ROI) boundaries
  int roi_start_x = x_width / 4;
  int roi_start_y = x_height / 4;
  int roi_end_x = x_width * 3 / 4;
  int roi_end_y = x_height * 3 / 4;

  float minValue = 99999.0f;
  for (int y = rmin; y <= rmax; y++) {
    for (int x = cmin; x <= cmax; x++) {
      int off_tmp = y * x_width + x;
      if (x >= roi_start_x && x < roi_end_x && y >= roi_start_y &&
          y < roi_end_y) {
        // Inside ROI: Use the minimum value within the window
        minValue =
            fmin(minValue, fmin(fmin(r[off_tmp], g[off_tmp]), b[off_tmp]));
      } else {
        // Outside ROI: Use the center pixel value
        minValue = fmin(minValue, r[off]);
      }
    }
  }
  darkChannel[off] = minValue;
}

__kernel void get_atmosphere(__global float *input, __global float *output,
                             int n) {
  int global_id = get_global_id(0);
  if (global_id == 0) {
    float3 maxVal = (float3)(0.0f, 0.0f, 0.0f);
    int maxIdx = 0;
    for (int i = 0; i < n; i++) {
      float3 pixel = vload3(i, input);
      if (length(pixel) > length(maxVal)) {
        maxVal = pixel;
        maxIdx = i;
      }
    }
    vstore3(maxVal, 0, output);
  }
}

__kernel void get_transmission_estimate(__global const float *image,
                                        __global const float *atmosphere,
                                        __global float *trans_est, float omega,
                                        int m, int n) {
  const int idx = get_global_id(0);
  if (idx < m * n) {
    int i = idx / n;
    int j = idx % n;

    // Region of interest (ROI) boundaries
    int roi_start_x = n / 4;
    int roi_start_y = m / 4;
    int roi_end_x = n * 3 / 4;
    int roi_end_y = m * 3 / 4;

    float3 pixel = vload3(idx, image);
    float3 atm = vload3(0, atmosphere);

    // Normalize pixel and atmosphere values to the range [0, 1]
    pixel /= 255.0f;
    atm /= 255.0f;

    // Calculate the minimum color channel value
    float min_channel = fmin(fmin(pixel.x, pixel.y), pixel.z);

    // Calculate the transmission estimate
    float transmission;
    if (i >= roi_start_y && i < roi_end_y && j >= roi_start_x &&
        j < roi_end_x) {
      transmission = 1.0f - omega * min_channel;
    } else {
      transmission = 1.0f - 0.95f * min_channel;
    }

    // Clamp the transmission value to the range [0.1, 1]
    transmission = clamp(transmission, 0.1f, 1.0f);

    trans_est[idx] = transmission;
  }
}

__kernel void get_radiance(__global const float *image,
                           __global const float *transmission,
                           __global const float *atmosphere,
                           __global float *radiance, int m, int n) {
  int idx = get_global_id(0);
  if (idx < m * n) {
    float3 pixel = vload3(idx, image);
    float3 atm = vload3(0, atmosphere);
    float trans = transmission[idx];

    // Normalize pixel and atmosphere values to the range [0, 1]
    pixel /= 255.0f;
    atm /= 255.0f;

    // Apply a maximum transmission threshold
    float tmax = 1.0f; // Maximum transmission threshold
    trans = min(trans, tmax);

    // Calculate the radiance
    float t0 = 0.1f; // Minimum transmission threshold
    float3 rad = (pixel - atm) / max(trans, t0) + atm;

    // Clamp the radiance values to the range [0, 1]
    rad = clamp(rad, 0.0f, 1.0f);

    // Scale the radiance values to the range [0, 255]
    rad *= 255.0f;

    // Store the output radiance values
    vstore3(rad, idx, radiance);
  }
}