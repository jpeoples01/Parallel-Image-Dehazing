__kernel void get_dark_channel(const int startThreadNum,
                               __global const float *r, __global const float *g,
                               __global const float *b, const int x_height,
                               const int x_width, int wnd,
                               __global float *darkChannel) {
    const int idx = get_global_id(0) + startThreadNum;
    int i = idx / x_width;
    int j = idx % x_width;
    int off = i * x_width + j;

    // printf("Work-item %d, r: %f, g: %f, b: %f\n", idx, r[off], g[off], b[off]);

    wnd = min(wnd, min(x_height, x_width));

    int rmin = max(i - wnd / 2, 0);
    int rmax = min(i + wnd / 2, x_height - 1);
    int cmin = max(j - wnd / 2, 0);
    int cmax = min(j + wnd / 2, x_width - 1);

    float minValue = 99999.0f;

    for (int y = cmin; y <= cmax; y++) {
        for (int x = rmin; x <= rmax; x++) {
            int off_tmp = y * x_width + x;
            minValue = fmin(minValue, fmin(fmin(r[off_tmp], g[off_tmp]), b[off_tmp]));
        }
    }

    darkChannel[off] = minValue;

    // printf("Work-item %d, darkChannel: %f\n", idx, darkChannel[off]);
}

__kernel void get_atmosphere(__global float *input, __global float *output,
                             __local float *local_sums, int n) {
  int global_id = get_global_id(0);
  int local_id = get_local_id(0);
  int local_size = get_local_size(0);

  if (global_id < n) {
    local_sums[local_id] = input[global_id];
  } else {
    local_sums[local_id] = 0;
  }

  barrier(CLK_LOCAL_MEM_FENCE);

  if (local_id == 0) {
    output[get_group_id(0)] = local_sums[0];
  }
}


__kernel void get_transmission_estimate(__global const float *image, __global const float *atmosphere, __global float *trans_est, float omega, int m, int n) {
    const int idx = get_global_id(0);
    int i = idx / n;
    int j = idx % n;
    
    if (idx < m * n) {
        float3 pixel = vload3(idx, image);
        float3 atm = vload3(0, atmosphere);
        
        // Normalize pixel and atmosphere values to the range [0, 1]
        pixel /= 255.0f;
        atm /= 255.0f;
        
        // Calculate the minimum color channel value
        float min_channel = fmin(fmin(pixel.x, pixel.y), pixel.z);
        
        // Calculate the transmission estimate
        float transmission = 1.0f - omega * min_channel;
        
        // Clamp the transmission value to the range [0.1, 1]
        transmission = clamp(transmission, 0.1f, 1.0f);
        
        float3 trans = (float3)(transmission, transmission, transmission);
        vstore3(trans, idx, trans_est);
    }
}


__kernel void get_radiance(__global const float *image, __global const float *transmission, __global const float *atmosphere, __global float *radiance, int m, int n) {
    int idx = get_global_id(0);
    if (idx < m * n) {
        float3 pixel = vload3(idx, image);
        float3 atm = vload3(0, atmosphere);
        float trans = transmission[idx];

        // Normalize pixel and atmosphere values to the range [0, 1]
        pixel /= 255.0f;
        atm /= 255.0f;

        // Calculate the radiance using the modified dehazing formula
        float3 rad = (pixel - atm) / (trans + 0.0001f) + atm;

        // Clamp the radiance values to the range [0, 1]
        rad = clamp(rad, 0.0f, 1.0f);

        // Scale the radiance values to the range [0, 255]
        rad *= 255.0f;

        // Store the output radiance values
        vstore3(rad, idx, radiance);
    }
}




