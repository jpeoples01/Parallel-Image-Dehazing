__kernel void get_dark_channel(const int startThreadNum,
                               __global const float *r, __global const float *g,
                               __global const float *b, const int x_height,
                               const int x_width, int wnd,
                               __global float *darkChannel ) {
                              //  __local float *local_min) {
  const int idx = get_global_id(0) + startThreadNum;
  int i = idx / x_width;
  int j = idx % x_width;
  int off = idx;

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

  if (get_local_id(0) == 0) {
    darkChannel[get_group_id(0)] = minValue;

    // printf("Work-item %d, darkChannel: %f\n", idx, darkChannel[off]);
  }
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
  if (local_id == 0) {
    output[get_group_id(0)] = local_sums[0];
  }
}

__kernel void get_transmission_estimate(__global const float *image, 
                                         __global const float *atmosphere, float omega,
                                         int win_size, __global const float *darkChannel, __global float *trans_est, int m,
                                         int n) {
  int idx = get_global_id(0);
  if (idx < m * n) {
    int i = idx / n;
    int j = idx % n;

    float3 pixel = vload3(idx * n + j, image);
    float3 atm = vload3(0, atmosphere);

    float trans = 1.0f - omega * darkChannel[idx]; 

    vstore3(trans, idx * n + j, trans_est);
  }
}



__kernel void get_radiance(__global const float *image, 
                               __global const float *trans_est, 
                               __global const float *atmosphere, 
                               __global float *radiance, int m, int n) {
  int idx = get_global_id(0);
  if (idx < m * n) {
    int i = idx / n;
    int j = idx % n; 

    float3 J = vload3(idx * n + j, image);
    float3 A = vload3(0, atmosphere);
    float T = trans_est[idx * n + j];

    // Radiance recovery formula
    float3 radiance_est = (J - A) / max(T, 0.1f);

    vstore3(radiance_est, idx * n + j, radiance);
  }
}



