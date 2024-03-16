__kernel void get_dark_channel(const int startThreadNum,
                               __global const int16 *r, __global const int16 *g,
                               __global const int16 *b, const int x_height,
                               const int x_width, int wnd,
                               __global int16 *darkChannel ) {
  const int idx = get_global_id(0) + startThreadNum;
  int i = idx / x_width;
  int j = idx % x_width;
  int off = idx;

  wnd = min(wnd, min(x_height, x_width));

  int rmin = max(i - wnd / 2, 0);
  int rmax = min(i + wnd / 2, x_height - 1);
  int cmin = max(j - wnd / 2, 0);
  int cmax = min(j + wnd / 2, x_width - 1);

  int16 minValue = 32767;

  int roi_start_x = x_width / 4; 
  int roi_start_y = x_height / 4; 
  int roi_width = x_width / 2; 
  int roi_height = x_height / 2; 

  for (int y = cmin; y <= cmax; y++) {
    for (int x = rmin; x <= rmax; x++) {
      int off_tmp = y * x_width + x;
     
      if (x >= roi_start_x && x < roi_start_x + roi_width && y >= roi_start_y && y < roi_start_y + roi_height) {
        minValue = min(minValue, min(min(r[off_tmp], g[off_tmp]), b[off_tmp]));
      } else {
        minValue = r[off];
      }
    }
  }

  if (get_local_id(0) == 0) {
    darkChannel[get_group_id(0)] = minValue;
  }
}


// __kernel void get_atmosphere(__global float *input, __global float *output,
//                              __local float *local_sums, int n) {
//   int global_id = get_global_id(0);
//   int local_id = get_local_id(0);
//   int local_size = get_local_size(0);

//   if (global_id < n) {
//     local_sums[local_id] = input[global_id];
//   } else {
//     local_sums[local_id] = 0;
//   }

//   barrier(CLK_LOCAL_MEM_FENCE);

//   for (int offset = local_size / 2; offset > 0; offset /= 2) {
//     if (local_id < offset) {
//       local_sums[local_id] += local_sums[local_id + offset];
//     }

//     barrier(CLK_LOCAL_MEM_FENCE);
//   }
//   if (local_id == 0) {
//     output[get_group_id(0)] = local_sums[0];
//   }
// }

// __kernel void get_transmission_estimate(__global float *image,
//                                         __global float *atmosphere, float omega,
//                                         int win_size,
//                                         __global float *dark_channel,
//                                         __global float *trans_est, int m,
//                                         int n) {
//   int idx = get_global_id(0);
//   int idy = get_global_id(1);

//   if (idx < m && idy < n) {
//     float3 pixel = vload3(idx * n + idy, image);
//     float3 atm = vload3(0, atmosphere);
//     float dark = dark_channel[idx * n + idy];

//     float trans = 1.0f - omega * dark;

//     trans = clamp(trans, 0.0f, 1.0f);

//     // printf("Work-item %d, Pixel: (%f, %f, %f), Atm: (%f, %f, %f), Dark: %f, Trans: %f\n", idx, pixel.x, pixel.y, pixel.z, atm.x, atm.y, atm.z, dark, trans);

//     vstore3(trans, idx * n + idy, trans_est);
//   }
// }


// __kernel void get_radiance(__global float *image, __global float *transmission,
//                            __global float *atmosphere, __global float *radiance,
//                            int m, int n) {
//   int idx = get_global_id(0);
//   int idy = get_global_id(1);

//   if (idx < m && idy < n) {
//     float3 pixel = vload3(idx * n + idy, image);
//     float3 atm = vload3(0, atmosphere);
//     float trans = max(transmission[idx * n + idy], 0.1f);

//     // printf("Work-item %d, Pixel: (%f, %f, %f), Atm: (%f, %f, %f), Trans: %f\n", idx, pixel.x, pixel.y, pixel.z, atm.x, atm.y, atm.z, trans);

//     float3 division_result = (pixel - atm) / trans;
//     float3 rad = division_result + atm;

//     // Clamp the values of rad between 0 and 1
//     rad = clamp(rad, 0.0f, 1.0f);

//     // printf("Work-item %d, Division result: (%f, %f, %f), Rad: (%f, %f, %f)\n", idx, division_result.x, division_result.y, division_result.z, rad.x, rad.y, rad.z);

//     vstore3(rad, idx * n + idy, radiance);
//   }
// }





