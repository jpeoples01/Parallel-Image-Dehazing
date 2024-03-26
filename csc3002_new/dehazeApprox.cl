#pragma OPENCL EXTENSION cl_khr_fp16 : enable

__kernel void get_dark_channel(const int startThreadNum, __global const half *r, __global const half *g, __global const half *b, const int x_height, const int x_width, int wnd, __global half *darkChannel) {
    const int idx = get_global_id(0) + startThreadNum;
    int i = idx / x_width;
    int j = idx % x_width;
    int off = idx; // Use the 1D global ID as the offset
    
    wnd = min(wnd, min(x_height, x_width));
    int rmin = max(i - wnd / 2, 0);
    int rmax = min(i + wnd / 2, x_height - 1);
    int cmin = max(j - wnd / 2, 0);
    int cmax = min(j + wnd / 2, x_width - 1);

    half minValue = 65504.0h;
    for (int y = rmin; y <= rmax; y++) {
        for (int x = cmin; x <= cmax; x++) {
            int off_tmp = y * x_width + x;
            minValue = fmin(minValue, fmin(fmin(r[off_tmp], g[off_tmp]), b[off_tmp]));
        }
    }
    darkChannel[off] = minValue;
}

__kernel void get_atmosphere(__global half *input, __global half *output, int n) {
    int global_id = get_global_id(0);
    if (global_id == 0) {
        half3 maxVal = (half3)(0.0h, 0.0h, 0.0h);
        int maxIdx = 0;
        for (int i = 0; i < n; i++) {
            half3 pixel = vload3(i, input);
            if (length(pixel) > length(maxVal)) {
                maxVal = pixel;
                maxIdx = i;
            }
        }
        vstore3(maxVal, 0, output);
    }
}

__kernel void get_transmission_estimate(__global const half *image, __global const half *atmosphere, __global half *trans_est, half omega, int m, int n) {
    const int idx = get_global_id(0);
    if (idx < m * n) {
        int i = idx / n;
        int j = idx % n;
        
        half3 pixel = vload3(idx, image);
        half3 atm = vload3(0, atmosphere);
        
        // Normalize pixel and atmosphere values to the range [0, 1]
        pixel /= 255.0h;
        atm /= 255.0h;
        
        // Calculate the minimum color channel value
        half min_channel = fmin(fmin(pixel.x, pixel.y), pixel.z);
        
        // Calculate the transmission estimate
        half transmission = 1.0h - omega * min_channel;
        
        // Clamp the transmission value to the range [0.1, 1]
        transmission = clamp(transmission, 0.1h, 1.0h);
        
        trans_est[idx] = transmission;
    }
}

__kernel void get_radiance(__global const half *image, __global const half *transmission, __global const half *atmosphere, __global half *radiance, int m, int n) {
    int idx = get_global_id(0);
    if (idx < m * n) {
        half3 pixel = vload3(idx, image);
        half3 atm = vload3(0, atmosphere);
        half trans = transmission[idx];
        
        // Normalize pixel and atmosphere values to the range [0, 1]
        pixel /= 255.0h;
        atm /= 255.0h;
        
        // Apply a maximum transmission threshold
        half tmax = 1.0h; // Maximum transmission threshold
        trans = min(trans, tmax);
        
        // Calculate the radiance using the modified dehazing formula
        half t0 = 0.1h; // Minimum transmission threshold
        half3 rad = (pixel - atm) / max(trans, t0) + atm;
        
        // Clamp the radiance values to the range [0, 1]
        rad = clamp(rad, 0.0h, 1.0h);
        
        // Scale the radiance values to the range [0, 255]
        rad *= 255.0h;
        
        // Store the output radiance values
        vstore3(rad, idx, radiance);
    }
}