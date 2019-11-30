#pragma once

__inline__ __device__
long long warpReduceSum(long long val) {
	for (int offset = warpSize / 2; offset > 0; offset /= 2)
		val += __shfl_down_sync(0xFFFFFFFF, val, offset);
	return val;
}