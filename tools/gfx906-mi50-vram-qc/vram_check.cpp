#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr uint32_t kSeed = 0xA5A55A5Au;

__device__ __forceinline__ uint32_t pattern_for(uint64_t index, int device_id) {
  uint32_t x = static_cast<uint32_t>(index);
  x ^= static_cast<uint32_t>(index >> 32);
  x ^= static_cast<uint32_t>(device_id) * 0x9E3779B9u;
  return kSeed ^ (x * 0x85EBCA6Bu) ^ (x >> 13);
}

__global__ void fill_kernel(uint32_t* data, uint64_t count, int device_id) {
  uint64_t stride = static_cast<uint64_t>(blockDim.x) * gridDim.x;
  uint64_t index = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  for (; index < count; index += stride) {
    data[index] = pattern_for(index, device_id);
  }
}

__global__ void check_kernel(const uint32_t* data,
                             uint64_t count,
                             int device_id,
                             unsigned long long* errors) {
  uint64_t stride = static_cast<uint64_t>(blockDim.x) * gridDim.x;
  uint64_t index = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  unsigned long long local_errors = 0;
  for (; index < count; index += stride) {
    if (data[index] != pattern_for(index, device_id)) {
      ++local_errors;
    }
  }
  if (local_errors != 0) {
    atomicAdd(errors, local_errors);
  }
}

std::string hip_error(hipError_t err) {
  return std::string(hipGetErrorName(err)) + ": " + hipGetErrorString(err);
}

bool parse_double(const std::string& text, double* value) {
  char* end = nullptr;
  *value = std::strtod(text.c_str(), &end);
  return end != text.c_str() && *end == '\0' && std::isfinite(*value);
}

int parse_device_id(const std::string& text) {
  char* end = nullptr;
  long value = std::strtol(text.c_str(), &end, 10);
  if (end == text.c_str() || *end != '\0' || value < 0 ||
      value > std::numeric_limits<int>::max()) {
    throw std::runtime_error("device must be all or a non-negative integer");
  }
  return static_cast<int>(value);
}

void usage(const char* argv0) {
  std::cerr << "Usage: " << argv0
            << " --target-gib <float> --device <id|all>\n";
}

bool run_device_check(int device_id, uint64_t bytes, double target_gib) {
  std::cout << "Device " << device_id << ":\n";

  hipError_t err = hipSetDevice(device_id);
  if (err != hipSuccess) {
    std::cout << "  result: FAIL\n";
    std::cout << "  error: hipSetDevice failed: " << hip_error(err) << "\n";
    return false;
  }

  hipDeviceProp_t props{};
  err = hipGetDeviceProperties(&props, device_id);
  if (err == hipSuccess) {
    std::cout << "  name: " << props.name << "\n";
    std::cout << "  totalGlobalMem: "
              << static_cast<unsigned long long>(props.totalGlobalMem)
              << " bytes\n";
  } else {
    std::cout << "  name: unknown (hipGetDeviceProperties failed: "
              << hip_error(err) << ")\n";
  }

  size_t free_bytes = 0;
  size_t total_bytes = 0;
  err = hipMemGetInfo(&free_bytes, &total_bytes);
  if (err == hipSuccess) {
    std::cout << "  hipMemGetInfo free: "
              << static_cast<unsigned long long>(free_bytes) << " bytes\n";
    std::cout << "  hipMemGetInfo total: "
              << static_cast<unsigned long long>(total_bytes) << " bytes\n";
  }

  std::cout << std::fixed << std::setprecision(2);
  std::cout << "  target allocation: " << target_gib << " GiB\n";
  std::cout.unsetf(std::ios::floatfield);

  uint32_t* data = nullptr;
  err = hipMalloc(reinterpret_cast<void**>(&data), static_cast<size_t>(bytes));
  if (err != hipSuccess) {
    std::cout << "  result: FAIL\n";
    std::cout << "  error: hipMalloc(" << bytes << " bytes) failed: "
              << hip_error(err) << "\n";
    return false;
  }

  unsigned long long* d_errors = nullptr;
  err = hipMalloc(reinterpret_cast<void**>(&d_errors), sizeof(unsigned long long));
  if (err != hipSuccess) {
    hipFree(data);
    std::cout << "  result: FAIL\n";
    std::cout << "  error: hipMalloc(error counter) failed: " << hip_error(err)
              << "\n";
    return false;
  }

  err = hipMemset(d_errors, 0, sizeof(unsigned long long));
  if (err != hipSuccess) {
    hipFree(d_errors);
    hipFree(data);
    std::cout << "  result: FAIL\n";
    std::cout << "  error: hipMemset(error counter) failed: " << hip_error(err)
              << "\n";
    return false;
  }

  uint64_t count = bytes / sizeof(uint32_t);
  constexpr int kThreads = 256;
  int blocks = static_cast<int>((count + kThreads - 1) / kThreads);
  if (blocks < 1) {
    blocks = 1;
  }
  if (blocks > 65535) {
    blocks = 65535;
  }

  hipLaunchKernelGGL(fill_kernel,
                     dim3(blocks),
                     dim3(kThreads),
                     0,
                     0,
                     data,
                     count,
                     device_id);
  err = hipGetLastError();
  if (err == hipSuccess) {
    err = hipDeviceSynchronize();
  }
  if (err != hipSuccess) {
    hipFree(d_errors);
    hipFree(data);
    std::cout << "  result: FAIL\n";
    std::cout << "  error: fill kernel failed: " << hip_error(err) << "\n";
    return false;
  }

  hipLaunchKernelGGL(check_kernel,
                     dim3(blocks),
                     dim3(kThreads),
                     0,
                     0,
                     data,
                     count,
                     device_id,
                     d_errors);
  err = hipGetLastError();
  if (err == hipSuccess) {
    err = hipDeviceSynchronize();
  }
  if (err != hipSuccess) {
    hipFree(d_errors);
    hipFree(data);
    std::cout << "  result: FAIL\n";
    std::cout << "  error: check kernel failed: " << hip_error(err) << "\n";
    return false;
  }

  unsigned long long host_errors = 0;
  err = hipMemcpy(&host_errors,
                  d_errors,
                  sizeof(host_errors),
                  hipMemcpyDeviceToHost);

  hipError_t free_errors_err = hipFree(d_errors);
  hipError_t free_data_err = hipFree(data);

  if (err != hipSuccess) {
    std::cout << "  result: FAIL\n";
    std::cout << "  error: copying error count failed: " << hip_error(err)
              << "\n";
    return false;
  }
  if (free_errors_err != hipSuccess || free_data_err != hipSuccess) {
    std::cout << "  result: FAIL\n";
    std::cout << "  error: hipFree failed: "
              << hip_error(free_errors_err != hipSuccess ? free_errors_err
                                                         : free_data_err)
              << "\n";
    return false;
  }

  if (host_errors == 0) {
    std::cout << "  checked words: " << static_cast<unsigned long long>(count)
              << "\n";
    std::cout << "  result: PASS\n";
    return true;
  }

  std::cout << "  checked words: " << static_cast<unsigned long long>(count)
            << "\n";
  std::cout << "  mismatched words: " << host_errors << "\n";
  std::cout << "  result: FAIL\n";
  return false;
}

}  // namespace

int main(int argc, char** argv) {
  std::string device_arg = "all";
  double target_gib = 30.0;

  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--target-gib") {
      if (++i >= argc || !parse_double(argv[i], &target_gib)) {
        usage(argv[0]);
        return 2;
      }
    } else if (arg == "--device") {
      if (++i >= argc) {
        usage(argv[0]);
        return 2;
      }
      device_arg = argv[i];
    } else if (arg == "--help" || arg == "-h") {
      usage(argv[0]);
      return 0;
    } else {
      std::cerr << "unknown argument: " << arg << "\n";
      usage(argv[0]);
      return 2;
    }
  }

  if (!(target_gib > 0.0)) {
    std::cerr << "target GiB must be greater than zero\n";
    return 2;
  }

  long double requested_bytes =
      static_cast<long double>(target_gib) * 1024.0L * 1024.0L * 1024.0L;
  if (requested_bytes >
      static_cast<long double>(std::numeric_limits<uint64_t>::max())) {
    std::cerr << "target GiB is too large\n";
    return 2;
  }
  uint64_t bytes = static_cast<uint64_t>(requested_bytes);
  bytes -= bytes % sizeof(uint32_t);
  if (bytes == 0) {
    std::cerr << "target allocation rounds down to zero bytes\n";
    return 2;
  }
  if (bytes > static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
    std::cerr << "target allocation exceeds this platform's size_t range\n";
    return 2;
  }

  int device_count = 0;
  hipError_t err = hipGetDeviceCount(&device_count);
  if (err != hipSuccess) {
    std::cerr << "hipGetDeviceCount failed: " << hip_error(err) << "\n";
    return 1;
  }
  if (device_count <= 0) {
    std::cerr << "No HIP devices found.\n";
    return 1;
  }

  std::vector<int> devices;
  try {
    if (device_arg == "all") {
      for (int id = 0; id < device_count; ++id) {
        devices.push_back(id);
      }
    } else {
      int id = parse_device_id(device_arg);
      if (id >= device_count) {
        std::cerr << "Requested device " << id << ", but only " << device_count
                  << " HIP device(s) are visible.\n";
        return 2;
      }
      devices.push_back(id);
    }
  } catch (const std::exception& ex) {
    std::cerr << ex.what() << "\n";
    return 2;
  }

  std::cout << "HIP devices visible: " << device_count << "\n";
  std::cout << "Selected devices:";
  for (int id : devices) {
    std::cout << " " << id;
  }
  std::cout << "\n\n";

  bool all_ok = true;
  for (size_t index = 0; index < devices.size(); ++index) {
    bool ok = run_device_check(devices[index], bytes, target_gib);
    all_ok = all_ok && ok;
    if (index + 1 < devices.size()) {
      std::cout << "\n";
    }
  }

  return all_ok ? 0 : 1;
}
