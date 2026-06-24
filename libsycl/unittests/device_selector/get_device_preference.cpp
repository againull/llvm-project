//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <common/device_images.hpp>

#include <detail/device_impl.hpp>
#include <detail/platform_impl.hpp>
#include <detail/program_manager.hpp>

#include <mock/helpers.hpp>

#include <sycl/__impl/detail/obj_utils.hpp>
#include <sycl/__impl/device_selector.hpp>
#include <sycl/sycl.hpp>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <optional>

using namespace sycl;
using namespace ::testing;

namespace {

class ScopedBinaryRegistration {
public:
  explicit ScopedBinaryRegistration(llvm::ArrayRef<llvm::StringRef> KernelNames)
      : MBinary(sycl::unittest::createSYCLDeviceBinary(KernelNames)) {
    sycl::detail::ProgramAndKernelManager::getInstance().registerFatBin(
        MBinary.data(), MBinary.size());
  }

  ~ScopedBinaryRegistration() {
    sycl::detail::ProgramAndKernelManager::getInstance().unregisterFatBin(
        MBinary.data(), MBinary.size());
  }

private:
  llvm::SmallString<0> MBinary;
};

// Builds an in-memory PlatformImpl + DeviceImpls without going through
// liboffload device discovery, so the global platform cache is never touched.
class DeviceSelectorScoreTest : public ::testing::Test {
protected:
  void SetUp() override {
    Device1Handle = mock::createDummyHandle<ol_device_handle_t>();
    Device2Handle = mock::createDummyHandle<ol_device_handle_t>();

    Platform = std::make_unique<detail::PlatformImpl>(
        backend::level_zero, detail::PlatformImpl::ForTestingTag{});

    auto MakeDevice = [this](ol_device_handle_t Handle) {
      auto Impl = std::make_unique<detail::DeviceImpl>(
          Handle, *Platform, detail::DeviceImpl::ForTestingTag{});
      sycl::device Dev =
          detail::createSyclObjFromImpl<sycl::device>(*Impl);
      DeviceImpls.push_back(std::move(Impl));
      return Dev;
    };
    Device1 = MakeDevice(Device1Handle);
    Device2 = MakeDevice(Device2Handle);
  }

  void TearDown() override {
    DeviceImpls.clear();
    Platform.reset();
    mock::releaseDummyHandles(Device1Handle, Device2Handle);
  }

  mock::MockWrapper Mock;
  std::unique_ptr<detail::PlatformImpl> Platform;
  std::vector<std::unique_ptr<detail::DeviceImpl>> DeviceImpls;
  ol_device_handle_t Device1Handle{};
  ol_device_handle_t Device2Handle{};
  std::optional<sycl::device> Device1;
  std::optional<sycl::device> Device2;
};

TEST_F(DeviceSelectorScoreTest, CPUAndGPU) {
  EXPECT_CALL(Mock.get(), olGetDeviceInfo(_, OL_DEVICE_INFO_TYPE, _, _))
      .WillRepeatedly([this](ol_device_handle_t Device,
                             ol_device_info_t /*PropName*/, size_t /*PropSize*/,
                             void *PropValue) -> ol_result_t {
        if (Device == Device1Handle)
          *static_cast<ol_device_type_t *>(PropValue) = OL_DEVICE_TYPE_GPU;
        else if (Device == Device2Handle)
          *static_cast<ol_device_type_t *>(PropValue) = OL_DEVICE_TYPE_CPU;
        else
          return mock::getMockLiboffload().makeEmptyStrError(
              OL_ERRC_INVALID_NULL_HANDLE);

        return OL_SUCCESS;
      });

  ASSERT_TRUE(Device1->is_gpu());
  EXPECT_EQ(sycl::default_selector_v(*Device1), 550);
  EXPECT_EQ(sycl::gpu_selector_v(*Device1), 1050);
  EXPECT_EQ(sycl::cpu_selector_v(*Device1), -1);
  EXPECT_EQ(sycl::accelerator_selector_v(*Device1), -1);

  ASSERT_TRUE(Device2->is_cpu());
  EXPECT_EQ(sycl::default_selector_v(*Device2), 350);
  EXPECT_EQ(sycl::gpu_selector_v(*Device2), -1);
  EXPECT_EQ(sycl::cpu_selector_v(*Device2), 1050);
  EXPECT_EQ(sycl::accelerator_selector_v(*Device2), -1);
}

TEST_F(DeviceSelectorScoreTest, TwoGpusOneCompatibleImage) {
  EXPECT_CALL(Mock.get(), olGetDeviceInfo(_, OL_DEVICE_INFO_TYPE, _, _))
      .WillRepeatedly([](ol_device_handle_t /*Device*/,
                         ol_device_info_t /*PropName*/, size_t /*PropSize*/,
                         void *PropValue) -> ol_result_t {
        *static_cast<ol_device_type_t *>(PropValue) = OL_DEVICE_TYPE_GPU;
        return OL_SUCCESS;
      });

  EXPECT_CALL(Mock.get(), olIsValidBinary(_, _, _, _))
      .WillRepeatedly([this](ol_device_handle_t Device,
                             const void * /*ProgData*/, size_t /*ProgDataSize*/,
                             bool *Valid) -> ol_result_t {
        *Valid = (Device == Device2Handle);
        return OL_SUCCESS;
      });

  std::array<llvm::StringRef, 1> KernelNames = {"kernel"};
  ScopedBinaryRegistration Registration{KernelNames};

  EXPECT_EQ(sycl::default_selector_v(*Device1), 550);
  EXPECT_EQ(sycl::default_selector_v(*Device2), 1550);
}

TEST(DeviceSelector, AspectSelector) {
  auto Devices = sycl::device::get_devices();
  ASSERT_FALSE(Devices.empty());

  const sycl::device &Dev = Devices.front();

  const std::vector<sycl::aspect> EmptyAspects{};
  const std::vector<sycl::aspect> RequireGpu{sycl::aspect::gpu};
  const std::vector<sycl::aspect> DenyGpu{sycl::aspect::gpu};

  auto FallbackSelector = sycl::aspect_selector(EmptyAspects, EmptyAspects);
  EXPECT_EQ(FallbackSelector(Dev), sycl::default_selector_v(Dev));

  auto RequireGpuSelector = sycl::aspect_selector(RequireGpu, EmptyAspects);
  EXPECT_EQ(RequireGpuSelector(Dev), 1050);

  auto DenyGpuSelector = sycl::aspect_selector(EmptyAspects, DenyGpu);
  EXPECT_EQ(DenyGpuSelector(Dev), -1);
}

} // namespace
