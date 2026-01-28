#include "tl_templates/ascend/common.h"
#include "acl/acl.h"
#include <runtime/rt_ffts.h>
using namespace Catlass;
using uint = unsigned int;
using uchar = unsigned char;
using ushort = unsigned short;

extern "C" __global__ __aicore__ void main_kernel( GM_ADDR Q_handle,  GM_ADDR K_handle,  GM_ADDR V_handle,  GM_ADDR Output_handle,  GM_ADDR workspace_1_handle,  GM_ADDR workspace_2_handle,  GM_ADDR workspace_3_handle, uint64_t fftsAddr) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
  AscendC::TPipe pipe;

  AscendC::GlobalTensor<half> Q;
  Q.SetGlobalBuffer((__gm__ half*)Q_handle);
  AscendC::GlobalTensor<half> K;
  K.SetGlobalBuffer((__gm__ half*)K_handle);
  AscendC::GlobalTensor<half> V;
  V.SetGlobalBuffer((__gm__ half*)V_handle);
  AscendC::GlobalTensor<half> Output;
  Output.SetGlobalBuffer((__gm__ half*)Output_handle);
  AscendC::GlobalTensor<float> workspace_1;
  workspace_1.SetGlobalBuffer((__gm__ float*)workspace_1_handle);
  AscendC::GlobalTensor<half> workspace_2;
  workspace_2.SetGlobalBuffer((__gm__ half*)workspace_2_handle);
  AscendC::GlobalTensor<float> workspace_3;
  workspace_3.SetGlobalBuffer((__gm__ float*)workspace_3_handle);

  AscendC::TBuf<AscendC::TPosition::A2> ascend_l0a;
  pipe.InitBuffer(ascend_l0a, 65536);
  AscendC::TBuf<AscendC::TPosition::B2> ascend_l0b;
  pipe.InitBuffer(ascend_l0b, 131072);
  AscendC::TBuf<AscendC::TPosition::A1> ascend_l1; pipe.InitBuffer(ascend_l1, 524032);
  AscendC::TBuf<AscendC::TPosition::CO1> ascend_l0c; pipe.InitBuffer(ascend_l0c, 131072);
  AscendC::TBuf<AscendC::TPosition::VECCALC> ascend_ub; pipe.InitBuffer(ascend_ub, 196352);
  pipe.Destroy();
  auto cid = AscendC::GetBlockIdx();
  if ASCEND_IS_AIV {
    cid = cid / 2;
  }
  auto q_l1 = ascend_l1.GetWithOffset<half>(32768, 0);
  auto k_l1 = ascend_l1.GetWithOffset<half>(32768, 65536);
  auto acc_s_l0c = ascend_l0c.GetWithOffset<float>(4096, 0);
  auto acc_s_l1 = ascend_l1.GetWithOffset<half>(4096, 65536);
  auto v_l1 = ascend_l1.GetWithOffset<half>(32768, 73728);
  auto acc_o_l0c = ascend_l0c.GetWithOffset<float>(32768, 0);
  auto acc_o = ascend_ub.GetWithOffset<float>(16384, 0);
  auto sumexp = ascend_ub.GetWithOffset<float>(32, 65536);
  auto m_i = ascend_ub.GetWithOffset<float>(32, 65664);
  auto acc_s_ub = ascend_ub.GetWithOffset<float>(2048, 66048);
  auto m_i_prev = ascend_ub.GetWithOffset<float>(32, 74240);
  auto acc_s_ub_ = ascend_ub.GetWithOffset<float>(2048, 74368);
  auto tmp_ub = ascend_ub.GetWithOffset<uint8_t>(24576, 74368);
  auto sumexp_i_ub = ascend_ub.GetWithOffset<float>(32, 98944);
  auto acc_s_half = ascend_ub.GetWithOffset<half>(2048, 98944);
  auto acc_o_ub = ascend_ub.GetWithOffset<float>(16384, 98944);
  auto acc_o_half = ascend_ub.GetWithOffset<half>(16384, 98944);
  auto vid = AscendC::GetSubBlockIdx();
  if ASCEND_IS_AIC {
    tl::ascend::copy_gm_to_l1<half, 64, 512>(q_l1[0], Q[(cid * 32768)], 512);
    AscendC::PipeBarrier<PIPE_ALL>();
    for (int32_t k = 0; k < 2; ++k) {
      tl::ascend::copy_gm_to_l1<half, 64, 512>(k_l1[0], K[(k * 32768)], 512);
      AscendC::PipeBarrier<PIPE_ALL>();
      tl::ascend::gemm_v0<half, float, 64, 64, 512, false, true>(q_l1[0], k_l1[0], acc_s_l0c[0], ascend_l0a, ascend_l0b, (bool)1);
      AscendC::PipeBarrier<PIPE_ALL>();
      tl::ascend::copy_l0c_to_gm<float, float, layout::RowMajor, 64, 64, 0>(workspace_1[(cid * 4096)], acc_s_l0c[0], 64);
      AscendC::PipeBarrier<PIPE_ALL>();
      AscendC::CrossCoreSetFlag<0x2, PIPE_FIX>(0);
      AscendC::CrossCoreWaitFlag(1);
      AscendC::PipeBarrier<PIPE_ALL>();
      tl::ascend::copy_gm_to_l1<half, 64, 64>(acc_s_l1[0], workspace_2[(cid * 4096)], 64);
      tl::ascend::copy_gm_to_l1<half, 64, 512>(v_l1[0], V[(k * 32768)], 512);
      AscendC::PipeBarrier<PIPE_ALL>();
      tl::ascend::gemm_v0<half, float, 64, 512, 64, false, false>(acc_s_l1[0], v_l1[0], acc_o_l0c[0], ascend_l0a, ascend_l0b, (bool)1);
      AscendC::PipeBarrier<PIPE_ALL>();
      tl::ascend::copy_l0c_to_gm<float, float, layout::RowMajor, 64, 512, 0>(workspace_3[(cid * 32768)], acc_o_l0c[0], 512);
      AscendC::PipeBarrier<PIPE_ALL>();
      AscendC::CrossCoreSetFlag<0x2, PIPE_FIX>(2);
      AscendC::CrossCoreWaitFlag(3);
    }
  }
  if ASCEND_IS_AIV {
    AscendC::Duplicate<float>(acc_o[0], 0.000000e+00f, 16384);
    AscendC::Duplicate<float>(sumexp[0], 0.000000e+00f, 32);
    AscendC::Duplicate<float>(m_i[0], -1073741824, 32);
    AscendC::PipeBarrier<PIPE_ALL>();
    for (int32_t _k = 0; _k < 2; ++_k) {
      AscendC::Duplicate<float>(acc_s_ub[0], 0.000000e+00f, 2048);
      AscendC::PipeBarrier<PIPE_ALL>();
      tl::ascend::copy_ub_to_ub<float, float, 32>(m_i_prev[0], m_i[0]);
      AscendC::PipeBarrier<PIPE_ALL>();
      AscendC::CrossCoreWaitFlag(0);
      tl::ascend::copy_gm_to_ub<float, 64, 32>(acc_s_ub_[0], workspace_1[((cid * 4096) + (vid * 2048))], 64);
      AscendC::PipeBarrier<PIPE_ALL>();
      AscendC::Add(acc_s_ub[0], acc_s_ub[0], acc_s_ub_[0], 2048);
      AscendC::PipeBarrier<PIPE_ALL>();
      AscendC::Muls(acc_s_ub[0], acc_s_ub[0], 4.419417e-02f, 2048);
      AscendC::PipeBarrier<PIPE_ALL>();
      tl::ascend::reduce_max<float, 32, 64, -1>(m_i[0], acc_s_ub[0], tmp_ub[0]);
      AscendC::PipeBarrier<PIPE_ALL>();
      AscendC::Max(m_i[0], m_i[0], m_i_prev[0], 32);
      AscendC::PipeBarrier<PIPE_ALL>();
      AscendC::Sub(m_i_prev[0], m_i_prev[0], m_i[0], 32);
      AscendC::PipeBarrier<PIPE_ALL>();
      AscendC::Exp(m_i_prev[0], m_i_prev[0], 32);
      AscendC::PipeBarrier<PIPE_ALL>();
      for (int32_t h_i = 0; h_i < 32; ++h_i) {
        AscendC::PipeBarrier<PIPE_ALL>();
        AscendC::PipeBarrier<PIPE_ALL>();
        auto m_i_scalar = m_i.GetValue(h_i);
        AscendC::Adds(acc_s_ub[(h_i * 64)], acc_s_ub[(h_i * 64)], -m_i_scalar, 64);
        AscendC::PipeBarrier<PIPE_ALL>();
      }
      AscendC::Exp(acc_s_ub[0], acc_s_ub[0], 2048);
      AscendC::PipeBarrier<PIPE_ALL>();
      tl::ascend::reduce_sum<float, 32, 64, -1>(sumexp_i_ub[0], acc_s_ub[0], tmp_ub[0]);
      AscendC::PipeBarrier<PIPE_ALL>();
      AscendC::Mul(sumexp[0], sumexp[0], m_i_prev[0], 32);
      AscendC::PipeBarrier<PIPE_ALL>();
      AscendC::Add(sumexp[0], sumexp[0], sumexp_i_ub[0], 32);
      AscendC::PipeBarrier<PIPE_ALL>();
      for (int32_t h_i_1 = 0; h_i_1 < 32; ++h_i_1) {
        AscendC::PipeBarrier<PIPE_ALL>();
        AscendC::PipeBarrier<PIPE_ALL>();
        auto m_i_prev_scalar = m_i_prev.GetValue(h_i_1);
        AscendC::Muls(acc_o[(h_i_1 * 512)], acc_o[(h_i_1 * 512)], m_i_prev_scalar, 512);
        AscendC::PipeBarrier<PIPE_ALL>();
      }
      tl::ascend::copy_ub_to_ub<half, float, 2048>(acc_s_half[0], acc_s_ub[0]);
      AscendC::PipeBarrier<PIPE_ALL>();
      tl::ascend::copy_ub_to_gm<half, 64, 32>(workspace_2[((cid * 4096) + (vid * 2048))], acc_s_half[0], 64);
      AscendC::PipeBarrier<PIPE_ALL>();
      AscendC::CrossCoreSetFlag<0x2, PIPE_MTE3>(1);
      AscendC::CrossCoreWaitFlag(2);
      AscendC::PipeBarrier<PIPE_ALL>();
      tl::ascend::copy_gm_to_ub<float, 512, 32>(acc_o_ub[0], workspace_3[((cid * 32768) + (vid * 16384))], 512);
      AscendC::PipeBarrier<PIPE_ALL>();
      AscendC::Add(acc_o[0], acc_o[0], acc_o_ub[0], 16384);
      AscendC::PipeBarrier<PIPE_ALL>();
      AscendC::CrossCoreSetFlag<0x2, PIPE_V>(3);
      AscendC::PipeBarrier<PIPE_ALL>();
    }
    for (int32_t h_i_2 = 0; h_i_2 < 32; ++h_i_2) {
      AscendC::PipeBarrier<PIPE_ALL>();
      AscendC::PipeBarrier<PIPE_ALL>();
      auto sumexp_scalar = 1.0f / sumexp.GetValue(h_i_2);
      AscendC::Muls(acc_o[(h_i_2 * 512)], acc_o[(h_i_2 * 512)], sumexp_scalar, 512);
      AscendC::PipeBarrier<PIPE_ALL>();
    }
    tl::ascend::copy_ub_to_ub<half, float, 16384>(acc_o_half[0], acc_o[0]);
    AscendC::PipeBarrier<PIPE_ALL>();
    tl::ascend::copy_ub_to_gm<half, 512, 32>(Output[((cid * 32768) + (vid * 16384))], acc_o_half[0], 512);
  }
}

void main_kernel_tiling() {
}

extern "C" void call(uint8_t* Q_handle, uint8_t* K_handle, uint8_t* V_handle, uint8_t* Output_handle, uint8_t* workspace_1_handle, uint8_t* workspace_2_handle, uint8_t* workspace_3_handle, aclrtStream stream) {
  uint32_t fftsLen{0};
  uint64_t fftsAddr{0};
  rtGetC2cCtrlAddr(&fftsAddr, &fftsLen);
  main_kernel_tiling();
  main_kernel<<<2, nullptr, stream>>>(Q_handle, K_handle, V_handle, Output_handle, workspace_1_handle, workspace_2_handle, workspace_3_handle, fftsAddr);
}