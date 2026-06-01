#include "Opdefine.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) 
{
    m.def("fuse_silu_bwd", &fused_SiLU_Bwd, "fuse_silu_bwd");
    m.def("fuse_silu_grad_bwd", &fused_SiLU_Grad_Bwd, "fuse_silu_grad_bwd"); 
    m.def("quant_linear_w8a32", &quant_linear_w8a32, "quant_linear_w8a32");
    m.def("quant_linear_w8a8_static_wmma", &quant_linear_w8a8_static_wmma, "quant_linear_w8a8_static_wmma");
    m.def("quant_linear_w8a8_static_wmma_dual_gated_tail_n128", &quant_linear_w8a8_static_wmma_dual_gated_tail_n128, "quant_linear_w8a8_static_wmma_dual_gated_tail_n128");
    m.def("quant_linear_w8a8_static_wmma_dual_gated_tail_n128_fast", &quant_linear_w8a8_static_wmma_dual_gated_tail_n128_fast, "W8A8 dual gated tail n128 fast wrapper");
    m.def("quant_linear_w8a8_static_wmma_dual_gated_tail_n128_with_pre", &quant_linear_w8a8_static_wmma_dual_gated_tail_n128_with_pre, "W8A8 dual gated tail n128 with pre-tail outputs");
    m.def("quant_linear_w8a8_static_cutlass", &quant_linear_w8a8_static_cutlass, "quant_linear_w8a8_static_cutlass");
    m.def("quant_linear_w8a8_static_cutlass_dual", &quant_linear_w8a8_static_cutlass_dual, "quant_linear_w8a8_static_cutlass_dual");
    m.def("quant_linear_w8a8_static_cutlass_dual_gated_tail", &quant_linear_w8a8_static_cutlass_dual_gated_tail, "quant_linear_w8a8_static_cutlass_dual_gated_tail");
    m.def("fp32_gated_tail_forward_n128",
          &fp32_gated_tail_forward_n128,
          "P77 FP32 gated tail forward n128");
    m.def("quant_linear_w8a8_static_cutlass_grouped_dual", &quant_linear_w8a8_static_cutlass_grouped_dual, "quant_linear_w8a8_static_cutlass_grouped_dual");
    m.def("quant_linear_w8a8_static_cutlass_grouped_pair", &quant_linear_w8a8_static_cutlass_grouped_pair, "quant_linear_w8a8_static_cutlass_grouped_pair");
    m.def("quant_ffn_w8a8_static_cutlass_grouped_pair", &quant_ffn_w8a8_static_cutlass_grouped_pair, "quant_ffn_w8a8_static_cutlass_grouped_pair");
    m.def("quant_linear_w8a8_static_wmma_dual", &quant_linear_w8a8_static_wmma_dual, "quant_linear_w8a8_static_wmma_dual");
    m.def("target_attention_sum_forward", &target_attention_sum_forward, "target_attention_sum_forward");
    m.def("target_attention_sum_backward", &target_attention_sum_backward, "target_attention_sum_backward");
    m.def("directed2undirected_average_forward", &directed2undirected_average_forward, "directed2undirected_average_forward");
    m.def("directed2undirected_average_backward", &directed2undirected_average_backward, "directed2undirected_average_backward");
    m.def("edge_vectors_forward", &edge_vectors_forward, "edge_vectors_forward");
    m.def("edge_vectors_backward", &edge_vectors_backward, "edge_vectors_backward");
    m.def("force_stress_from_edge_vectors", &force_stress_from_edge_vectors, "force_stress_from_edge_vectors");
    m.def("fused_line_attention_forward", &fused_line_attention_forward, "fused_line_attention_forward");
    m.def("fused_line_attention_forward_v2", &fused_line_attention_forward_v2, "P82 fused_line_attention_forward_v2");
    m.def("fused_line_attention_max_atomic", &fused_line_attention_max_atomic, "P82B fused_line_attention_max_atomic");
    m.def("fused_line_attention_single_max_atomic",
          &fused_line_attention_single_max_atomic,
          "P82B fused_line_attention_single_max_atomic");
    m.def("fused_line_attention_forward_with_max",
          &fused_line_attention_forward_with_max,
          "P82C fused_line_attention_forward_with_max");
    m.def("fused_line_attention_forward_target_offsets",
          &fused_line_attention_forward_target_offsets,
          "P83B fused_line_attention_forward_target_offsets");
    m.def("fused_line_attention_node_input_forward_target_offsets",
          &fused_line_attention_node_input_forward_target_offsets,
          "P83C fused_line_attention_node_input_forward_target_offsets");
    m.def("fused_line_attention_backward", &fused_line_attention_backward, "fused_line_attention_backward");
    m.def("fused_line_attention_backward_with_edge_direct",
          &fused_line_attention_backward_with_edge_direct,
          "P106 fused_line_attention_backward with direct edge grad accumulation");
    m.def("fused_line_attention_values_backward_with_edge_direct",
          &fused_line_attention_values_backward_with_edge_direct,
          "P107 fused_line_attention values-only backward with direct edge grad accumulation");
    m.def("line_edge_gather_cat_forward", &line_edge_gather_cat_forward, "line_edge_gather_cat_forward");
    m.def("line_edge_cat_grad_scatter_backward", &line_edge_cat_grad_scatter_backward, "line_edge_cat_grad_scatter_backward");
    m.def("line_node_triple_cat_forward",
          &line_node_triple_cat_forward,
          "P80 line-node triple cat forward");
    m.def("line_node_triple_cat_backward",
          &line_node_triple_cat_backward,
          "P80 line-node triple cat backward");
    m.def("directed_edge_gather_cat_forward", &directed_edge_gather_cat_forward, "directed_edge_gather_cat_forward");
    m.def("directed_edge_cat_grad_scatter_backward",
          &directed_edge_cat_grad_scatter_backward,
          "directed_edge_cat_grad_scatter_backward");
    m.def("directed_edge_silu_project_grad_scatter_backward_tile32",
          &directed_edge_silu_project_grad_scatter_backward_tile32,
          "AT-CUDA1B directed edge fused SiLU/project/scatter backward");
    m.def("refine_line_edge_gather_cat_forward",
          &refine_line_edge_gather_cat_forward,
          "refine_line_edge_gather_cat_forward");
    m.def("refine_line_edge_cat_grad_scatter_backward",
          &refine_line_edge_cat_grad_scatter_backward,
          "refine_line_edge_cat_grad_scatter_backward");
    m.def("refine_line_project_grad_scatter_backward_tile32",
          &refine_line_project_grad_scatter_backward_tile32,
          "refine_line_project_grad_scatter_backward_tile32");
    m.def("refine_line_project_dual_grad_scatter_add_tile32",
          &refine_line_project_dual_grad_scatter_add_tile32,
          "refine_line project dual core/gate grad scatter-add into existing grads");
    m.def("refine_line_first_silu_forward",
          &refine_line_first_silu_forward,
          "P60 refine-line fused first projection + SiLU forward");
    m.def("refine_line_first_silu_forward_acts",
          &refine_line_first_silu_forward_acts,
          "P61 refine-line fused first projection + SiLU activations-only forward");
    m.def("refine_line_first_tail_w8a8_forward",
          &refine_line_first_tail_w8a8_forward,
          "P61B refine-line fused first projection + W8A8 gated tail forward");
    m.def("refine_line_first_tail_w8a8_forward_with_pre",
          &refine_line_first_tail_w8a8_forward_with_pre,
          "P66 refine-line fused first projection + W8A8 gated tail forward with saved raw/pre tensors");
    m.def("refine_line_first_tail_smooth_reduce_w8a8_forward_with_pre",
          &refine_line_first_tail_smooth_reduce_w8a8_forward_with_pre,
          "P69 refine-line fused first projection + W8A8 gated tail + smooth reduce forward with saved raw/pre tensors");
    m.def("refine_line_first_tail_w8a8_packed_forward",
          &refine_line_first_tail_w8a8_packed_forward,
          "P62 refine-line fused first projection + int8 activation packed W8A8 gated tail forward");
    m.def("refine_line_first_silu_backward",
          &refine_line_first_silu_backward,
          "P60 refine-line fused first projection + SiLU input-gradient backward");
    m.def("refine_line_first_silu_backward_packed",
          &refine_line_first_silu_backward_packed,
          "P65B refine-line packed first projection + SiLU input-gradient backward");
    m.def("refine_line_smooth_reduce_forward",
          &refine_line_smooth_reduce_forward,
          "refine_line_smooth_reduce_forward");
    m.def("refine_line_smooth_reduce_backward",
          &refine_line_smooth_reduce_backward,
          "refine_line_smooth_reduce_backward");
    m.def("refine_line_smooth_reduce_sorted_forward",
          &refine_line_smooth_reduce_sorted_forward,
          "refine_line_smooth_reduce_sorted_forward");
    m.def("refine_line_smooth_reduce_sorted_backward",
          &refine_line_smooth_reduce_sorted_backward,
          "refine_line_smooth_reduce_sorted_backward");
    m.def("refine_line_edge_smooth_w8a8_backward_n128",
          &refine_line_edge_smooth_w8a8_backward_n128,
          "P64 refine-line smooth reduce + W8A8 tail + first projection fused backward");
    m.def("refine_line_smooth_w8a8_tail_input_grad_backward_n128",
          &refine_line_smooth_w8a8_tail_input_grad_backward_n128,
          "P64B refine-line smooth reduce + W8A8 tail input-grad fused backward");
    m.def("refine_line_smooth_w8a8_tail_actgrad_backward_n128",
          &refine_line_smooth_w8a8_tail_actgrad_backward_n128,
          "P65B refine-line smooth reduce + W8A8 packed tail-grad backward");
    m.def("refine_line_smooth_tail_first_silu_backward_tile",
          &refine_line_smooth_tail_first_silu_backward_tile,
          "P65 refine-line tiled smooth/tail/first-projection fused backward");
    m.def("line_edge_project_grad_scatter_backward",
          &line_edge_project_grad_scatter_backward,
          "line_edge_project_grad_scatter_backward");
    m.def("line_edge_project_grad_scatter_backward_tiled",
          &line_edge_project_grad_scatter_backward_tiled,
          "line_edge_project_grad_scatter_backward_tiled");
    m.def("line_edge_project_grad_scatter_backward_tile32",
          &line_edge_project_grad_scatter_backward_tile32,
          "line_edge_project_grad_scatter_backward_tile32");
    m.def("line_edge_silu_project_grad_scatter_backward_tile32",
          &line_edge_silu_project_grad_scatter_backward_tile32,
          "line_edge_silu_project_grad_scatter_backward_tile32");
    m.def("line_edge_silu_project_alpha_grad_scatter_backward_tile32",
          &line_edge_silu_project_alpha_grad_scatter_backward_tile32,
          "line_edge_silu_project_alpha_grad_scatter_backward_tile32");
    m.def("line_edge_silu_project_alpha_grad_scatter_backward_alpha_tile32",
          &line_edge_silu_project_alpha_grad_scatter_backward_alpha_tile32,
          "P108 line_edge_silu_project_alpha_grad_scatter_backward alpha-tiled");
    m.def("line_edge_silu_project_alpha_grad_scatter_backward_dense_gemm",
          &line_edge_silu_project_alpha_grad_scatter_backward_dense_gemm,
          "P108 line_edge_silu_project_alpha_grad_scatter_backward dense-gemm");
    m.def("line_edge_silu_project_alpha_grad_scatter_backward_target_reduce_tile32",
          &line_edge_silu_project_alpha_grad_scatter_backward_target_reduce_tile32,
          "P108 line_edge_silu_project_alpha_grad_scatter_backward target-reduce");
    m.def("line_edge_silu_project_alpha_attention_grad_scatter_backward_tile32",
          &line_edge_silu_project_alpha_attention_grad_scatter_backward_tile32,
          "P107 line-edge fused first projection + attention alpha projection input grad + scatter");
    m.def("line_edge_w8a8_tail_project_scatter_backward_n128",
          &line_edge_w8a8_tail_project_scatter_backward_n128,
          "line_edge_w8a8_tail_project_scatter_backward_n128");
    m.def("input_grad_only_gated_tail_backward", &input_grad_only_gated_tail_backward, "input_grad_only_gated_tail_backward");
    m.def("input_grad_only_gated_tail_backward_n128_v2",
          &input_grad_only_gated_tail_backward_n128_v2,
          "P89B input_grad_only_gated_tail_backward_n128_v2");
    m.def("gated_tail_second_silu_input_grad_macro",
          &gated_tail_second_silu_input_grad_macro,
          "GatedMLP tail backward + second Linear input-grad + pre-second SiLU grad macro");
    m.def("gated_tail_second_silu_residual_input_grad_macro",
          &gated_tail_second_silu_residual_input_grad_macro,
          "GatedMLP second-tail macro plus residual input/res-weight grad");
    m.def("input_grad_only_gated_tail_backward_stack",
          &input_grad_only_gated_tail_backward_stack,
          "input_grad_only_gated_tail_backward_stack");
    m.def("param_grad_gated_tail_backward", &param_grad_gated_tail_backward, "param_grad_gated_tail_backward");
    m.def("w8a8_dual_gated_tail_input_grad_backward_n128",
          &w8a8_dual_gated_tail_input_grad_backward_n128,
          "W8A8 dual gated-tail fused input-grad backward n128");
    m.def("w8a8_dual_gated_tail_saved_pre_input_grad_backward_n128",
          &w8a8_dual_gated_tail_saved_pre_input_grad_backward_n128,
          "W8A8 dual gated-tail saved-pre fused input-grad backward n128");
    m.def("w8a8_dual_gated_tail_saved_pre_dq_input_grad_backward_n128",
          &w8a8_dual_gated_tail_saved_pre_dq_input_grad_backward_n128,
          "W8A8 dual gated-tail saved-pre dq fused input-grad backward n128");
    m.def("w8a8_saved_pre_backward_driver",
          &w8a8_saved_pre_backward_driver,
          "P91A W8A8 saved-pre backward driver using tail backward + ATen linear");
    m.def("w8a8_saved_pre_backward_group4_driver",
          &w8a8_saved_pre_backward_group4_driver,
          "P95B W8A8 saved-pre backward group4 driver using tail backward + ATen linear");
    m.def("w8a8_dual_input_grad_matmul_n128_tiled",
          &w8a8_dual_input_grad_matmul_n128_tiled,
          "W8A8 dual input-grad tiled matmul n128");
    m.def("w8a8_dual_input_grad_matmul_n128_tiled_m32n8",
          &w8a8_dual_input_grad_matmul_n128_tiled_m32n8,
          "W8A8 dual input-grad tiled matmul n128 m32n8");
    m.def("w8a8_dual_input_grad_matmul_n128_cublas_grouped",
          &w8a8_dual_input_grad_matmul_n128_cublas_grouped,
          "W8A8 dual input-grad cuBLAS grouped matmul n128");
    m.def("w8a8_dual_input_grad_matmul_n128_cublas_pair",
          &w8a8_dual_input_grad_matmul_n128_cublas_pair,
          "W8A8 dual input-grad cuBLAS pair matmul n128");
    m.def("w8a8_group8_input_grad_matmul_n128_cublas_grouped",
          &w8a8_group8_input_grad_matmul_n128_cublas_grouped,
          "P97B W8A8 group8 input-grad cuBLAS grouped matmul n128");
    m.def("two_linear_silu_input_grad_backward_n128",
          &two_linear_silu_input_grad_backward_n128,
          "Two-linear SiLU MLP fused input-grad backward n128");
}

TORCH_LIBRARY(matris_op, m)
{
    m.def("fuse_silu_bwd", &fused_SiLU_Bwd);
    m.def("fuse_silu_grad_bwd", &fused_SiLU_Grad_Bwd);  
    m.def("quant_linear_w8a32", &quant_linear_w8a32);
    m.def("quant_linear_w8a8_static_wmma", &quant_linear_w8a8_static_wmma);
    m.def("quant_linear_w8a8_static_wmma_dual_gated_tail_n128", &quant_linear_w8a8_static_wmma_dual_gated_tail_n128);
    m.def("quant_linear_w8a8_static_wmma_dual_gated_tail_n128_fast", &quant_linear_w8a8_static_wmma_dual_gated_tail_n128_fast);
    m.def("quant_linear_w8a8_static_wmma_dual_gated_tail_n128_with_pre", &quant_linear_w8a8_static_wmma_dual_gated_tail_n128_with_pre);
    m.def("quant_linear_w8a8_static_cutlass", &quant_linear_w8a8_static_cutlass);
    m.def("quant_linear_w8a8_static_cutlass_dual", &quant_linear_w8a8_static_cutlass_dual);
    m.def("quant_linear_w8a8_static_cutlass_dual_gated_tail", &quant_linear_w8a8_static_cutlass_dual_gated_tail);
    m.def("fp32_gated_tail_forward_n128", &fp32_gated_tail_forward_n128);
    m.def("quant_linear_w8a8_static_cutlass_grouped_dual", &quant_linear_w8a8_static_cutlass_grouped_dual);
    m.def("quant_linear_w8a8_static_cutlass_grouped_pair", &quant_linear_w8a8_static_cutlass_grouped_pair);
    m.def("quant_ffn_w8a8_static_cutlass_grouped_pair", &quant_ffn_w8a8_static_cutlass_grouped_pair);
    m.def("quant_linear_w8a8_static_wmma_dual", &quant_linear_w8a8_static_wmma_dual);
    m.def("target_attention_sum_forward", &target_attention_sum_forward);
    m.def("target_attention_sum_backward", &target_attention_sum_backward);
    m.def("directed2undirected_average_forward", &directed2undirected_average_forward);
    m.def("directed2undirected_average_backward", &directed2undirected_average_backward);
    m.def("edge_vectors_forward", &edge_vectors_forward);
    m.def("edge_vectors_backward", &edge_vectors_backward);
    m.def("force_stress_from_edge_vectors", &force_stress_from_edge_vectors);
    m.def("fused_line_attention_forward", &fused_line_attention_forward);
    m.def("fused_line_attention_forward_v2", &fused_line_attention_forward_v2);
    m.def("fused_line_attention_max_atomic", &fused_line_attention_max_atomic);
    m.def("fused_line_attention_single_max_atomic", &fused_line_attention_single_max_atomic);
    m.def("fused_line_attention_forward_with_max", &fused_line_attention_forward_with_max);
    m.def("fused_line_attention_forward_target_offsets", &fused_line_attention_forward_target_offsets);
    m.def("fused_line_attention_node_input_forward_target_offsets",
          &fused_line_attention_node_input_forward_target_offsets);
    m.def("fused_line_attention_backward", &fused_line_attention_backward);
    m.def("fused_line_attention_backward_with_edge_direct",
          &fused_line_attention_backward_with_edge_direct);
    m.def("fused_line_attention_values_backward_with_edge_direct",
          &fused_line_attention_values_backward_with_edge_direct);
    m.def("line_edge_gather_cat_forward", &line_edge_gather_cat_forward);
    m.def("line_edge_cat_grad_scatter_backward", &line_edge_cat_grad_scatter_backward);
    m.def("line_node_triple_cat_forward", &line_node_triple_cat_forward);
    m.def("line_node_triple_cat_backward", &line_node_triple_cat_backward);
    m.def("directed_edge_gather_cat_forward", &directed_edge_gather_cat_forward);
    m.def("directed_edge_cat_grad_scatter_backward", &directed_edge_cat_grad_scatter_backward);
    m.def("directed_edge_silu_project_grad_scatter_backward_tile32",
          &directed_edge_silu_project_grad_scatter_backward_tile32);
    m.def("refine_line_edge_gather_cat_forward", &refine_line_edge_gather_cat_forward);
    m.def("refine_line_edge_cat_grad_scatter_backward", &refine_line_edge_cat_grad_scatter_backward);
    m.def("refine_line_project_grad_scatter_backward_tile32",
          &refine_line_project_grad_scatter_backward_tile32);
    m.def("refine_line_project_dual_grad_scatter_add_tile32",
          &refine_line_project_dual_grad_scatter_add_tile32);
    m.def("refine_line_first_silu_forward", &refine_line_first_silu_forward);
    m.def("refine_line_first_silu_forward_acts", &refine_line_first_silu_forward_acts);
    m.def("refine_line_first_tail_w8a8_forward", &refine_line_first_tail_w8a8_forward);
    m.def("refine_line_first_tail_w8a8_forward_with_pre", &refine_line_first_tail_w8a8_forward_with_pre);
    m.def("refine_line_first_tail_smooth_reduce_w8a8_forward_with_pre",
          &refine_line_first_tail_smooth_reduce_w8a8_forward_with_pre);
    m.def("refine_line_first_tail_w8a8_packed_forward", &refine_line_first_tail_w8a8_packed_forward);
    m.def("refine_line_first_silu_backward", &refine_line_first_silu_backward);
    m.def("refine_line_first_silu_backward_packed", &refine_line_first_silu_backward_packed);
    m.def("refine_line_smooth_reduce_forward", &refine_line_smooth_reduce_forward);
    m.def("refine_line_smooth_reduce_backward", &refine_line_smooth_reduce_backward);
    m.def("refine_line_smooth_reduce_sorted_forward", &refine_line_smooth_reduce_sorted_forward);
    m.def("refine_line_smooth_reduce_sorted_backward", &refine_line_smooth_reduce_sorted_backward);
    m.def("refine_line_edge_smooth_w8a8_backward_n128",
          &refine_line_edge_smooth_w8a8_backward_n128);
    m.def("refine_line_smooth_w8a8_tail_input_grad_backward_n128",
          &refine_line_smooth_w8a8_tail_input_grad_backward_n128);
    m.def("refine_line_smooth_w8a8_tail_actgrad_backward_n128",
          &refine_line_smooth_w8a8_tail_actgrad_backward_n128);
    m.def("refine_line_smooth_tail_first_silu_backward_tile",
          &refine_line_smooth_tail_first_silu_backward_tile);
    m.def("line_edge_project_grad_scatter_backward", &line_edge_project_grad_scatter_backward);
    m.def("line_edge_project_grad_scatter_backward_tiled", &line_edge_project_grad_scatter_backward_tiled);
    m.def("line_edge_project_grad_scatter_backward_tile32", &line_edge_project_grad_scatter_backward_tile32);
    m.def("line_edge_silu_project_grad_scatter_backward_tile32",
          &line_edge_silu_project_grad_scatter_backward_tile32);
    m.def("line_edge_silu_project_alpha_grad_scatter_backward_tile32",
          &line_edge_silu_project_alpha_grad_scatter_backward_tile32);
    m.def("line_edge_silu_project_alpha_grad_scatter_backward_alpha_tile32",
          &line_edge_silu_project_alpha_grad_scatter_backward_alpha_tile32);
    m.def("line_edge_silu_project_alpha_grad_scatter_backward_dense_gemm",
          &line_edge_silu_project_alpha_grad_scatter_backward_dense_gemm);
    m.def("line_edge_silu_project_alpha_grad_scatter_backward_target_reduce_tile32",
          &line_edge_silu_project_alpha_grad_scatter_backward_target_reduce_tile32);
    m.def("line_edge_silu_project_alpha_attention_grad_scatter_backward_tile32",
          &line_edge_silu_project_alpha_attention_grad_scatter_backward_tile32);
    m.def("line_edge_w8a8_tail_project_scatter_backward_n128",
          &line_edge_w8a8_tail_project_scatter_backward_n128);
    m.def("input_grad_only_gated_tail_backward", &input_grad_only_gated_tail_backward);
    m.def("input_grad_only_gated_tail_backward_n128_v2", &input_grad_only_gated_tail_backward_n128_v2);
    m.def("gated_tail_second_silu_input_grad_macro", &gated_tail_second_silu_input_grad_macro);
    m.def("gated_tail_second_silu_residual_input_grad_macro",
          &gated_tail_second_silu_residual_input_grad_macro);
    m.def("input_grad_only_gated_tail_backward_stack", &input_grad_only_gated_tail_backward_stack);
    m.def("param_grad_gated_tail_backward", &param_grad_gated_tail_backward);
    m.def("w8a8_dual_gated_tail_input_grad_backward_n128", &w8a8_dual_gated_tail_input_grad_backward_n128);
    m.def("w8a8_dual_gated_tail_saved_pre_input_grad_backward_n128",
          &w8a8_dual_gated_tail_saved_pre_input_grad_backward_n128);
    m.def("w8a8_dual_gated_tail_saved_pre_dq_input_grad_backward_n128",
          &w8a8_dual_gated_tail_saved_pre_dq_input_grad_backward_n128);
    m.def("w8a8_saved_pre_backward_driver", &w8a8_saved_pre_backward_driver);
    m.def("w8a8_dual_input_grad_matmul_n128_tiled", &w8a8_dual_input_grad_matmul_n128_tiled);
    m.def("w8a8_dual_input_grad_matmul_n128_tiled_m32n8",
          &w8a8_dual_input_grad_matmul_n128_tiled_m32n8);
    m.def("w8a8_dual_input_grad_matmul_n128_cublas_grouped",
          &w8a8_dual_input_grad_matmul_n128_cublas_grouped);
    m.def("w8a8_dual_input_grad_matmul_n128_cublas_pair",
          &w8a8_dual_input_grad_matmul_n128_cublas_pair);
    m.def("w8a8_group8_input_grad_matmul_n128_cublas_grouped",
          &w8a8_group8_input_grad_matmul_n128_cublas_grouped);
    m.def("two_linear_silu_input_grad_backward_n128", &two_linear_silu_input_grad_backward_n128);
	}
