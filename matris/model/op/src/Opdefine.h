#ifndef OP_SRC_OPDECLARE_H_
#define OP_SRC_OPDECLARE_H_

#include <torch/extension.h>


torch::Tensor fused_SiLU_Bwd(const torch::Tensor &dgrad, const torch::Tensor &input);

std::vector<torch::Tensor> fused_SiLU_Grad_Bwd(const torch::Tensor &grad_grad_input, const torch::Tensor &grad_output,
                                                const torch::Tensor &input);

torch::Tensor quant_linear_w8a32(const torch::Tensor &input,
                                 const torch::Tensor &q_weight,
                                 const torch::Tensor &scale,
                                 const torch::Tensor &bias,
                                 bool has_bias);

torch::Tensor quant_linear_w8a8_static_wmma(const torch::Tensor &input,
                                            const torch::Tensor &q_weight,
                                            const torch::Tensor &weight_scale,
                                            const torch::Tensor &activation_scale,
                                            const torch::Tensor &bias,
                                            bool has_bias);

torch::Tensor quant_linear_w8a8_static_cutlass(const torch::Tensor &input,
                                               const torch::Tensor &q_weight,
                                               const torch::Tensor &weight_scale,
                                               const torch::Tensor &activation_scale,
                                               const torch::Tensor &bias,
                                               bool has_bias);

torch::Tensor quant_linear_w8a8_static_wmma_dual_gated_tail_n128(const torch::Tensor &core_input,
                                                                 const torch::Tensor &gate_input,
                                                                 const torch::Tensor &core_q_weight,
                                                                 const torch::Tensor &gate_q_weight,
                                                                 const torch::Tensor &core_weight_scale,
                                                                 const torch::Tensor &gate_weight_scale,
                                                                 const torch::Tensor &core_activation_scale,
                                                                 const torch::Tensor &gate_activation_scale,
                                                                 const torch::Tensor &core_bias,
                                                                 const torch::Tensor &gate_bias,
                                                                 bool core_has_bias,
                                                                 bool gate_has_bias,
                                                                 const torch::Tensor &core_norm_weight,
                                                                 const torch::Tensor &core_norm_bias,
                                                                 const torch::Tensor &gate_norm_weight,
                                                                 const torch::Tensor &gate_norm_bias,
                                                                 double eps);

torch::Tensor quant_linear_w8a8_static_wmma_dual_gated_tail_n128_fast(const torch::Tensor &core_input,
                                                                      const torch::Tensor &gate_input,
                                                                      const torch::Tensor &core_q_weight,
                                                                      const torch::Tensor &gate_q_weight,
                                                                      const torch::Tensor &core_weight_scale,
                                                                      const torch::Tensor &gate_weight_scale,
                                                                      const torch::Tensor &core_activation_scale,
                                                                      const torch::Tensor &gate_activation_scale,
                                                                      const torch::Tensor &core_bias,
                                                                      const torch::Tensor &gate_bias,
                                                                      bool core_has_bias,
                                                                      bool gate_has_bias,
                                                                      const torch::Tensor &core_norm_weight,
                                                                      const torch::Tensor &core_norm_bias,
                                                                      const torch::Tensor &gate_norm_weight,
                                                                      const torch::Tensor &gate_norm_bias,
                                                                      double eps);

std::vector<torch::Tensor> quant_linear_w8a8_static_wmma_dual_gated_tail_n128_with_pre(
    const torch::Tensor &core_input,
    const torch::Tensor &gate_input,
    const torch::Tensor &core_q_weight,
    const torch::Tensor &gate_q_weight,
    const torch::Tensor &core_weight_scale,
    const torch::Tensor &gate_weight_scale,
    const torch::Tensor &core_activation_scale,
    const torch::Tensor &gate_activation_scale,
    const torch::Tensor &core_bias,
    const torch::Tensor &gate_bias,
    bool core_has_bias,
    bool gate_has_bias,
    const torch::Tensor &core_norm_weight,
    const torch::Tensor &core_norm_bias,
    const torch::Tensor &gate_norm_weight,
    const torch::Tensor &gate_norm_bias,
    double eps);

std::vector<torch::Tensor> quant_linear_w8a8_static_cutlass_dual(const torch::Tensor &core_input,
                                                                 const torch::Tensor &gate_input,
                                                                 const torch::Tensor &core_q_weight,
                                                                 const torch::Tensor &gate_q_weight,
                                                                 const torch::Tensor &core_weight_scale,
                                                                 const torch::Tensor &gate_weight_scale,
                                                                 const torch::Tensor &core_activation_scale,
                                                                 const torch::Tensor &gate_activation_scale,
                                                                 const torch::Tensor &core_bias,
                                                                 const torch::Tensor &gate_bias,
                                                                 bool core_has_bias,
                                                                 bool gate_has_bias);

torch::Tensor quant_linear_w8a8_static_cutlass_dual_gated_tail(const torch::Tensor &core_input,
                                                               const torch::Tensor &gate_input,
                                                               const torch::Tensor &core_q_weight,
                                                               const torch::Tensor &gate_q_weight,
                                                               const torch::Tensor &core_weight_scale,
                                                               const torch::Tensor &gate_weight_scale,
                                                               const torch::Tensor &core_activation_scale,
                                                               const torch::Tensor &gate_activation_scale,
                                                               const torch::Tensor &core_bias,
                                                               const torch::Tensor &gate_bias,
                                                               bool core_has_bias,
                                                               bool gate_has_bias,
                                                               const torch::Tensor &core_norm_weight,
                                                               const torch::Tensor &core_norm_bias,
                                                               const torch::Tensor &gate_norm_weight,
                                                               const torch::Tensor &gate_norm_bias,
                                                               double eps);

torch::Tensor fp32_gated_tail_forward_n128(const torch::Tensor &core,
                                           const torch::Tensor &gate,
                                           const torch::Tensor &core_norm_weight,
                                           const torch::Tensor &core_norm_bias,
                                           const torch::Tensor &gate_norm_weight,
                                           const torch::Tensor &gate_norm_bias,
                                           double eps);

std::vector<torch::Tensor> quant_linear_w8a8_static_cutlass_grouped_dual(const torch::Tensor &core_input,
                                                                         const torch::Tensor &gate_input,
                                                                         const torch::Tensor &core_q_weight,
                                                                         const torch::Tensor &gate_q_weight,
                                                                         const torch::Tensor &core_weight_scale,
                                                                         const torch::Tensor &gate_weight_scale,
                                                                         const torch::Tensor &core_activation_scale,
                                                                         const torch::Tensor &gate_activation_scale,
                                                                         const torch::Tensor &core_bias,
                                                                         const torch::Tensor &gate_bias,
                                                                         bool core_has_bias,
                                                                         bool gate_has_bias);

std::vector<torch::Tensor> quant_linear_w8a8_static_cutlass_grouped_pair(const torch::Tensor &input_a,
                                                                         const torch::Tensor &input_b,
                                                                         const torch::Tensor &q_weight_a,
                                                                         const torch::Tensor &q_weight_b,
                                                                         const torch::Tensor &weight_scale_a,
                                                                         const torch::Tensor &weight_scale_b,
                                                                         const torch::Tensor &activation_scale_a,
                                                                         const torch::Tensor &activation_scale_b,
                                                                         const torch::Tensor &bias_a,
                                                                         const torch::Tensor &bias_b,
                                                                         bool has_bias_a,
                                                                         bool has_bias_b);

std::vector<torch::Tensor> quant_ffn_w8a8_static_cutlass_grouped_pair(const torch::Tensor &input_a,
                                                                      const torch::Tensor &input_b,
                                                                      const torch::Tensor &q_weight1_a,
                                                                      const torch::Tensor &q_weight1_b,
                                                                      const torch::Tensor &weight_scale1_a,
                                                                      const torch::Tensor &weight_scale1_b,
                                                                      const torch::Tensor &activation_scale1_a,
                                                                      const torch::Tensor &activation_scale1_b,
                                                                      const torch::Tensor &bias1_a,
                                                                      const torch::Tensor &bias1_b,
                                                                      bool has_bias1_a,
                                                                      bool has_bias1_b,
                                                                      const torch::Tensor &q_weight2_a,
                                                                      const torch::Tensor &q_weight2_b,
                                                                      const torch::Tensor &weight_scale2_a,
                                                                      const torch::Tensor &weight_scale2_b,
                                                                      const torch::Tensor &activation_scale2_a,
                                                                      const torch::Tensor &activation_scale2_b,
                                                                      const torch::Tensor &bias2_a,
                                                                      const torch::Tensor &bias2_b,
                                                                      bool has_bias2_a,
                                                                      bool has_bias2_b);

std::vector<torch::Tensor> quant_linear_w8a8_static_wmma_dual(const torch::Tensor &core_input,
                                                              const torch::Tensor &gate_input,
                                                              const torch::Tensor &core_q_weight,
                                                              const torch::Tensor &gate_q_weight,
                                                              const torch::Tensor &core_weight_scale,
                                                              const torch::Tensor &gate_weight_scale,
                                                              const torch::Tensor &core_activation_scale,
                                                              const torch::Tensor &gate_activation_scale,
                                                              const torch::Tensor &core_bias,
                                                              const torch::Tensor &gate_bias,
                                                              bool core_has_bias,
                                                              bool gate_has_bias);

std::vector<torch::Tensor> target_attention_sum_forward(const torch::Tensor &logits,
                                                        const torch::Tensor &values,
                                                        const torch::Tensor &lengths);

std::vector<torch::Tensor> target_attention_sum_backward(const torch::Tensor &grad_out,
                                                         const torch::Tensor &values,
                                                         const torch::Tensor &out,
                                                         const torch::Tensor &alpha,
                                                         const torch::Tensor &lengths);

torch::Tensor directed2undirected_average_forward(const torch::Tensor &input,
                                                  const torch::Tensor &segment,
                                                  int64_t num_segment);

torch::Tensor directed2undirected_average_backward(const torch::Tensor &grad_out,
                                                   const torch::Tensor &segment,
                                                   int64_t rows);

torch::Tensor edge_vectors_forward(const torch::Tensor &coords,
                                   const torch::Tensor &lattice,
                                   const torch::Tensor &image,
                                   const torch::Tensor &target,
                                   const torch::Tensor &source);

std::vector<torch::Tensor> edge_vectors_backward(const torch::Tensor &grad_edge,
                                                 const torch::Tensor &image,
                                                 const torch::Tensor &target,
                                                 const torch::Tensor &source,
                                                 int64_t num_coords,
                                                 int64_t lattice_rows);

std::vector<torch::Tensor> force_stress_from_edge_vectors(const torch::Tensor &grad_edge,
                                                          const torch::Tensor &edge_vectors,
                                                          const torch::Tensor &target,
                                                          const torch::Tensor &source,
                                                          const torch::Tensor &atom_segment,
                                                          const torch::Tensor &volumes);

std::vector<torch::Tensor> fused_line_attention_forward(const torch::Tensor &source_logits,
                                                        const torch::Tensor &target_logits,
                                                        const torch::Tensor &values,
                                                        const torch::Tensor &source_index,
                                                        const torch::Tensor &target_index,
                                                        int64_t num_segments);

std::vector<torch::Tensor> fused_line_attention_forward_v2(const torch::Tensor &source_logits,
                                                           const torch::Tensor &target_logits,
                                                           const torch::Tensor &values,
                                                           const torch::Tensor &source_index,
                                                           const torch::Tensor &target_index,
                                                           int64_t num_segments);

std::vector<torch::Tensor> fused_line_attention_max_atomic(const torch::Tensor &source_logits,
                                                           const torch::Tensor &target_logits,
                                                           const torch::Tensor &source_index,
                                                           const torch::Tensor &target_index,
                                                           int64_t num_segments);

torch::Tensor fused_line_attention_single_max_atomic(const torch::Tensor &logits,
                                                     const torch::Tensor &index,
                                                     int64_t num_segments);

std::vector<torch::Tensor> fused_line_attention_forward_with_max(const torch::Tensor &source_logits,
                                                                 const torch::Tensor &target_logits,
                                                                 const torch::Tensor &values,
                                                                 const torch::Tensor &source_index,
                                                                 const torch::Tensor &target_index,
                                                                 const torch::Tensor &source_max,
                                                                 const torch::Tensor &target_max,
                                                                 int64_t num_segments);

std::vector<torch::Tensor> fused_line_attention_forward_target_offsets(const torch::Tensor &source_logits,
                                                                       const torch::Tensor &target_logits,
                                                                       const torch::Tensor &values,
                                                                       const torch::Tensor &source_index,
                                                                       const torch::Tensor &target_index,
                                                                       const torch::Tensor &target_offsets,
                                                                       int64_t num_segments);

std::vector<torch::Tensor> fused_line_attention_node_input_forward_target_offsets(
    const torch::Tensor &source_logits,
    const torch::Tensor &target_logits,
    const torch::Tensor &values,
    const torch::Tensor &source_index,
    const torch::Tensor &target_index,
    const torch::Tensor &target_offsets,
    const torch::Tensor &node_feat,
    int64_t num_segments);

std::vector<torch::Tensor> fused_line_attention_backward(const torch::Tensor &grad_source_out,
                                                         const torch::Tensor &grad_target_out,
                                                         const torch::Tensor &values,
                                                         const torch::Tensor &source_out,
                                                         const torch::Tensor &target_out,
                                                         const torch::Tensor &source_alpha,
                                                         const torch::Tensor &target_alpha,
                                                         const torch::Tensor &source_index,
                                                         const torch::Tensor &target_index);

std::vector<torch::Tensor> fused_line_attention_backward_with_edge_direct(
    const torch::Tensor &grad_source_out,
    const torch::Tensor &grad_target_out,
    const torch::Tensor &grad_edge_direct,
    const torch::Tensor &values,
    const torch::Tensor &source_out,
    const torch::Tensor &target_out,
    const torch::Tensor &source_alpha,
    const torch::Tensor &target_alpha,
    const torch::Tensor &source_index,
    const torch::Tensor &target_index);

torch::Tensor fused_line_attention_values_backward_with_edge_direct(
    const torch::Tensor &grad_source_out,
    const torch::Tensor &grad_target_out,
    const torch::Tensor &grad_edge_direct,
    const torch::Tensor &source_alpha,
    const torch::Tensor &target_alpha,
    const torch::Tensor &source_index,
    const torch::Tensor &target_index);

torch::Tensor line_edge_gather_cat_forward(const torch::Tensor &node_feat,
                                           const torch::Tensor &edge_feat,
                                           const torch::Tensor &source_index,
                                           const torch::Tensor &target_index);

std::vector<torch::Tensor> line_edge_cat_grad_scatter_backward(const torch::Tensor &grad_cat,
                                                               const torch::Tensor &source_index,
                                                               const torch::Tensor &target_index,
                                                               int64_t node_rows);

torch::Tensor line_node_triple_cat_forward(const torch::Tensor &node_feat,
                                           const torch::Tensor &target_feat,
                                           const torch::Tensor &source_feat);

std::vector<torch::Tensor> line_node_triple_cat_backward(const torch::Tensor &grad_cat);

torch::Tensor directed_edge_gather_cat_forward(const torch::Tensor &node_feat,
                                               const torch::Tensor &edge_feat,
                                               const torch::Tensor &edge_index,
                                               const torch::Tensor &source_index,
                                               const torch::Tensor &target_index);

std::vector<torch::Tensor> directed_edge_cat_grad_scatter_backward(const torch::Tensor &grad_cat,
                                                                   const torch::Tensor &edge_index,
                                                                   const torch::Tensor &source_index,
                                                                   const torch::Tensor &target_index,
                                                                   int64_t node_rows,
                                                                   int64_t edge_rows);

std::vector<torch::Tensor> directed_edge_silu_project_grad_scatter_backward_tile32(
    const torch::Tensor &grad_core,
    const torch::Tensor &grad_gate,
    const torch::Tensor &core_projected,
    const torch::Tensor &gate_projected,
    const torch::Tensor &weight,
    const torch::Tensor &edge_index,
    const torch::Tensor &source_index,
    const torch::Tensor &target_index,
    int64_t node_rows,
    int64_t edge_rows);

torch::Tensor refine_line_edge_gather_cat_forward(const torch::Tensor &node_feat,
                                                  const torch::Tensor &edge_feat,
                                                  const torch::Tensor &atom_feat,
                                                  const torch::Tensor &atom_index,
                                                  const torch::Tensor &source_index,
                                                  const torch::Tensor &target_index);

std::vector<torch::Tensor> refine_line_edge_cat_grad_scatter_backward(const torch::Tensor &grad_cat,
                                                                      const torch::Tensor &atom_index,
                                                                      const torch::Tensor &source_index,
                                                                      const torch::Tensor &target_index,
                                                                      int64_t node_rows,
                                                                      int64_t edge_rows,
                                                                      int64_t atom_rows);

std::vector<torch::Tensor> refine_line_project_grad_scatter_backward_tile32(
    const torch::Tensor &grad_projected,
    const torch::Tensor &weight,
    const torch::Tensor &atom_index,
    const torch::Tensor &source_index,
    const torch::Tensor &target_index,
    int64_t node_rows,
    int64_t edge_rows,
    int64_t atom_rows);

std::vector<torch::Tensor> refine_line_project_dual_grad_scatter_add_tile32(
    const torch::Tensor &grad_core,
    const torch::Tensor &grad_gate,
    const torch::Tensor &weight,
    const torch::Tensor &atom_index,
    const torch::Tensor &source_index,
    const torch::Tensor &target_index,
    const torch::Tensor &grad_node_in,
    const torch::Tensor &grad_edge_in,
    int64_t atom_rows);

std::vector<torch::Tensor> refine_line_first_silu_forward(
    const torch::Tensor &node_feat,
    const torch::Tensor &edge_feat,
    const torch::Tensor &atom_feat,
    const torch::Tensor &atom_index,
    const torch::Tensor &source_index,
    const torch::Tensor &target_index,
    const torch::Tensor &weight,
    const torch::Tensor &bias,
    bool has_bias);

std::vector<torch::Tensor> refine_line_first_silu_forward_acts(
    const torch::Tensor &node_feat,
    const torch::Tensor &edge_feat,
    const torch::Tensor &atom_feat,
    const torch::Tensor &atom_index,
    const torch::Tensor &source_index,
    const torch::Tensor &target_index,
    const torch::Tensor &weight,
    const torch::Tensor &bias,
    bool has_bias);

torch::Tensor refine_line_first_tail_w8a8_forward(
    const torch::Tensor &node_feat,
    const torch::Tensor &edge_feat,
    const torch::Tensor &atom_feat,
    const torch::Tensor &atom_index,
    const torch::Tensor &source_index,
    const torch::Tensor &target_index,
    const torch::Tensor &first_weight,
    const torch::Tensor &first_bias,
    bool first_has_bias,
    const torch::Tensor &core_q_weight,
    const torch::Tensor &gate_q_weight,
    const torch::Tensor &core_weight_scale,
    const torch::Tensor &gate_weight_scale,
    const torch::Tensor &core_activation_scale,
    const torch::Tensor &gate_activation_scale,
    const torch::Tensor &core_bias,
    const torch::Tensor &gate_bias,
    bool core_has_bias,
    bool gate_has_bias,
    const torch::Tensor &core_norm_weight,
    const torch::Tensor &core_norm_bias,
    const torch::Tensor &gate_norm_weight,
    const torch::Tensor &gate_norm_bias,
    double eps,
    bool use_parallel_tail,
    bool use_dynamic_activation_scale,
    bool use_tile_scale_partials);

std::vector<torch::Tensor> refine_line_first_tail_w8a8_forward_with_pre(
    const torch::Tensor &node_feat,
    const torch::Tensor &edge_feat,
    const torch::Tensor &atom_feat,
    const torch::Tensor &atom_index,
    const torch::Tensor &source_index,
    const torch::Tensor &target_index,
    const torch::Tensor &first_weight,
    const torch::Tensor &first_bias,
    bool first_has_bias,
    const torch::Tensor &core_q_weight,
    const torch::Tensor &gate_q_weight,
    const torch::Tensor &core_weight_scale,
    const torch::Tensor &gate_weight_scale,
    const torch::Tensor &core_activation_scale,
    const torch::Tensor &gate_activation_scale,
    const torch::Tensor &core_bias,
    const torch::Tensor &gate_bias,
    bool core_has_bias,
    bool gate_has_bias,
    const torch::Tensor &core_norm_weight,
    const torch::Tensor &core_norm_bias,
    const torch::Tensor &gate_norm_weight,
    const torch::Tensor &gate_norm_bias,
    double eps,
    bool use_parallel_tail);

std::vector<torch::Tensor> refine_line_first_tail_smooth_reduce_w8a8_forward_with_pre(
    const torch::Tensor &node_feat,
    const torch::Tensor &edge_feat,
    const torch::Tensor &atom_feat,
    const torch::Tensor &base_envelope,
    const torch::Tensor &atom_index,
    const torch::Tensor &source_index,
    const torch::Tensor &target_index,
    const torch::Tensor &first_weight,
    const torch::Tensor &first_bias,
    bool first_has_bias,
    const torch::Tensor &core_q_weight,
    const torch::Tensor &gate_q_weight,
    const torch::Tensor &core_weight_scale,
    const torch::Tensor &gate_weight_scale,
    const torch::Tensor &core_activation_scale,
    const torch::Tensor &gate_activation_scale,
    const torch::Tensor &core_bias,
    const torch::Tensor &gate_bias,
    bool core_has_bias,
    bool gate_has_bias,
    const torch::Tensor &core_norm_weight,
    const torch::Tensor &core_norm_bias,
    const torch::Tensor &gate_norm_weight,
    const torch::Tensor &gate_norm_bias,
    double eps,
    bool use_parallel_tail,
    int64_t num_nodes);

torch::Tensor refine_line_first_tail_w8a8_packed_forward(
    const torch::Tensor &node_feat,
    const torch::Tensor &edge_feat,
    const torch::Tensor &atom_feat,
    const torch::Tensor &atom_index,
    const torch::Tensor &source_index,
    const torch::Tensor &target_index,
    const torch::Tensor &first_weight,
    const torch::Tensor &first_bias,
    bool first_has_bias,
    const torch::Tensor &core_q_weight,
    const torch::Tensor &gate_q_weight,
    const torch::Tensor &core_weight_scale,
    const torch::Tensor &gate_weight_scale,
    const torch::Tensor &core_activation_scale,
    const torch::Tensor &gate_activation_scale,
    const torch::Tensor &core_bias,
    const torch::Tensor &gate_bias,
    bool core_has_bias,
    bool gate_has_bias,
    const torch::Tensor &core_norm_weight,
    const torch::Tensor &core_norm_bias,
    const torch::Tensor &gate_norm_weight,
    const torch::Tensor &gate_norm_bias,
    double eps,
    bool use_parallel_tail,
    bool use_dynamic_activation_scale,
    bool use_tile_scale_partials);

std::vector<torch::Tensor> refine_line_first_silu_backward(
    const torch::Tensor &grad_core,
    const torch::Tensor &grad_gate,
    const torch::Tensor &core_raw,
    const torch::Tensor &gate_raw,
    const torch::Tensor &weight,
    const torch::Tensor &atom_index,
    const torch::Tensor &source_index,
    const torch::Tensor &target_index,
    int64_t node_rows,
    int64_t edge_rows,
    int64_t atom_rows);

std::vector<torch::Tensor> refine_line_first_silu_backward_packed(
    const torch::Tensor &grad_act,
    const torch::Tensor &core_raw,
    const torch::Tensor &gate_raw,
    const torch::Tensor &weight,
    const torch::Tensor &atom_index,
    const torch::Tensor &source_index,
    const torch::Tensor &target_index,
    int64_t node_rows,
    int64_t edge_rows,
    int64_t atom_rows);

torch::Tensor refine_line_smooth_reduce_forward(const torch::Tensor &nonlinear,
                                                const torch::Tensor &base_envelope,
                                                const torch::Tensor &source_index,
                                                const torch::Tensor &target_index,
                                                int64_t num_nodes);

std::vector<torch::Tensor> refine_line_smooth_reduce_backward(const torch::Tensor &grad_out,
                                                              const torch::Tensor &nonlinear,
                                                              const torch::Tensor &base_envelope,
                                                              const torch::Tensor &source_index,
                                                              const torch::Tensor &target_index);

torch::Tensor refine_line_smooth_reduce_sorted_forward(const torch::Tensor &nonlinear,
                                                       const torch::Tensor &base_envelope,
                                                       const torch::Tensor &source_index,
                                                       const torch::Tensor &target_offsets);

std::vector<torch::Tensor> refine_line_smooth_reduce_sorted_backward(const torch::Tensor &grad_out,
                                                                     const torch::Tensor &nonlinear,
                                                                     const torch::Tensor &base_envelope,
                                                                     const torch::Tensor &source_index,
                                                                     const torch::Tensor &target_offsets);

std::vector<torch::Tensor> refine_line_edge_smooth_w8a8_backward_n128(
    const torch::Tensor &grad_refine_node,
    const torch::Tensor &grad_nonlinear_direct,
    const torch::Tensor &nonlinear,
    const torch::Tensor &base_envelope,
    const torch::Tensor &atom_index,
    const torch::Tensor &source_index,
    const torch::Tensor &target_index,
    const torch::Tensor &first_weight,
    const torch::Tensor &core_raw,
    const torch::Tensor &gate_raw,
    const torch::Tensor &core_pre,
    const torch::Tensor &gate_pre,
    const torch::Tensor &core_q_weight,
    const torch::Tensor &gate_q_weight,
    const torch::Tensor &core_weight_scale,
    const torch::Tensor &gate_weight_scale,
    const torch::Tensor &core_norm_weight,
    const torch::Tensor &core_norm_bias,
    const torch::Tensor &gate_norm_weight,
    const torch::Tensor &gate_norm_bias,
    int64_t node_rows,
    int64_t edge_rows,
    int64_t atom_rows,
    bool has_direct_grad,
    double eps);

std::vector<torch::Tensor> refine_line_smooth_w8a8_tail_input_grad_backward_n128(
    const torch::Tensor &grad_refine_node,
    const torch::Tensor &grad_nonlinear_direct,
    const torch::Tensor &nonlinear,
    const torch::Tensor &base_envelope,
    const torch::Tensor &source_index,
    const torch::Tensor &target_index,
    const torch::Tensor &core_pre,
    const torch::Tensor &gate_pre,
    const torch::Tensor &core_q_weight,
    const torch::Tensor &gate_q_weight,
    const torch::Tensor &core_weight_scale,
    const torch::Tensor &gate_weight_scale,
    const torch::Tensor &core_norm_weight,
    const torch::Tensor &core_norm_bias,
    const torch::Tensor &gate_norm_weight,
    const torch::Tensor &gate_norm_bias,
    bool has_direct_grad,
    double eps);

std::vector<torch::Tensor> refine_line_smooth_w8a8_tail_actgrad_backward_n128(
    const torch::Tensor &grad_refine_node,
    const torch::Tensor &grad_nonlinear_direct,
    const torch::Tensor &nonlinear,
    const torch::Tensor &base_envelope,
    const torch::Tensor &source_index,
    const torch::Tensor &target_index,
    const torch::Tensor &core_pre,
    const torch::Tensor &gate_pre,
    const torch::Tensor &core_q_weight,
    const torch::Tensor &gate_q_weight,
    const torch::Tensor &core_weight_scale,
    const torch::Tensor &gate_weight_scale,
    const torch::Tensor &core_norm_weight,
    const torch::Tensor &core_norm_bias,
    const torch::Tensor &gate_norm_weight,
    const torch::Tensor &gate_norm_bias,
    bool has_direct_grad,
    double eps);

std::vector<torch::Tensor> refine_line_smooth_tail_first_silu_backward_tile(
    const torch::Tensor &grad_refine_node,
    const torch::Tensor &grad_nonlinear_direct,
    const torch::Tensor &nonlinear,
    const torch::Tensor &base_envelope,
    const torch::Tensor &atom_index,
    const torch::Tensor &source_index,
    const torch::Tensor &target_index,
    const torch::Tensor &first_weight,
    const torch::Tensor &core_raw,
    const torch::Tensor &gate_raw,
    const torch::Tensor &core_pre,
    const torch::Tensor &gate_pre,
    const torch::Tensor &core_q_weight,
    const torch::Tensor &gate_q_weight,
    const torch::Tensor &core_weight_scale,
    const torch::Tensor &gate_weight_scale,
    const torch::Tensor &core_norm_weight,
    const torch::Tensor &core_norm_bias,
    const torch::Tensor &gate_norm_weight,
    const torch::Tensor &gate_norm_bias,
    int64_t node_rows,
    int64_t edge_rows,
    int64_t atom_rows,
    bool has_direct_grad,
    double eps,
    int64_t tile_rows);

std::vector<torch::Tensor> line_edge_project_grad_scatter_backward(const torch::Tensor &grad_projected,
                                                                   const torch::Tensor &weight,
                                                                   const torch::Tensor &source_index,
                                                                   const torch::Tensor &target_index,
                                                                   int64_t node_rows);

std::vector<torch::Tensor> line_edge_project_grad_scatter_backward_tiled(const torch::Tensor &grad_projected,
                                                                         const torch::Tensor &weight,
                                                                         const torch::Tensor &source_index,
                                                                         const torch::Tensor &target_index,
                                                                         int64_t node_rows);

std::vector<torch::Tensor> line_edge_project_grad_scatter_backward_tile32(const torch::Tensor &grad_projected,
                                                                          const torch::Tensor &weight,
                                                                          const torch::Tensor &source_index,
                                                                          const torch::Tensor &target_index,
                                                                          int64_t node_rows);

std::vector<torch::Tensor> line_edge_silu_project_grad_scatter_backward_tile32(
    const torch::Tensor &grad_core,
    const torch::Tensor &grad_gate,
    const torch::Tensor &core_projected,
    const torch::Tensor &gate_projected,
    const torch::Tensor &weight,
    const torch::Tensor &source_index,
    const torch::Tensor &target_index,
    int64_t node_rows);

std::vector<torch::Tensor> line_edge_silu_project_alpha_grad_scatter_backward_tile32(
    const torch::Tensor &grad_core,
    const torch::Tensor &grad_gate,
    const torch::Tensor &core_projected,
    const torch::Tensor &gate_projected,
    const torch::Tensor &first_weight,
    const torch::Tensor &grad_source_logits,
    const torch::Tensor &grad_target_logits,
    const torch::Tensor &source_alpha_weight,
    const torch::Tensor &target_alpha_weight,
    const torch::Tensor &source_index,
    const torch::Tensor &target_index,
    int64_t node_rows);

std::vector<torch::Tensor> line_edge_silu_project_alpha_grad_scatter_backward_alpha_tile32(
    const torch::Tensor &grad_core,
    const torch::Tensor &grad_gate,
    const torch::Tensor &core_projected,
    const torch::Tensor &gate_projected,
    const torch::Tensor &first_weight,
    const torch::Tensor &grad_source_logits,
    const torch::Tensor &grad_target_logits,
    const torch::Tensor &source_alpha_weight,
    const torch::Tensor &target_alpha_weight,
    const torch::Tensor &source_index,
    const torch::Tensor &target_index,
    int64_t node_rows);

std::vector<torch::Tensor> line_edge_silu_project_alpha_grad_scatter_backward_dense_gemm(
    const torch::Tensor &grad_core,
    const torch::Tensor &grad_gate,
    const torch::Tensor &core_projected,
    const torch::Tensor &gate_projected,
    const torch::Tensor &first_weight,
    const torch::Tensor &grad_source_logits,
    const torch::Tensor &grad_target_logits,
    const torch::Tensor &source_alpha_weight,
    const torch::Tensor &target_alpha_weight,
    const torch::Tensor &source_index,
    const torch::Tensor &target_index,
    int64_t node_rows);

std::vector<torch::Tensor> line_edge_silu_project_alpha_grad_scatter_backward_target_reduce_tile32(
    const torch::Tensor &grad_core,
    const torch::Tensor &grad_gate,
    const torch::Tensor &core_projected,
    const torch::Tensor &gate_projected,
    const torch::Tensor &first_weight,
    const torch::Tensor &grad_source_logits,
    const torch::Tensor &grad_target_logits,
    const torch::Tensor &source_alpha_weight,
    const torch::Tensor &target_alpha_weight,
    const torch::Tensor &source_index,
    const torch::Tensor &target_index,
    const torch::Tensor &target_offsets,
    int64_t node_rows);

std::vector<torch::Tensor> line_edge_silu_project_alpha_attention_grad_scatter_backward_tile32(
    const torch::Tensor &grad_core,
    const torch::Tensor &grad_gate,
    const torch::Tensor &core_projected,
    const torch::Tensor &gate_projected,
    const torch::Tensor &first_weight,
    const torch::Tensor &grad_source_out,
    const torch::Tensor &grad_target_out,
    const torch::Tensor &values,
    const torch::Tensor &source_out,
    const torch::Tensor &target_out,
    const torch::Tensor &source_alpha,
    const torch::Tensor &target_alpha,
    const torch::Tensor &source_alpha_weight,
    const torch::Tensor &target_alpha_weight,
    const torch::Tensor &source_index,
    const torch::Tensor &target_index,
    int64_t node_rows);

std::vector<torch::Tensor> line_edge_w8a8_tail_project_scatter_backward_n128(
    const torch::Tensor &grad_out,
    const torch::Tensor &core_pre,
    const torch::Tensor &gate_pre,
    const torch::Tensor &core_projected,
    const torch::Tensor &gate_projected,
    const torch::Tensor &core_q_weight,
    const torch::Tensor &gate_q_weight,
    const torch::Tensor &core_weight_scale,
    const torch::Tensor &gate_weight_scale,
    const torch::Tensor &core_norm_weight,
    const torch::Tensor &core_norm_bias,
    const torch::Tensor &gate_norm_weight,
    const torch::Tensor &gate_norm_bias,
    const torch::Tensor &first_weight,
    const torch::Tensor &source_index,
    const torch::Tensor &target_index,
    int64_t node_rows,
    double eps);

std::vector<torch::Tensor> input_grad_only_gated_tail_backward(const torch::Tensor &grad_out,
                                                               const torch::Tensor &core,
                                                               const torch::Tensor &gate,
                                                               const torch::Tensor &core_weight,
                                                               const torch::Tensor &core_bias,
                                                               const torch::Tensor &gate_weight,
                                                               const torch::Tensor &gate_bias,
                                                               double eps);

std::vector<torch::Tensor> input_grad_only_gated_tail_backward_n128_v2(const torch::Tensor &grad_out,
                                                                       const torch::Tensor &core,
                                                                       const torch::Tensor &gate,
                                                                       const torch::Tensor &core_weight,
                                                                       const torch::Tensor &core_bias,
                                                                       const torch::Tensor &gate_weight,
                                                                       const torch::Tensor &gate_bias,
                                                                       double eps);

std::vector<torch::Tensor> gated_tail_second_silu_input_grad_macro(
    const torch::Tensor &grad_out,
    const torch::Tensor &core,
    const torch::Tensor &gate,
    const torch::Tensor &core_weight,
    const torch::Tensor &core_bias,
    const torch::Tensor &gate_weight,
    const torch::Tensor &gate_bias,
    double eps,
    const torch::Tensor &core_second_weight,
    const torch::Tensor &gate_second_weight,
    const torch::Tensor &core_first_hidden,
    const torch::Tensor &gate_first_hidden,
    bool use_tail_bwd_v2);

std::vector<torch::Tensor> gated_tail_second_silu_residual_input_grad_macro(
    const torch::Tensor &grad_out,
    const torch::Tensor &core,
    const torch::Tensor &gate,
    const torch::Tensor &core_weight,
    const torch::Tensor &core_bias,
    const torch::Tensor &gate_weight,
    const torch::Tensor &gate_bias,
    double eps,
    const torch::Tensor &core_second_weight,
    const torch::Tensor &gate_second_weight,
    const torch::Tensor &core_first_hidden,
    const torch::Tensor &gate_first_hidden,
    bool use_tail_bwd_v2,
    const torch::Tensor &old_feat,
    const torch::Tensor &res_weight);

torch::Tensor input_grad_only_gated_tail_backward_stack(const torch::Tensor &grad_out,
                                                        const torch::Tensor &core,
                                                        const torch::Tensor &gate,
                                                        const torch::Tensor &core_weight,
                                                        const torch::Tensor &core_bias,
                                                        const torch::Tensor &gate_weight,
                                                        const torch::Tensor &gate_bias,
                                                        double eps);

std::vector<torch::Tensor> param_grad_gated_tail_backward(const torch::Tensor &grad_out,
                                                          const torch::Tensor &core,
                                                          const torch::Tensor &gate,
                                                          const torch::Tensor &core_weight,
                                                          const torch::Tensor &core_bias,
                                                          const torch::Tensor &gate_weight,
                                                          const torch::Tensor &gate_bias,
                                                          double eps);

std::vector<torch::Tensor> w8a8_dual_gated_tail_input_grad_backward_n128(
    const torch::Tensor &grad_out,
    const torch::Tensor &core_input,
    const torch::Tensor &gate_input,
    const torch::Tensor &core_q_weight,
    const torch::Tensor &gate_q_weight,
    const torch::Tensor &core_weight_scale,
    const torch::Tensor &gate_weight_scale,
    const torch::Tensor &core_activation_scale,
    const torch::Tensor &gate_activation_scale,
    const torch::Tensor &core_bias,
    const torch::Tensor &gate_bias,
    bool core_has_bias,
    bool gate_has_bias,
    const torch::Tensor &core_norm_weight,
    const torch::Tensor &core_norm_bias,
    const torch::Tensor &gate_norm_weight,
    const torch::Tensor &gate_norm_bias,
    double eps);

std::vector<torch::Tensor> w8a8_dual_gated_tail_saved_pre_input_grad_backward_n128(
    const torch::Tensor &grad_out,
    const torch::Tensor &core_pre,
    const torch::Tensor &gate_pre,
    const torch::Tensor &core_q_weight,
    const torch::Tensor &gate_q_weight,
    const torch::Tensor &core_weight_scale,
    const torch::Tensor &gate_weight_scale,
    const torch::Tensor &core_norm_weight,
    const torch::Tensor &core_norm_bias,
    const torch::Tensor &gate_norm_weight,
    const torch::Tensor &gate_norm_bias,
    double eps);

std::vector<torch::Tensor> w8a8_dual_gated_tail_saved_pre_dq_input_grad_backward_n128(
    const torch::Tensor &grad_out,
    const torch::Tensor &core_pre,
    const torch::Tensor &gate_pre,
    const torch::Tensor &core_dq_weight_t,
    const torch::Tensor &gate_dq_weight_t,
    const torch::Tensor &core_norm_weight,
    const torch::Tensor &core_norm_bias,
    const torch::Tensor &gate_norm_weight,
    const torch::Tensor &gate_norm_bias,
    double eps);

std::vector<torch::Tensor> w8a8_saved_pre_backward_driver(
    const torch::Tensor &grad_out,
    const torch::Tensor &core_pre,
    const torch::Tensor &gate_pre,
    const torch::Tensor &core_dq_weight_t,
    const torch::Tensor &gate_dq_weight_t,
    const torch::Tensor &core_norm_weight,
    const torch::Tensor &core_norm_bias,
    const torch::Tensor &gate_norm_weight,
    const torch::Tensor &gate_norm_bias,
    double eps,
    std::vector<int64_t> core_shape,
    std::vector<int64_t> gate_shape,
    bool use_tail_bwd_v2,
    bool use_cublas_pair);

std::vector<torch::Tensor> w8a8_saved_pre_backward_group4_driver(
    const std::vector<torch::Tensor> &grad_outs,
    const std::vector<torch::Tensor> &core_pres,
    const std::vector<torch::Tensor> &gate_pres,
    const std::vector<torch::Tensor> &core_dq_weight_ts,
    const std::vector<torch::Tensor> &gate_dq_weight_ts,
    const std::vector<torch::Tensor> &core_norm_weights,
    const std::vector<torch::Tensor> &core_norm_biases,
    const std::vector<torch::Tensor> &gate_norm_weights,
    const std::vector<torch::Tensor> &gate_norm_biases,
    const std::vector<double> &eps_values,
    bool use_tail_bwd_v2,
    bool use_cublas_pair);

std::vector<torch::Tensor> w8a8_dual_input_grad_matmul_n128_tiled(
    const torch::Tensor &grad_core_pre,
    const torch::Tensor &grad_gate_pre,
    const torch::Tensor &core_q_weight,
    const torch::Tensor &gate_q_weight,
    const torch::Tensor &core_weight_scale,
    const torch::Tensor &gate_weight_scale);

std::vector<torch::Tensor> w8a8_dual_input_grad_matmul_n128_tiled_m32n8(
    const torch::Tensor &grad_core_pre,
    const torch::Tensor &grad_gate_pre,
    const torch::Tensor &core_q_weight,
    const torch::Tensor &gate_q_weight,
    const torch::Tensor &core_weight_scale,
    const torch::Tensor &gate_weight_scale);

std::vector<torch::Tensor> w8a8_dual_input_grad_matmul_n128_cublas_grouped(
    const torch::Tensor &grad_core_pre,
    const torch::Tensor &grad_gate_pre,
    const torch::Tensor &core_dq_weight_t,
    const torch::Tensor &gate_dq_weight_t);

std::vector<torch::Tensor> w8a8_dual_input_grad_matmul_n128_cublas_pair(
    const torch::Tensor &grad_core_pre,
    const torch::Tensor &grad_gate_pre,
    const torch::Tensor &core_dq_weight_t,
    const torch::Tensor &gate_dq_weight_t);

std::vector<torch::Tensor> w8a8_group8_input_grad_matmul_n128_cublas_grouped(
    const std::vector<torch::Tensor> &grad_pres,
    const std::vector<torch::Tensor> &dq_weight_ts);

torch::Tensor two_linear_silu_input_grad_backward_n128(
    const torch::Tensor &grad_out,
    const torch::Tensor &weight2,
    const torch::Tensor &hidden,
    const torch::Tensor &weight1);

#endif  // OP_SRC_OPDECLARE_H_
