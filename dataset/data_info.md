# 内置算子数据集

> 自动生成自 `dataset/operator_index.json`

## 统计概览

| 复杂度 | 数量 |
|--------|------|
| Low    | 49 |
| Medium | 101 |
| High   | 150 |
| **总计** | **300** |

## 使用方式

```bash
# 生成内置算子
cann-claude generate relu dataset/py_reference/activation/relu.py

# 指定迭代次数
cann-claude generate softmax dataset/py_reference/activation/softmax.py -n 20

# 测试模式（跳过编译）
cann-claude generate relu dataset/py_reference/activation/relu.py --fake-mode
```

---

## 算子列表

### 激活函数 (Activation)

| 算子名称 | 复杂度 | 路径 |
|----------|--------|------|
| `elu` | 🟢 low | `dataset/py_reference/activation/elu.py` |
| `gelu` | 🟢 low | `dataset/py_reference/activation/gelu.py` |
| `hardsigmoid` | 🟢 low | `dataset/py_reference/activation/hardsigmoid.py` |
| `hardtanh` | 🟢 low | `dataset/py_reference/activation/hardtanh.py` |
| `leaky_relu` | 🟢 low | `dataset/py_reference/activation/leaky_relu.py` |
| `log_softmax` | 🟢 low | `dataset/py_reference/activation/log_softmax.py` |
| `min_gpt_new_gelu` | 🟢 low | `dataset/py_reference/activation/min_gpt_new_gelu.py` |
| `relu` | 🟢 low | `dataset/py_reference/activation/relu.py` |
| `selu` | 🟢 low | `dataset/py_reference/activation/selu.py` |
| `sigmoid` | 🟢 low | `dataset/py_reference/activation/sigmoid.py` |
| `softmax` | 🟢 low | `dataset/py_reference/activation/softmax.py` |
| `softplus` | 🟢 low | `dataset/py_reference/activation/softplus.py` |
| `softsign` | 🟢 low | `dataset/py_reference/activation/softsign.py` |
| `swish` | 🟢 low | `dataset/py_reference/activation/swish.py` |
| `tanh` | 🟢 low | `dataset/py_reference/activation/tanh.py` |

### 数学运算 (Math)

| 算子名称 | 复杂度 | 路径 |
|----------|--------|------|
| `cumprod` | 🟢 low | `dataset/py_reference/math/cumprod.py` |
| `cumsum` | 🟢 low | `dataset/py_reference/math/cumsum.py` |
| `cumsum_exclusive` | 🟢 low | `dataset/py_reference/math/cumsum_exclusive.py` |
| `cumsum_reverse` | 🟢 low | `dataset/py_reference/math/cumsum_reverse.py` |
| `masked_cumsum` | 🟢 low | `dataset/py_reference/math/masked_cumsum.py` |
| `matrix_scalar_multiplication` | 🟢 low | `dataset/py_reference/math/matrix_scalar_multiplication.py` |

### 归约运算 (Reduce)

| 算子名称 | 复杂度 | 路径 |
|----------|--------|------|
| `max_reduction_over_a_dimension` | 🟢 low | `dataset/py_reference/reduce/max_reduction_over_a_dimension.py` |
| `mean_reduction_over_a_dimension` | 🟢 low | `dataset/py_reference/reduce/mean_reduction_over_a_dimension.py` |
| `min_reduction_over_a_dimension` | 🟢 low | `dataset/py_reference/reduce/min_reduction_over_a_dimension.py` |
| `product_reduction_over_a_dimension` | 🟢 low | `dataset/py_reference/reduce/product_reduction_over_a_dimension.py` |
| `sum_reduction_over_a_dimension` | 🟢 low | `dataset/py_reference/reduce/sum_reduction_over_a_dimension.py` |

### 池化 (Pooling)

| 算子名称 | 复杂度 | 路径 |
|----------|--------|------|
| `average_pooling_1d` | 🟢 low | `dataset/py_reference/pooling/average_pooling_1d.py` |
| `average_pooling_2d` | 🟢 low | `dataset/py_reference/pooling/average_pooling_2d.py` |
| `average_pooling_3d` | 🟢 low | `dataset/py_reference/pooling/average_pooling_3d.py` |
| `max_pooling_1d` | 🟢 low | `dataset/py_reference/pooling/max_pooling_1d.py` |
| `max_pooling_2d` | 🟢 low | `dataset/py_reference/pooling/max_pooling_2d.py` |
| `max_pooling_3d` | 🟢 low | `dataset/py_reference/pooling/max_pooling_3d.py` |

### 广播运算 (Broadcast)

| 算子名称 | 复杂度 | 路径 |
|----------|--------|------|
| `add_bias_broadcast` | 🟢 low | `dataset/py_reference/broadcast/add_bias_broadcast.py` |
| `add_bias_four_dim_broadcast` | 🟢 low | `dataset/py_reference/broadcast/add_bias_four_dim_broadcast.py` |
| `clamp_broadcast` | 🟢 low | `dataset/py_reference/broadcast/clamp_broadcast.py` |
| `division_broadcast` | 🟢 low | `dataset/py_reference/broadcast/division_broadcast.py` |
| `elmentwise_mul_broadcast` | 🟢 low | `dataset/py_reference/broadcast/elmentwise_mul_broadcast.py` |
| `logic_and_broadcast` | 🟢 low | `dataset/py_reference/broadcast/logic_and_broadcast.py` |
| `max_broadcast` | 🟢 low | `dataset/py_reference/broadcast/max_broadcast.py` |
| `power_broadcast` | 🟢 low | `dataset/py_reference/broadcast/power_broadcast.py` |
| `subtract_with_bias_broadcast` | 🟢 low | `dataset/py_reference/broadcast/subtract_with_bias_broadcast.py` |
| `where_broadcast` | 🟢 low | `dataset/py_reference/broadcast/where_broadcast.py` |

### 损失函数 (Loss)

| 算子名称 | 复杂度 | 路径 |
|----------|--------|------|
| `cosine_similarity_loss` | 🟢 low | `dataset/py_reference/loss/cosine_similarity_loss.py` |
| `cross_entropy_loss` | 🟢 low | `dataset/py_reference/loss/cross_entropy_loss.py` |
| `hinge_loss` | 🟢 low | `dataset/py_reference/loss/hinge_loss.py` |
| `huber_loss` | 🟢 low | `dataset/py_reference/loss/huber_loss.py` |
| `kl_div_loss` | 🟢 low | `dataset/py_reference/loss/kl_div_loss.py` |
| `mse_loss` | 🟢 low | `dataset/py_reference/loss/mse_loss.py` |
| `triplet_margin_loss` | 🟢 low | `dataset/py_reference/loss/triplet_margin_loss.py` |

### 归一化 (Normalization)

| 算子名称 | 复杂度 | 路径 |
|----------|--------|------|
| `batch_norm` | 🟡 medium | `dataset/py_reference/normalization/batch_norm.py` |
| `frobenius_norm` | 🟡 medium | `dataset/py_reference/normalization/frobenius_norm.py` |
| `group_norm` | 🟡 medium | `dataset/py_reference/normalization/group_norm.py` |
| `instance_norm` | 🟡 medium | `dataset/py_reference/normalization/instance_norm.py` |
| `l1_norm` | 🟡 medium | `dataset/py_reference/normalization/l1_norm.py` |
| `l2_norm` | 🟡 medium | `dataset/py_reference/normalization/l2_norm.py` |
| `layer_norm` | 🟡 medium | `dataset/py_reference/normalization/layer_norm.py` |
| `rms_norm` | 🟡 medium | `dataset/py_reference/normalization/rms_norm.py` |

### 矩阵乘法 (MatMul)

| 算子名称 | 复杂度 | 路径 |
|----------|--------|------|
| `batched_matrix_multiplication` | 🟡 medium | `dataset/py_reference/matmul/batched_matrix_multiplication.py` |
| `four_dim_tensor_matrix_multiplication` | 🟡 medium | `dataset/py_reference/matmul/four_dim_tensor_matrix_multiplication.py` |
| `matmul_for_lower_triangular_matrices` | 🟡 medium | `dataset/py_reference/matmul/matmul_for_lower_triangular_matrices.py` |
| `matmul_for_symmetric_matrices` | 🟡 medium | `dataset/py_reference/matmul/matmul_for_symmetric_matrices.py` |
| `matmul_for_upper_triangular_matrices` | 🟡 medium | `dataset/py_reference/matmul/matmul_for_upper_triangular_matrices.py` |
| `matmul_with_diagonal_matrices` | 🟡 medium | `dataset/py_reference/matmul/matmul_with_diagonal_matrices.py` |
| `matmul_with_irregular_shapes` | 🟡 medium | `dataset/py_reference/matmul/matmul_with_irregular_shapes.py` |
| `matmul_with_large_k_dimension` | 🟡 medium | `dataset/py_reference/matmul/matmul_with_large_k_dimension.py` |
| `matmul_with_small_k_dimension` | 🟡 medium | `dataset/py_reference/matmul/matmul_with_small_k_dimension.py` |
| `matmul_with_transposed_a` | 🟡 medium | `dataset/py_reference/matmul/matmul_with_transposed_a.py` |
| `matmul_with_transposed_b` | 🟡 medium | `dataset/py_reference/matmul/matmul_with_transposed_b.py` |
| `matmul_with_transposed_both` | 🟡 medium | `dataset/py_reference/matmul/matmul_with_transposed_both.py` |
| `matrix_vector_multiplication` | 🟡 medium | `dataset/py_reference/matmul/matrix_vector_multiplication.py` |
| `square_matrix_multiplication` | 🟡 medium | `dataset/py_reference/matmul/square_matrix_multiplication.py` |
| `standard_matrix_multiplication` | 🟡 medium | `dataset/py_reference/matmul/standard_matrix_multiplication.py` |
| `tall_skinny_matrix_multiplication` | 🟡 medium | `dataset/py_reference/matmul/tall_skinny_matrix_multiplication.py` |
| `three_dim_tensor_matrix_multiplication` | 🟡 medium | `dataset/py_reference/matmul/three_dim_tensor_matrix_multiplication.py` |

### 索引操作 (Index)

| 算子名称 | 复杂度 | 路径 |
|----------|--------|------|
| `argmax_over_a_dimension` | 🟡 medium | `dataset/py_reference/index/argmax_over_a_dimension.py` |
| `argmin_over_a_dimension` | 🟡 medium | `dataset/py_reference/index/argmin_over_a_dimension.py` |
| `embedding` | 🟡 medium | `dataset/py_reference/index/embedding.py` |
| `gather` | 🟡 medium | `dataset/py_reference/index/gather.py` |
| `index_add` | 🟡 medium | `dataset/py_reference/index/index_add.py` |
| `index_copy` | 🟡 medium | `dataset/py_reference/index/index_copy.py` |
| `index_select` | 🟡 medium | `dataset/py_reference/index/index_select.py` |
| `inplace_update` | 🟡 medium | `dataset/py_reference/index/inplace_update.py` |
| `masked_fill` | 🟡 medium | `dataset/py_reference/index/masked_fill.py` |
| `scatter` | 🟡 medium | `dataset/py_reference/index/scatter.py` |
| `scatter_add` | 🟡 medium | `dataset/py_reference/index/scatter_add.py` |
| `take_along_dim` | 🟡 medium | `dataset/py_reference/index/take_along_dim.py` |

### 尺寸变换 (Resize)

| 算子名称 | 复杂度 | 路径 |
|----------|--------|------|
| `bicubic_upsample` | 🟡 medium | `dataset/py_reference/resize/bicubic_upsample.py` |
| `bilinear_upsample` | 🟡 medium | `dataset/py_reference/resize/bilinear_upsample.py` |
| `downsample_bilinear` | 🟡 medium | `dataset/py_reference/resize/downsample_bilinear.py` |
| `grid_sample_affine` | 🟡 medium | `dataset/py_reference/resize/grid_sample_affine.py` |
| `grid_sample_random_warp` | 🟡 medium | `dataset/py_reference/resize/grid_sample_random_warp.py` |
| `interpolate_dynamic` | 🟡 medium | `dataset/py_reference/resize/interpolate_dynamic.py` |
| `nearest_neighbor_upsample` | 🟡 medium | `dataset/py_reference/resize/nearest_neighbor_upsample.py` |
| `resize_with_antialias` | 🟡 medium | `dataset/py_reference/resize/resize_with_antialias.py` |
| `trilinear_upsample` | 🟡 medium | `dataset/py_reference/resize/trilinear_upsample.py` |
| `upsample_grid_sample` | 🟡 medium | `dataset/py_reference/resize/upsample_grid_sample.py` |

### 卷积 (Convolution)

| 算子名称 | 复杂度 | 路径 |
|----------|--------|------|
| `conv_depthwise_2d_asymmetric_input_asymmetric_kernel` | 🟡 medium | `dataset/py_reference/convolution/conv_depthwise_2d_asymmetric_input_asymmetric_kernel.py` |
| `conv_depthwise_2d_asymmetric_input_square_kernel` | 🟡 medium | `dataset/py_reference/convolution/conv_depthwise_2d_asymmetric_input_square_kernel.py` |
| `conv_depthwise_2d_square_input_asymmetric_kernel` | 🟡 medium | `dataset/py_reference/convolution/conv_depthwise_2d_square_input_asymmetric_kernel.py` |
| `conv_depthwise_2d_square_input_square_kernel` | 🟡 medium | `dataset/py_reference/convolution/conv_depthwise_2d_square_input_square_kernel.py` |
| `conv_depthwise_separable_2d` | 🟡 medium | `dataset/py_reference/convolution/conv_depthwise_separable_2d.py` |
| `conv_pointwise_2d` | 🟡 medium | `dataset/py_reference/convolution/conv_pointwise_2d.py` |
| `conv_standard_1d` | 🟡 medium | `dataset/py_reference/convolution/conv_standard_1d.py` |
| `conv_standard_1d_dilated_strided` | 🟡 medium | `dataset/py_reference/convolution/conv_standard_1d_dilated_strided.py` |
| `conv_standard_2d_asymmetric_input_asymmetric_kernel` | 🟡 medium | `dataset/py_reference/convolution/conv_standard_2d_asymmetric_input_asymmetric_kernel.py` |
| `conv_standard_2d_asymmetric_input_square_kernel` | 🟡 medium | `dataset/py_reference/convolution/conv_standard_2d_asymmetric_input_square_kernel.py` |
| `conv_standard_2d_square_input_asymmetric_kernel` | 🟡 medium | `dataset/py_reference/convolution/conv_standard_2d_square_input_asymmetric_kernel.py` |
| `conv_standard_2d_square_input_asymmetric_kernel_dilated_padded` | 🟡 medium | `dataset/py_reference/convolution/conv_standard_2d_square_input_asymmetric_kernel_dilated_padded.py` |
| `conv_standard_2d_square_input_square_kernel` | 🟡 medium | `dataset/py_reference/convolution/conv_standard_2d_square_input_square_kernel.py` |
| `conv_standard_3d_asymmetric_input_asymmetric_kernel` | 🟡 medium | `dataset/py_reference/convolution/conv_standard_3d_asymmetric_input_asymmetric_kernel.py` |
| `conv_standard_3d_asymmetric_input_square_kernel` | 🟡 medium | `dataset/py_reference/convolution/conv_standard_3d_asymmetric_input_square_kernel.py` |
| `conv_standard_3d_square_input_asymmetric_kernel` | 🟡 medium | `dataset/py_reference/convolution/conv_standard_3d_square_input_asymmetric_kernel.py` |
| `conv_standard_3d_square_input_square_kernel` | 🟡 medium | `dataset/py_reference/convolution/conv_standard_3d_square_input_square_kernel.py` |
| `conv_transposed_1d` | 🟡 medium | `dataset/py_reference/convolution/conv_transposed_1d.py` |
| `conv_transposed_1d_asymmetric_input_square_kernel_padded_strided_dilated` | 🟡 medium | `dataset/py_reference/convolution/conv_transposed_1d_asymmetric_input_square_kernel_padded_strided_dilated.py` |
| `conv_transposed_1d_dilated` | 🟡 medium | `dataset/py_reference/convolution/conv_transposed_1d_dilated.py` |
| `conv_transposed_2d_asymmetric_input_asymmetric_kernel` | 🟡 medium | `dataset/py_reference/convolution/conv_transposed_2d_asymmetric_input_asymmetric_kernel.py` |
| `conv_transposed_2d_asymmetric_input_asymmetric_kernel_padded` | 🟡 medium | `dataset/py_reference/convolution/conv_transposed_2d_asymmetric_input_asymmetric_kernel_padded.py` |
| `conv_transposed_2d_asymmetric_input_asymmetric_kernel_strided_grouped_padded_dilated` | 🟡 medium | `dataset/py_reference/convolution/conv_transposed_2d_asymmetric_input_asymmetric_kernel_strided_grouped_padded_dilated.py` |
| `conv_transposed_2d_asymmetric_input_square_kernel` | 🟡 medium | `dataset/py_reference/convolution/conv_transposed_2d_asymmetric_input_square_kernel.py` |
| `conv_transposed_2d_asymmetric_input_square_kernel_dilated_padded_strided` | 🟡 medium | `dataset/py_reference/convolution/conv_transposed_2d_asymmetric_input_square_kernel_dilated_padded_strided.py` |
| `conv_transposed_2d_square_input_asymmetric_kernel` | 🟡 medium | `dataset/py_reference/convolution/conv_transposed_2d_square_input_asymmetric_kernel.py` |
| `conv_transposed_2d_square_input_square_kernel` | 🟡 medium | `dataset/py_reference/convolution/conv_transposed_2d_square_input_square_kernel.py` |
| `conv_transposed_3d_asymmetric_input_asymmetric_kernel` | 🟡 medium | `dataset/py_reference/convolution/conv_transposed_3d_asymmetric_input_asymmetric_kernel.py` |
| `conv_transposed_3d_asymmetric_input_asymmetric_kernel_strided_padded_grouped` | 🟡 medium | `dataset/py_reference/convolution/conv_transposed_3d_asymmetric_input_asymmetric_kernel_strided_padded_grouped.py` |
| `conv_transposed_3d_asymmetric_input_square_kernel` | 🟡 medium | `dataset/py_reference/convolution/conv_transposed_3d_asymmetric_input_square_kernel.py` |
| `conv_transposed_3d_asymmetric_input_square_kernel_strided_padded_grouped` | 🟡 medium | `dataset/py_reference/convolution/conv_transposed_3d_asymmetric_input_square_kernel_strided_padded_grouped.py` |
| `conv_transposed_3d_square_input_asymmetric_kernel` | 🟡 medium | `dataset/py_reference/convolution/conv_transposed_3d_square_input_asymmetric_kernel.py` |
| `conv_transposed_3d_square_input_square_kernel` | 🟡 medium | `dataset/py_reference/convolution/conv_transposed_3d_square_input_square_kernel.py` |
| `conv_transposed_3d_square_input_square_kernel_padded_dilated_strided` | 🟡 medium | `dataset/py_reference/convolution/conv_transposed_3d_square_input_square_kernel_padded_dilated_strided.py` |

### 注意力机制 (Attention)

| 算子名称 | 复杂度 | 路径 |
|----------|--------|------|
| `causal_attention` | 🟡 medium | `dataset/py_reference/attention/causal_attention.py` |
| `cross_attention` | 🟡 medium | `dataset/py_reference/attention/cross_attention.py` |
| `cross_modal_attention` | 🟡 medium | `dataset/py_reference/attention/cross_modal_attention.py` |
| `group_query_attention` | 🟡 medium | `dataset/py_reference/attention/group_query_attention.py` |
| `kv_cached_attention_inference` | 🟡 medium | `dataset/py_reference/attention/kv_cached_attention_inference.py` |
| `kv_cached_chat_batch_attention` | 🟡 medium | `dataset/py_reference/attention/kv_cached_chat_batch_attention.py` |
| `kv_cached_speculative_attention` | 🟡 medium | `dataset/py_reference/attention/kv_cached_speculative_attention.py` |
| `linear_attention` | 🟡 medium | `dataset/py_reference/attention/linear_attention.py` |
| `multi_head_attention` | 🟡 medium | `dataset/py_reference/attention/multi_head_attention.py` |
| `multi_query_attention` | 🟡 medium | `dataset/py_reference/attention/multi_query_attention.py` |
| `scaled_dot_product_attention` | 🟡 medium | `dataset/py_reference/attention/scaled_dot_product_attention.py` |
| `scaled_dot_product_attention_inference` | 🟡 medium | `dataset/py_reference/attention/scaled_dot_product_attention_inference.py` |
| `scaled_dot_product_attention_long_context` | 🟡 medium | `dataset/py_reference/attention/scaled_dot_product_attention_long_context.py` |
| `sparse_attention` | 🟡 medium | `dataset/py_reference/attention/sparse_attention.py` |
| `windowed_causal_attention` | 🟡 medium | `dataset/py_reference/attention/windowed_causal_attention.py` |

### 优化器 (Optimizer)

| 算子名称 | 复杂度 | 路径 |
|----------|--------|------|
| `adagrad` | 🟡 medium | `dataset/py_reference/optimizer/adagrad.py` |
| `adam` | 🟡 medium | `dataset/py_reference/optimizer/adam.py` |
| `lamb` | 🟡 medium | `dataset/py_reference/optimizer/lamb.py` |
| `rmsprop` | 🟡 medium | `dataset/py_reference/optimizer/rmsprop.py` |
| `sgd` | 🟡 medium | `dataset/py_reference/optimizer/sgd.py` |

### 融合算子 (Fused)

| 算子名称 | 复杂度 | 路径 |
|----------|--------|------|
| `bmm_instance_norm_sum_residual_add_multiply` | 🔴 high | `dataset/py_reference/fuse/bmm_instance_norm_sum_residual_add_multiply.py` |
| `conv2d_activation_batch_norm` | 🔴 high | `dataset/py_reference/fuse/conv2d_activation_batch_norm.py` |
| `conv2d_add_scale_sigmoid_group_norm` | 🔴 high | `dataset/py_reference/fuse/conv2d_add_scale_sigmoid_group_norm.py` |
| `conv2d_avg_pool_sigmoid_sum` | 🔴 high | `dataset/py_reference/fuse/conv2d_avg_pool_sigmoid_sum.py` |
| `conv2d_batch_norm_scaling` | 🔴 high | `dataset/py_reference/fuse/conv2d_batch_norm_scaling.py` |
| `conv2d_divide_leaky_relu` | 🔴 high | `dataset/py_reference/fuse/conv2d_divide_leaky_relu.py` |
| `conv2d_gelu_global_avg_pool` | 🔴 high | `dataset/py_reference/fuse/conv2d_gelu_global_avg_pool.py` |
| `conv2d_group_norm_scale_max_pool_clamp` | 🔴 high | `dataset/py_reference/fuse/conv2d_group_norm_scale_max_pool_clamp.py` |
| `conv2d_group_norm_tanh_hard_swish_residual_add_log_sum_exp` | 🔴 high | `dataset/py_reference/fuse/conv2d_group_norm_tanh_hard_swish_residual_add_log_sum_exp.py` |
| `conv2d_hard_swish_relu` | 🔴 high | `dataset/py_reference/fuse/conv2d_hard_swish_relu.py` |
| `conv2d_instance_norm_divide` | 🔴 high | `dataset/py_reference/fuse/conv2d_instance_norm_divide.py` |
| `conv2d_min_add_multiply` | 🔴 high | `dataset/py_reference/fuse/conv2d_min_add_multiply.py` |
| `conv2d_min_tanh_tanh` | 🔴 high | `dataset/py_reference/fuse/conv2d_min_tanh_tanh.py` |
| `conv2d_mish_mish` | 🔴 high | `dataset/py_reference/fuse/conv2d_mish_mish.py` |
| `conv2d_multiply_leaky_relu_gelu` | 🔴 high | `dataset/py_reference/fuse/conv2d_multiply_leaky_relu_gelu.py` |
| `conv2d_relu_bias_add` | 🔴 high | `dataset/py_reference/fuse/conv2d_relu_bias_add.py` |
| `conv2d_relu_hard_swish` | 🔴 high | `dataset/py_reference/fuse/conv2d_relu_hard_swish.py` |
| `conv2d_scaling_min` | 🔴 high | `dataset/py_reference/fuse/conv2d_scaling_min.py` |
| `conv2d_subtract_hard_swish_max_pool_mish` | 🔴 high | `dataset/py_reference/fuse/conv2d_subtract_hard_swish_max_pool_mish.py` |
| `conv2d_subtract_subtract_mish` | 🔴 high | `dataset/py_reference/fuse/conv2d_subtract_subtract_mish.py` |
| `conv2d_subtract_tanh_subtract_avg_pool` | 🔴 high | `dataset/py_reference/fuse/conv2d_subtract_tanh_subtract_avg_pool.py` |
| `conv2d_tanh_scaling_bias_add_max` | 🔴 high | `dataset/py_reference/fuse/conv2d_tanh_scaling_bias_add_max.py` |
| `conv3d_divide_max_global_avg_pool_bias_add_sum` | 🔴 high | `dataset/py_reference/fuse/conv3d_divide_max_global_avg_pool_bias_add_sum.py` |
| `conv3d_group_norm_mean` | 🔴 high | `dataset/py_reference/fuse/conv3d_group_norm_mean.py` |
| `conv3d_group_norm_min_clamp_dropout` | 🔴 high | `dataset/py_reference/fuse/conv3d_group_norm_min_clamp_dropout.py` |
| `conv3d_hardswish_relu_softmax_mean` | 🔴 high | `dataset/py_reference/fuse/conv3d_hardswish_relu_softmax_mean.py` |
| `conv3d_leaky_relu_sum_clamp_gelu` | 🔴 high | `dataset/py_reference/fuse/conv3d_leaky_relu_sum_clamp_gelu.py` |
| `conv3d_max_log_sum_exp_relu` | 🔴 high | `dataset/py_reference/fuse/conv3d_max_log_sum_exp_relu.py` |
| `conv3d_min_softmax` | 🔴 high | `dataset/py_reference/fuse/conv3d_min_softmax.py` |
| `conv3d_mish_tanh` | 🔴 high | `dataset/py_reference/fuse/conv3d_mish_tanh.py` |
| `conv3d_multiply_instance_norm_clamp_multiply_max` | 🔴 high | `dataset/py_reference/fuse/conv3d_multiply_instance_norm_clamp_multiply_max.py` |
| `conv3d_relu_leaky_relu_gelu_sigmoid_bias_add` | 🔴 high | `dataset/py_reference/fuse/conv3d_relu_leaky_relu_gelu_sigmoid_bias_add.py` |
| `conv3d_scaling_tanh_multiply_sigmoid` | 🔴 high | `dataset/py_reference/fuse/conv3d_scaling_tanh_multiply_sigmoid.py` |
| `conv3d_softmax_max_pool_max_pool` | 🔴 high | `dataset/py_reference/fuse/conv3d_softmax_max_pool_max_pool.py` |
| `conv_transpose2d_add_min_gelu_multiply` | 🔴 high | `dataset/py_reference/fuse/conv_transpose2d_add_min_gelu_multiply.py` |
| `conv_transpose2d_bias_add_clamp_scaling_clamp_divide` | 🔴 high | `dataset/py_reference/fuse/conv_transpose2d_bias_add_clamp_scaling_clamp_divide.py` |
| `conv_transpose2d_gelu_group_norm` | 🔴 high | `dataset/py_reference/fuse/conv_transpose2d_gelu_group_norm.py` |
| `conv_transpose2d_max_pool_hardtanh_mean_tanh` | 🔴 high | `dataset/py_reference/fuse/conv_transpose2d_max_pool_hardtanh_mean_tanh.py` |
| `conv_transpose2d_min_sum_gelu_add` | 🔴 high | `dataset/py_reference/fuse/conv_transpose2d_min_sum_gelu_add.py` |
| `conv_transpose2d_mish_add_hardtanh_scaling` | 🔴 high | `dataset/py_reference/fuse/conv_transpose2d_mish_add_hardtanh_scaling.py` |
| `conv_transpose2d_multiply_global_avg_pool_global_avg_pool_mean` | 🔴 high | `dataset/py_reference/fuse/conv_transpose2d_multiply_global_avg_pool_global_avg_pool_mean.py` |
| `conv_transpose2d_subtract_tanh` | 🔴 high | `dataset/py_reference/fuse/conv_transpose2d_subtract_tanh.py` |
| `conv_transpose3d_add_hard_swish` | 🔴 high | `dataset/py_reference/fuse/conv_transpose3d_add_hard_swish.py` |
| `conv_transpose3d_avg_pool_clamp_softmax_multiply` | 🔴 high | `dataset/py_reference/fuse/conv_transpose3d_avg_pool_clamp_softmax_multiply.py` |
| `conv_transpose3d_batch_norm_avg_pool_avg_pool` | 🔴 high | `dataset/py_reference/fuse/conv_transpose3d_batch_norm_avg_pool_avg_pool.py` |
| `conv_transpose3d_batch_norm_subtract` | 🔴 high | `dataset/py_reference/fuse/conv_transpose3d_batch_norm_subtract.py` |
| `conv_transpose3d_clamp_min_divide` | 🔴 high | `dataset/py_reference/fuse/conv_transpose3d_clamp_min_divide.py` |
| `conv_transpose3d_layer_norm_gelu_scaling` | 🔴 high | `dataset/py_reference/fuse/conv_transpose3d_layer_norm_gelu_scaling.py` |
| `conv_transpose3d_leaky_relu_multiply_leaky_relu_max` | 🔴 high | `dataset/py_reference/fuse/conv_transpose3d_leaky_relu_multiply_leaky_relu_max.py` |
| `conv_transpose3d_log_sum_exp_hard_swish_subtract_clamp_max` | 🔴 high | `dataset/py_reference/fuse/conv_transpose3d_log_sum_exp_hard_swish_subtract_clamp_max.py` |
| `conv_transpose3d_max_max_sum` | 🔴 high | `dataset/py_reference/fuse/conv_transpose3d_max_max_sum.py` |
| `conv_transpose3d_max_pool_softmax_subtract_swish_max` | 🔴 high | `dataset/py_reference/fuse/conv_transpose3d_max_pool_softmax_subtract_swish_max.py` |
| `conv_transpose3d_multiply_max_global_avg_pool_clamp` | 🔴 high | `dataset/py_reference/fuse/conv_transpose3d_multiply_max_global_avg_pool_clamp.py` |
| `conv_transpose3d_scale_batch_norm_global_avg_pool` | 🔴 high | `dataset/py_reference/fuse/conv_transpose3d_scale_batch_norm_global_avg_pool.py` |
| `conv_transpose3d_scaling_avg_pool_bias_add_scaling` | 🔴 high | `dataset/py_reference/fuse/conv_transpose3d_scaling_avg_pool_bias_add_scaling.py` |
| `conv_transpose3d_softmax_sigmoid` | 🔴 high | `dataset/py_reference/fuse/conv_transpose3d_softmax_sigmoid.py` |
| `conv_transpose3d_sum_layer_norm_avg_pool_gelu` | 🔴 high | `dataset/py_reference/fuse/conv_transpose3d_sum_layer_norm_avg_pool_gelu.py` |
| `conv_transpose3d_sum_residual_add_multiply_residual_add` | 🔴 high | `dataset/py_reference/fuse/conv_transpose3d_sum_residual_add_multiply_residual_add.py` |
| `conv_transpose3d_swish_group_norm_hard_swish` | 🔴 high | `dataset/py_reference/fuse/conv_transpose3d_swish_group_norm_hard_swish.py` |
| `convtranspose2d_batchnorm_tanh_maxpool_groupnorm` | 🔴 high | `dataset/py_reference/fuse/convtranspose2d_batchnorm_tanh_maxpool_groupnorm.py` |
| `convtranspose2d_globalavgpool_biasadd_logsumexp_sum_multiply` | 🔴 high | `dataset/py_reference/fuse/convtranspose2d_globalavgpool_biasadd_logsumexp_sum_multiply.py` |
| `convtranspose2d_softmax_biasadd_scaling_sigmoid` | 🔴 high | `dataset/py_reference/fuse/convtranspose2d_softmax_biasadd_scaling_sigmoid.py` |
| `convtranspose3d_mean_add_softmax_tanh_scaling` | 🔴 high | `dataset/py_reference/fuse/convtranspose3d_mean_add_softmax_tanh_scaling.py` |
| `convtranspose3d_relu_groupnorm` | 🔴 high | `dataset/py_reference/fuse/convtranspose3d_relu_groupnorm.py` |
| `gemm_add_relu` | 🔴 high | `dataset/py_reference/fuse/gemm_add_relu.py` |
| `gemm_batch_norm_gelu_group_norm_mean_relu` | 🔴 high | `dataset/py_reference/fuse/gemm_batch_norm_gelu_group_norm_mean_relu.py` |
| `gemm_batch_norm_scaling_softmax` | 🔴 high | `dataset/py_reference/fuse/gemm_batch_norm_scaling_softmax.py` |
| `gemm_bias_add_hardtanh_mish_group_norm` | 🔴 high | `dataset/py_reference/fuse/gemm_bias_add_hardtanh_mish_group_norm.py` |
| `gemm_divide_sum_scaling` | 🔴 high | `dataset/py_reference/fuse/gemm_divide_sum_scaling.py` |
| `gemm_group_norm_hardtanh` | 🔴 high | `dataset/py_reference/fuse/gemm_group_norm_hardtanh.py` |
| `gemm_group_norm_min_bias_add` | 🔴 high | `dataset/py_reference/fuse/gemm_group_norm_min_bias_add.py` |
| `gemm_group_norm_swish_multiply_swish` | 🔴 high | `dataset/py_reference/fuse/gemm_group_norm_swish_multiply_swish.py` |
| `gemm_log_sum_exp_leaky_relu_leaky_relu_gelu_gelu` | 🔴 high | `dataset/py_reference/fuse/gemm_log_sum_exp_leaky_relu_leaky_relu_gelu_gelu.py` |
| `gemm_max_subtract_gelu` | 🔴 high | `dataset/py_reference/fuse/gemm_max_subtract_gelu.py` |
| `gemm_multiply_leakyrelu` | 🔴 high | `dataset/py_reference/fuse/gemm_multiply_leakyrelu.py` |
| `gemm_relu_divide` | 🔴 high | `dataset/py_reference/fuse/gemm_relu_divide.py` |
| `gemm_scale_batch_norm` | 🔴 high | `dataset/py_reference/fuse/gemm_scale_batch_norm.py` |
| `gemm_scale_batchnorm` | 🔴 high | `dataset/py_reference/fuse/gemm_scale_batchnorm.py` |
| `gemm_scaling_hard_tanh_gelu` | 🔴 high | `dataset/py_reference/fuse/gemm_scaling_hard_tanh_gelu.py` |
| `gemm_sigmoid_scaling_residual_add` | 🔴 high | `dataset/py_reference/fuse/gemm_sigmoid_scaling_residual_add.py` |
| `gemm_sigmoid_sum_log_sum_exp` | 🔴 high | `dataset/py_reference/fuse/gemm_sigmoid_sum_log_sum_exp.py` |
| `gemm_subtract_global_avg_pool_log_sum_exp_gelu_residual_add` | 🔴 high | `dataset/py_reference/fuse/gemm_subtract_global_avg_pool_log_sum_exp_gelu_residual_add.py` |
| `gemm_swish_divide_clamp_tanh_clamp` | 🔴 high | `dataset/py_reference/fuse/gemm_swish_divide_clamp_tanh_clamp.py` |
| `matmul_add_swish_tanh_gelu_hardtanh` | 🔴 high | `dataset/py_reference/fuse/matmul_add_swish_tanh_gelu_hardtanh.py` |
| `matmul_avg_pool_gelu_scale_max` | 🔴 high | `dataset/py_reference/fuse/matmul_avg_pool_gelu_scale_max.py` |
| `matmul_batch_norm_bias_add_divide_swish` | 🔴 high | `dataset/py_reference/fuse/matmul_batch_norm_bias_add_divide_swish.py` |
| `matmul_divide_gelu` | 🔴 high | `dataset/py_reference/fuse/matmul_divide_gelu.py` |
| `matmul_dropout_mean_softmax` | 🔴 high | `dataset/py_reference/fuse/matmul_dropout_mean_softmax.py` |
| `matmul_gelu_softmax` | 🔴 high | `dataset/py_reference/fuse/matmul_gelu_softmax.py` |
| `matmul_group_norm_leaky_relu_sum` | 🔴 high | `dataset/py_reference/fuse/matmul_group_norm_leaky_relu_sum.py` |
| `matmul_max_pool_sum_scale` | 🔴 high | `dataset/py_reference/fuse/matmul_max_pool_sum_scale.py` |
| `matmul_min_subtract` | 🔴 high | `dataset/py_reference/fuse/matmul_min_subtract.py` |
| `matmul_mish_mish` | 🔴 high | `dataset/py_reference/fuse/matmul_mish_mish.py` |
| `matmul_scale_residual_add_clamp_log_sum_exp_mish` | 🔴 high | `dataset/py_reference/fuse/matmul_scale_residual_add_clamp_log_sum_exp_mish.py` |
| `matmul_scaling_residual_add` | 🔴 high | `dataset/py_reference/fuse/matmul_scaling_residual_add.py` |
| `matmul_sigmoid_sum` | 🔴 high | `dataset/py_reference/fuse/matmul_sigmoid_sum.py` |
| `matmul_subtract_multiply_relu` | 🔴 high | `dataset/py_reference/fuse/matmul_subtract_multiply_relu.py` |
| `matmul_sum_max_avg_pool_log_sum_exp_log_sum_exp` | 🔴 high | `dataset/py_reference/fuse/matmul_sum_max_avg_pool_log_sum_exp_log_sum_exp.py` |
| `matmul_swish_scaling` | 🔴 high | `dataset/py_reference/fuse/matmul_swish_scaling.py` |
| `matmul_swish_sum_group_norm` | 🔴 high | `dataset/py_reference/fuse/matmul_swish_sum_group_norm.py` |

### 网络架构 (Architecture)

| 算子名称 | 复杂度 | 路径 |
|----------|--------|------|
| `alexnet` | 🔴 high | `dataset/py_reference/arch/alexnet.py` |
| `convolutional_vision_transformer` | 🔴 high | `dataset/py_reference/arch/convolutional_vision_transformer.py` |
| `deep_narrow_mlp` | 🔴 high | `dataset/py_reference/arch/deep_narrow_mlp.py` |
| `densenet121` | 🔴 high | `dataset/py_reference/arch/densenet121.py` |
| `densenet121_dense_block` | 🔴 high | `dataset/py_reference/arch/densenet121_dense_block.py` |
| `densenet121_transition_layer` | 🔴 high | `dataset/py_reference/arch/densenet121_transition_layer.py` |
| `densenet201` | 🔴 high | `dataset/py_reference/arch/densenet201.py` |
| `efficientnet_b0` | 🔴 high | `dataset/py_reference/arch/efficientnet_b0.py` |
| `efficientnet_b1` | 🔴 high | `dataset/py_reference/arch/efficientnet_b1.py` |
| `efficientnet_b2` | 🔴 high | `dataset/py_reference/arch/efficientnet_b2.py` |
| `efficientnet_mb_conv` | 🔴 high | `dataset/py_reference/arch/efficientnet_mb_conv.py` |
| `googlenet_inception_module` | 🔴 high | `dataset/py_reference/arch/googlenet_inception_module.py` |
| `googlenet_inception_v1` | 🔴 high | `dataset/py_reference/arch/googlenet_inception_v1.py` |
| `gru` | 🔴 high | `dataset/py_reference/arch/gru.py` |
| `gru_bidirectional_hidden` | 🔴 high | `dataset/py_reference/arch/gru_bidirectional_hidden.py` |
| `gru_birectional` | 🔴 high | `dataset/py_reference/arch/gru_birectional.py` |
| `gru_hidden` | 🔴 high | `dataset/py_reference/arch/gru_hidden.py` |
| `lenet5` | 🔴 high | `dataset/py_reference/arch/lenet5.py` |
| `ltsm` | 🔴 high | `dataset/py_reference/arch/ltsm.py` |
| `ltsm_bidirectional` | 🔴 high | `dataset/py_reference/arch/ltsm_bidirectional.py` |
| `ltsm_cn` | 🔴 high | `dataset/py_reference/arch/ltsm_cn.py` |
| `ltsm_hn` | 🔴 high | `dataset/py_reference/arch/ltsm_hn.py` |
| `mamba_return_final_state` | 🔴 high | `dataset/py_reference/arch/mamba_return_final_state.py` |
| `mamba_return_y` | 🔴 high | `dataset/py_reference/arch/mamba_return_y.py` |
| `min_gpt_causal_attention` | 🔴 high | `dataset/py_reference/arch/min_gpt_causal_attention.py` |
| `mini_gpt_block` | 🔴 high | `dataset/py_reference/arch/mini_gpt_block.py` |
| `mlp` | 🔴 high | `dataset/py_reference/arch/mlp.py` |
| `mobilenet_v1` | 🔴 high | `dataset/py_reference/arch/mobilenet_v1.py` |
| `mobilenet_v2` | 🔴 high | `dataset/py_reference/arch/mobilenet_v2.py` |
| `net_vlad_no_ghost_clusters` | 🔴 high | `dataset/py_reference/arch/net_vlad_no_ghost_clusters.py` |
| `net_vlad_with_ghost_clusters` | 🔴 high | `dataset/py_reference/arch/net_vlad_with_ghost_clusters.py` |
| `regnet` | 🔴 high | `dataset/py_reference/arch/regnet.py` |
| `relu_self_attention` | 🔴 high | `dataset/py_reference/arch/relu_self_attention.py` |
| `resnet101` | 🔴 high | `dataset/py_reference/arch/resnet101.py` |
| `resnet18` | 🔴 high | `dataset/py_reference/arch/resnet18.py` |
| `resnet_basic_block` | 🔴 high | `dataset/py_reference/arch/resnet_basic_block.py` |
| `shallow_wide_mlp` | 🔴 high | `dataset/py_reference/arch/shallow_wide_mlp.py` |
| `shufflenet` | 🔴 high | `dataset/py_reference/arch/shufflenet.py` |
| `shufflenet_unit` | 🔴 high | `dataset/py_reference/arch/shufflenet_unit.py` |
| `squeeze_net` | 🔴 high | `dataset/py_reference/arch/squeeze_net.py` |
| `squeeze_net_fire_module` | 🔴 high | `dataset/py_reference/arch/squeeze_net_fire_module.py` |
| `swin_mlp` | 🔴 high | `dataset/py_reference/arch/swin_mlp.py` |
| `swintransformer_v2` | 🔴 high | `dataset/py_reference/arch/swintransformer_v2.py` |
| `unet_softmax` | 🔴 high | `dataset/py_reference/arch/unet_softmax.py` |
| `vanilla_rnn` | 🔴 high | `dataset/py_reference/arch/vanilla_rnn.py` |
| `vanilla_rnn_hidden` | 🔴 high | `dataset/py_reference/arch/vanilla_rnn_hidden.py` |
| `vgg16` | 🔴 high | `dataset/py_reference/arch/vgg16.py` |
| `vgg19` | 🔴 high | `dataset/py_reference/arch/vgg19.py` |
| `vision_attention` | 🔴 high | `dataset/py_reference/arch/vision_attention.py` |
| `vision_transformer` | 🔴 high | `dataset/py_reference/arch/vision_transformer.py` |

---

## 复杂度说明

- 🟢 **Low**: 简单算子，通常只涉及逐元素操作
- 🟡 **Medium**: 中等复杂度，涉及多维操作或简单的数据依赖
- 🔴 **High**: 高复杂度，融合算子或完整网络架构
