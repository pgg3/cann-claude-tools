"""
CANN Solution Templates.

Provides template generation for Ascend C operators (Vector and Cube).
"""

from typing import Literal

# Operator type classification
OperatorType = Literal["vector", "cube"]

# Known Cube operators
CUBE_OPERATORS = {
    "matmul", "mat_mul", "batch_matmul", "gemm",
    "conv2d", "conv3d", "conv1d",
    "batch_gemm", "bmm",
}


def detect_operator_type(op_name: str) -> OperatorType:
    """Detect operator type based on name.

    Args:
        op_name: Operator name (e.g., "relu", "matmul")

    Returns:
        "vector" or "cube"
    """
    op_lower = op_name.lower().replace("-", "_")
    if op_lower in CUBE_OPERATORS:
        return "cube"
    return "vector"


def generate_solution_template(
    op_name: str,
    npu_type: str = "Ascend910B2",
    op_type: OperatorType = None,
) -> dict:
    """Generate a solution template with correct format for the given operator.

    Args:
        op_name: Operator name (e.g., "relu", "matmul")
        npu_type: NPU type (default: Ascend910B2)
        op_type: Force operator type ("vector" or "cube"), auto-detect if None

    Returns:
        dict with keys: kernel_impl, kernel_entry_body, tiling_fields,
                       tiling_func_body, infer_shape_body, output_alloc_code
    """
    if op_type is None:
        op_type = detect_operator_type(op_name)

    if op_type == "cube":
        return _generate_cube_template(op_name, npu_type)
    else:
        return _generate_vector_template(op_name, npu_type)


def _generate_vector_template(op_name: str, npu_type: str) -> dict:
    """Generate Vector operator template."""
    class_name = "".join(word.capitalize() for word in op_name.split("_"))
    kernel_class = f"Kernel{class_name}"
    tiling_data_class = f"{class_name}CustomTilingData"

    # NPU-specific UB sizes (in KB)
    npu_ub_sizes = {
        "Ascend910B": 256,
        "Ascend910B2": 256,
        "Ascend910B3": 256,
        "Ascend310P": 256,
    }
    ub_size_kb = npu_ub_sizes.get(npu_type, 256)
    ub_safe_kb = ub_size_kb // 4

    kernel_impl = _generate_vector_kernel_impl(kernel_class)
    kernel_entry_body = _generate_vector_kernel_entry_body(kernel_class)
    tiling_func_body = _generate_vector_tiling_func_body(
        tiling_data_class, npu_type, ub_size_kb, ub_safe_kb
    )
    infer_shape_body = _generate_vector_infer_shape_body()

    return {
        "kernel_impl": kernel_impl,
        "kernel_entry_body": kernel_entry_body,
        "tiling_fields": [
            {"type": "uint32_t", "name": "totalLength"},
            {"type": "uint32_t", "name": "tileNum"},
        ],
        "tiling_func_body": tiling_func_body,
        "infer_shape_body": infer_shape_body,
        "output_alloc_code": "at::Tensor result = at::empty_like(x);",
        "_operator_type": "vector",
    }


def _generate_cube_template(op_name: str, npu_type: str) -> dict:
    """Generate Cube operator template (MatMul)."""
    class_name = "".join(word.capitalize() for word in op_name.split("_"))
    kernel_class = f"Kernel{class_name}"
    tiling_data_class = f"{class_name}CustomTilingData"

    kernel_impl = _generate_cube_kernel_impl(kernel_class)
    kernel_entry_body = _generate_cube_kernel_entry_body(kernel_class)
    tiling_func_body = _generate_cube_tiling_func_body(tiling_data_class, npu_type)
    infer_shape_body = _generate_cube_infer_shape_body()

    return {
        "kernel_impl": kernel_impl,
        "kernel_entry_body": kernel_entry_body,
        "tiling_fields": [
            {"type": "uint32_t", "name": "M"},
            {"type": "uint32_t", "name": "K"},
            {"type": "uint32_t", "name": "N"},
            {"type": "uint32_t", "name": "M_tile"},
            {"type": "uint32_t", "name": "K_tile"},
            {"type": "uint32_t", "name": "N_tile"},
        ],
        "tiling_func_body": tiling_func_body,
        "infer_shape_body": infer_shape_body,
        "output_alloc_code": "at::Tensor result = at::empty({x.size(0), y.size(1)}, x.options());",
        "_operator_type": "cube",
    }


# ============== Vector Template Functions ==============

def _generate_vector_kernel_impl(kernel_class: str) -> str:
    """Generate Vector kernel implementation template."""
    return f'''using namespace AscendC;
constexpr int32_t BUFFER_NUM = 2;

class {kernel_class} {{
public:
    __aicore__ inline {kernel_class}() {{}}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, uint32_t totalLength, uint32_t tileNum) {{
        this->blockLength = totalLength / GetBlockNum();
        this->tileNum = tileNum;
        this->tileLength = this->blockLength / tileNum / BUFFER_NUM;

        xGm.SetGlobalBuffer((__gm__ float*)x + this->blockLength * GetBlockIdx(), this->blockLength);
        yGm.SetGlobalBuffer((__gm__ float*)y + this->blockLength * GetBlockIdx(), this->blockLength);

        pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileLength * sizeof(float));
        pipe.InitBuffer(outQueueY, BUFFER_NUM, this->tileLength * sizeof(float));
    }}

    __aicore__ inline void Process() {{
        int32_t loopCount = this->tileNum * BUFFER_NUM;
        for (int32_t i = 0; i < loopCount; i++) {{
            CopyIn(i);
            Compute(i);
            CopyOut(i);
        }}
    }}

private:
    __aicore__ inline void CopyIn(int32_t progress) {{
        LocalTensor<float> xLocal = inQueueX.AllocTensor<float>();
        DataCopy(xLocal, xGm[progress * this->tileLength], this->tileLength);
        inQueueX.EnQue(xLocal);
    }}

    __aicore__ inline void Compute(int32_t progress) {{
        LocalTensor<float> xLocal = inQueueX.DeQue<float>();
        LocalTensor<float> yLocal = outQueueY.AllocTensor<float>();

        // TODO: Replace this with your computation logic
        // For ReLU: Relu(yLocal, xLocal, this->tileLength);
        // For Abs:  Abs(yLocal, xLocal, this->tileLength);
        // For Exp:  Exp(yLocal, xLocal, this->tileLength);

        outQueueY.EnQue(yLocal);
        inQueueX.FreeTensor(xLocal);
    }}

    __aicore__ inline void CopyOut(int32_t progress) {{
        LocalTensor<float> yLocal = outQueueY.DeQue<float>();
        DataCopy(yGm[progress * this->tileLength], yLocal, this->tileLength);
        outQueueY.FreeTensor(yLocal);
    }}

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueY;
    GlobalTensor<float> xGm, yGm;
    uint32_t blockLength, tileNum, tileLength;
}};'''


def _generate_vector_kernel_entry_body(kernel_class: str) -> str:
    """Generate Vector kernel entry body template."""
    return f'''    {kernel_class} op;
    op.Init(x, output, tilingData.totalLength, tilingData.tileNum);
    op.Process();'''


def _generate_vector_tiling_func_body(
    tiling_data_class: str, npu_type: str, ub_size_kb: int, ub_safe_kb: int
) -> str:
    """Generate Vector tiling function body template."""
    return f'''    {tiling_data_class} tiling;

    auto inputShape = context->GetInputShape(0);
    if (inputShape == nullptr) {{
        return ge::GRAPH_FAILED;
    }}
    auto shape = inputShape->GetStorageShape();
    uint32_t totalLength = static_cast<uint32_t>(shape.GetShapeSize());

    // ========== DYNAMIC TILING FOR {npu_type} (VECTOR) ==========
    // UB safe size: {ub_safe_kb}KB (1/4 of {ub_size_kb}KB total UB)
    constexpr uint32_t UB_SAFE_SIZE = {ub_safe_kb} * 1024;
    constexpr uint32_t BUFFER_NUM = 2;
    constexpr uint32_t NUM_BUFFERS = 2;  // input + output
    constexpr uint32_t BLOCK_DIM = 8;
    uint32_t elementSize = sizeof(float);

    // Calculate max elements per tile that fit in UB
    uint32_t maxTileElements = UB_SAFE_SIZE / (NUM_BUFFERS * BUFFER_NUM * elementSize);
    maxTileElements = (maxTileElements / 8) * 8;  // Align to 32 bytes

    // Calculate tileNum based on data size
    uint32_t blockLength = totalLength / BLOCK_DIM;
    uint32_t tileNum = (blockLength + maxTileElements - 1) / maxTileElements;
    tileNum = tileNum > 0 ? tileNum : 1;
    // ============================================================

    tiling.set_totalLength(totalLength);
    tiling.set_tileNum(tileNum);

    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(),
                        context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    context->SetBlockDim(BLOCK_DIM);

    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = 0;

    return ge::GRAPH_SUCCESS;'''


def _generate_vector_infer_shape_body() -> str:
    """Generate Vector infer shape body template."""
    return '''    const gert::Shape* x_shape = context->GetInputShape(0);
    gert::Shape* y_shape = context->GetOutputShape(0);
    *y_shape = *x_shape;
    return ge::GRAPH_SUCCESS;'''


# ============== Cube Template Functions ==============

def _generate_cube_kernel_impl(kernel_class: str) -> str:
    """Generate Cube kernel implementation template (MatMul)."""
    return f'''using namespace AscendC;

// Cube operator for matrix multiplication: C[M,N] = A[M,K] x B[K,N]
class {kernel_class} {{
public:
    __aicore__ inline {kernel_class}() {{}}

    __aicore__ inline void Init(
        GM_ADDR a, GM_ADDR b, GM_ADDR c,
        uint32_t M, uint32_t K, uint32_t N,
        uint32_t M_tile, uint32_t K_tile, uint32_t N_tile
    ) {{
        this->M = M;
        this->K = K;
        this->N = N;
        this->M_tile = M_tile;
        this->K_tile = K_tile;
        this->N_tile = N_tile;

        // Set global memory buffers
        aGm.SetGlobalBuffer((__gm__ half*)a, M * K);
        bGm.SetGlobalBuffer((__gm__ half*)b, K * N);
        cGm.SetGlobalBuffer((__gm__ half*)c, M * N);

        // Initialize L1 buffers for A and B tiles
        pipe.InitBuffer(l1BufA, M_tile * K_tile * sizeof(half));
        pipe.InitBuffer(l1BufB, K_tile * N_tile * sizeof(half));
        pipe.InitBuffer(l1BufC, M_tile * N_tile * sizeof(half));
    }}

    __aicore__ inline void Process() {{
        // TODO: Implement tiled matrix multiplication
        // Outer loops over M, N tiles
        // Inner loop over K tiles for accumulation
        //
        // Pseudocode:
        // for (m_idx = 0; m_idx < M; m_idx += M_tile)
        //   for (n_idx = 0; n_idx < N; n_idx += N_tile)
        //     Initialize C tile to zero
        //     for (k_idx = 0; k_idx < K; k_idx += K_tile)
        //       Load A[m_idx:m_idx+M_tile, k_idx:k_idx+K_tile] to L1
        //       Load B[k_idx:k_idx+K_tile, n_idx:n_idx+N_tile] to L1
        //       Mmad(C_tile, A_tile, B_tile)  // Accumulate
        //     Store C tile to GM
    }}

private:
    TPipe pipe;
    TBuf<TPosition::L1> l1BufA, l1BufB, l1BufC;
    GlobalTensor<half> aGm, bGm, cGm;
    uint32_t M, K, N;
    uint32_t M_tile, K_tile, N_tile;
}};'''


def _generate_cube_kernel_entry_body(kernel_class: str) -> str:
    """Generate Cube kernel entry body template."""
    return f'''    {kernel_class} op;
    op.Init(x, y, output,
            tilingData.M, tilingData.K, tilingData.N,
            tilingData.M_tile, tilingData.K_tile, tilingData.N_tile);
    op.Process();'''


def _generate_cube_tiling_func_body(tiling_data_class: str, npu_type: str) -> str:
    """Generate Cube tiling function body template."""
    return f'''    {tiling_data_class} tiling;

    // Get matrix dimensions from input shapes
    // A[M, K], B[K, N] -> C[M, N]
    auto shapeA = context->GetInputShape(0);
    auto shapeB = context->GetInputShape(1);
    if (shapeA == nullptr || shapeB == nullptr) {{
        return ge::GRAPH_FAILED;
    }}

    auto dimA = shapeA->GetStorageShape();
    auto dimB = shapeB->GetStorageShape();

    uint32_t M = static_cast<uint32_t>(dimA.GetDim(0));
    uint32_t K = static_cast<uint32_t>(dimA.GetDim(1));
    uint32_t N = static_cast<uint32_t>(dimB.GetDim(1));

    // ========== DYNAMIC TILING FOR {npu_type} (CUBE) ==========
    // Buffer constraints:
    // L0A: 64KB, L0B: 64KB, L0C: 256KB
    // L1: 1MB (shared for A and B tiles)
    constexpr uint32_t L0A_SIZE = 64 * 1024;
    constexpr uint32_t L0B_SIZE = 64 * 1024;
    constexpr uint32_t L0C_SIZE = 256 * 1024;
    constexpr uint32_t CUBE_BLOCK = 16;  // Alignment for float16
    uint32_t elementSize = sizeof(half);  // 2 bytes

    // Calculate max tile sizes based on buffer constraints
    // M_tile * K_tile * elementSize <= L0A_SIZE
    // K_tile * N_tile * elementSize <= L0B_SIZE
    // M_tile * N_tile * elementSize <= L0C_SIZE

    // Default tile sizes (can be optimized)
    uint32_t M_tile = 128;
    uint32_t K_tile = 256;
    uint32_t N_tile = 128;

    // Ensure tiles fit in buffers
    while (M_tile * K_tile * elementSize > L0A_SIZE && M_tile > CUBE_BLOCK) {{
        M_tile /= 2;
    }}
    while (K_tile * N_tile * elementSize > L0B_SIZE && K_tile > CUBE_BLOCK) {{
        K_tile /= 2;
    }}
    while (M_tile * N_tile * elementSize > L0C_SIZE && N_tile > CUBE_BLOCK) {{
        N_tile /= 2;
    }}

    // Align to Cube block size
    M_tile = (M_tile / CUBE_BLOCK) * CUBE_BLOCK;
    K_tile = (K_tile / CUBE_BLOCK) * CUBE_BLOCK;
    N_tile = (N_tile / CUBE_BLOCK) * CUBE_BLOCK;

    // Ensure minimum tile size
    M_tile = M_tile > 0 ? M_tile : CUBE_BLOCK;
    K_tile = K_tile > 0 ? K_tile : CUBE_BLOCK;
    N_tile = N_tile > 0 ? N_tile : CUBE_BLOCK;
    // ==========================================================

    tiling.set_M(M);
    tiling.set_K(K);
    tiling.set_N(N);
    tiling.set_M_tile(M_tile);
    tiling.set_K_tile(K_tile);
    tiling.set_N_tile(N_tile);

    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(),
                        context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    context->SetBlockDim(1);  // Cube typically uses single block

    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = 0;

    return ge::GRAPH_SUCCESS;'''


def _generate_cube_infer_shape_body() -> str:
    """Generate Cube infer shape body template (MatMul: [M,K] x [K,N] -> [M,N])."""
    return '''    // MatMul: A[M,K] x B[K,N] -> C[M,N]
    const gert::Shape* a_shape = context->GetInputShape(0);
    const gert::Shape* b_shape = context->GetInputShape(1);
    gert::Shape* c_shape = context->GetOutputShape(0);

    // Output shape: [M, N]
    c_shape->SetDimNum(2);
    c_shape->SetDim(0, a_shape->GetDim(0));  // M
    c_shape->SetDim(1, b_shape->GetDim(1));  // N

    return ge::GRAPH_SUCCESS;'''
