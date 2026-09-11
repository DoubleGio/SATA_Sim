
def compute_conv_output_size(H, W, K_h, K_w, stride, padding='same'):
    if padding == 'same':
        out_H = (H + stride - 1) // stride
        out_W = (W + stride - 1) // stride
    elif isinstance(padding, int):
        out_H = (H + 2 * padding - K_h) // stride + 1
        out_W = (W + 2 * padding - K_w) // stride + 1
    else:
        raise ValueError(f"Unsupported padding type: {padding}")
    return out_H, out_W