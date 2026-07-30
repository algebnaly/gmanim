fn aces_tone_map(color: vec3<f32>) -> vec3<f32> {
    let a = 2.51;
    let b = 0.03;
    let c = 2.43;
    let d = 0.59;
    let e = 0.14;
    return clamp(
        (color * (a * color + vec3<f32>(b)))
            / (color * (c * color + vec3<f32>(d)) + vec3<f32>(e)),
        vec3<f32>(0.0),
        vec3<f32>(1.0),
    );
}

fn linear_to_bt709(linear: vec3<f32>) -> vec3<f32> {
    let low = linear * 4.5;
    let high = vec3<f32>(1.099) * pow(linear, vec3<f32>(0.45)) - vec3<f32>(0.099);
    return select(high, low, linear < vec3<f32>(0.018));
}

fn hdr_to_bt709(color: vec3<f32>) -> vec3<f32> {
    return linear_to_bt709(aces_tone_map(color));
}

fn bt709_to_yuv_limited(rgb: vec3<f32>) -> vec3<f32> {
    let y = 16.0 / 255.0
        + 0.182586 * rgb.r
        + 0.614231 * rgb.g
        + 0.062007 * rgb.b;
    let u = 128.0 / 255.0
        - 0.100644 * rgb.r
        - 0.338572 * rgb.g
        + 0.439216 * rgb.b;
    let v = 128.0 / 255.0
        + 0.439216 * rgb.r
        - 0.398942 * rgb.g
        - 0.040274 * rgb.b;
    return vec3<f32>(y, u, v);
}
