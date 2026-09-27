#include <torch/extension.h>

#include "ply_utils_torch.h"
#include "rendering.h"

namespace py = pybind11;

struct GaussianModel {
    torch::Tensor anchors_;      // [N, 3]
    torch::Tensor offsets_;      // [N, 3]
    torch::Tensor scaling_;      // [N, 3] log-scale
    torch::Tensor quaternion_;   // [N, 4]
    torch::Tensor opacity_;      // [N] logit-opacity
    torch::Tensor features_dc_;  // [N, 1, 3]
    torch::Tensor features_rest_;// [N, D-1, 3]
    int sh_degree_ = 0;
    int num_gaussians_ = 0;
    torch::Device device_ = torch::kCPU;

    GaussianModel() = default;

    bool load_ply(const std::string &ply_path, int sh_degree, const std::string &device_str) {
        device_ = torch::Device(device_str);
        sh_degree_ = sh_degree;

        std::unique_ptr<std::istream> file_stream =
            std::make_unique<std::ifstream>(ply_path, std::ios::binary);
        if (!file_stream || file_stream->fail()) {
            std::cerr << "Failed to open file: " << ply_path << "\n";
            return false;
        }

        tinyply::PlyFile file;
        file.parse_header(*file_stream);

        std::shared_ptr<tinyply::PlyData> xyzs, f_dc, f_rest, opacities, scale, quat;

        try { xyzs = file.request_properties_from_element("vertex", {"x", "y", "z"}); }
        catch (const std::exception &e) {}

        int dim_dc = 3;
        try {
            std::vector<std::string> dc_keys;
            for (int i = 0; i < dim_dc; i++) dc_keys.push_back("f_dc_" + std::to_string(i));
            f_dc = file.request_properties_from_element("vertex", dc_keys);
        } catch (const std::exception &e) {}

        if (sh_degree > 0) {
            try {
                int dim_sh = (sh_degree + 1) * (sh_degree + 1);
                std::vector<std::string> rest_keys;
                for (int i = 0; i < (dim_sh - 1) * 3; i++)
                    rest_keys.push_back("f_rest_" + std::to_string(i));
                f_rest = file.request_properties_from_element("vertex", rest_keys);
            } catch (const std::exception &e) {}
        }

        try { opacities = file.request_properties_from_element("vertex", {"opacity"}); }
        catch (const std::exception &e) {}

        try { scale = file.request_properties_from_element("vertex", {"scale_0", "scale_1", "scale_2"}); }
        catch (const std::exception &e) {}

        try { quat = file.request_properties_from_element("vertex", {"rot_0", "rot_1", "rot_2", "rot_3"}); }
        catch (const std::exception &e) {}

        file.read(*file_stream);

        if (!xyzs) { std::cerr << "Missing xyz in PLY\n"; return false; }

        if (sh_degree_ > 0 && (!f_rest || !f_rest->buffer.get())) {
            std::cerr << "Missing spherical-harmonic coefficients for degree "
                      << sh_degree_ << "\n";
            return false;
        }

        auto float32_to_tensor = [&](std::shared_ptr<tinyply::PlyData> &data,
                                      std::vector<int64_t> shape) -> torch::Tensor {
            if (!data || data->t != tinyply::Type::FLOAT32) return torch::Tensor();
            return torch::from_blob(data->buffer.get(),
                                    torch::IntArrayRef(shape),
                                    torch::kFloat32).clone().to(device_).contiguous();
        };

        int N = xyzs->count;
        num_gaussians_ = N;

        anchors_ = float32_to_tensor(xyzs, {N, 3});
        if (!anchors_.defined()) { std::cerr << "Failed to load xyz\n"; return false; }
        offsets_ = torch::zeros_like(anchors_);

        auto f_dc_t = float32_to_tensor(f_dc, {N, dim_dc});
        if (f_dc_t.defined()) {
            features_dc_ = f_dc_t.view({N, -1, 1}).transpose(1, 2).contiguous();  // [N, 1, 3]
        } else {
            features_dc_ = torch::zeros({N, 1, 3}, device_);
        }

        if (sh_degree > 0 && f_rest) {
            int dim_sh = (sh_degree + 1) * (sh_degree + 1);
            auto f_rest_t = float32_to_tensor(f_rest, {N, (dim_sh - 1) * 3});
            if (f_rest_t.defined()) {
                features_rest_ = f_rest_t.view({N, 3, dim_sh - 1})
                                     .transpose(1, 2).contiguous();  // [N, D-1, 3]
            } else {
                features_rest_ = torch::zeros({N, 0, 3}, device_);
            }
        } else {
            features_rest_ = torch::zeros({N, 0, 3}, device_);
        }

        auto scale_t = float32_to_tensor(scale, {N, 3});
        if (scale_t.defined()) {
            scaling_ = scale_t;
        } else {
            scaling_ = torch::zeros({N, 3}, device_);
        }

        auto opa_t = float32_to_tensor(opacities, {N, 1});
        if (opa_t.defined()) {
            opacity_ = opa_t.squeeze(-1);
        } else {
            opacity_ = torch::zeros({N}, device_);
        }

        auto quat_t = float32_to_tensor(quat, {N, 4});
        if (quat_t.defined()) {
            quaternion_ = quat_t;
        } else {
            quaternion_ = torch::zeros({N, 4}, device_);
            quaternion_.index({"...", 0}) = 1.0f;  // WXYZ identity quaternion
        }

        std::cout << "Loaded " << N << " gaussians from " << ply_path << "\n";
        return true;
    }

    std::map<std::string, torch::Tensor> render(
        const torch::Tensor &pose_cam2world,  // [B, 3, 4]
        double fx, double fy, double cx, double cy,
        int width, int height,
        double near_plane, double far_plane,
        int background,
        bool packed
    ) {
        float fx_f = static_cast<float>(fx);
        float fy_f = static_cast<float>(fy);
        float cx_f = static_cast<float>(cx);
        float cy_f = static_cast<float>(cy);
        float near_f = static_cast<float>(near_plane);
        float far_f = static_cast<float>(far_plane);
        int B = pose_cam2world.size(0);
        auto device = anchors_.device();

        // Build intrinsics matrix [B, 3, 3]
        auto K = torch::tensor({{fx_f, 0.f, cx_f},
                                {0.f, fy_f, cy_f},
                                {0.f, 0.f, 1.f}}, device)
                      .unsqueeze(0)
                      .repeat({B, 1, 1});

        // Build world-to-camera matrices [B, 4, 4]
        // pose_cam2world: [B, 3, 4] = [R | t] (camera-to-world)
        // world2cam = [R^T | -R^T * t]
        auto rot = pose_cam2world.index({"...", torch::indexing::Slice(0, 3)});  // [B, 3, 3]
        auto trans = pose_cam2world.index({"...", 3}).unsqueeze(-1);              // [B, 3, 1]
        auto rot_t = rot.transpose(1, 2);
        auto viewmats = torch::cat({
            torch::cat({rot_t, -rot_t.bmm(trans)}, 2),
            torch::tensor({{0.f, 0.f, 0.f, 1.f}}, device).unsqueeze(0).repeat({B, 1, 1})
        }, 1);

        // Generate Gaussian parameters (from log/logit to actual)
        auto means = anchors_ + offsets_;
        auto quats = quaternion_.view({-1, 4});
        auto scales = torch::exp(scaling_.view({-1, 3}));
        auto opacities = torch::sigmoid(opacity_.view({-1}));
        auto sh_features = torch::cat({features_dc_, features_rest_}, 1);  // [N, SH_COUNT, 3]

        auto background_tensor = torch::zeros({height, width, 3}, device);

        if (background == 2) {
            background_tensor = torch::rand({height, width, 3}, device);
        } else if (background == 1) {
            background_tensor = torch::ones({height, width, 3}, device);
        }

        // Call rasterization
        auto [renders, alphas, meta] = gsplat_cpp::rasterization_2dgs(
            means, quats, scales, opacities, sh_features,
            viewmats, K,
            width, height, "RGB+ED",
            near_plane, far_plane, 0.0f,
            sh_degree_ >= 0 ? torch::optional<int>(sh_degree_) : torch::nullopt,
            packed, 16,
            at::nullopt,
            false, false, false,
            {}
        );

        // renders: [B, H, W, 4] -> RGB + Expected Depth
        // alphas:  [B, H, W, 1]

        torch::Tensor color, depth;
        if (renders.size(-1) == 4) {
            color = renders.index({"...", torch::indexing::Slice(0, 3)});
            depth = renders.index({"...", 3}).unsqueeze(-1);
        } else if (renders.size(-1) > 4) {
            int total_channels = renders.size(-1);
            color = renders.index({"...", torch::indexing::Slice(0, 3)});
            depth = renders.index({"...", total_channels - 1}).unsqueeze(-1);
        } else {
            color = renders;
            depth = torch::zeros({B, height, width, 1}, device);
        }

        if (background == 2) {
            auto bg = torch::rand({B, height, width, 3}, device);
            color = color + (1.f - alphas) * bg;
        } else if (background == 1) {
            auto bg = torch::ones({B, height, width, 3}, device);
            color = color + (1.f - alphas) * bg;
        }

        std::map<std::string, torch::Tensor> result;
        result["color"] = color;
        result["depth"] = depth;
        result["alpha"] = alphas;
        return result;
    }
};

PYBIND11_MODULE(_gs_bridge, m) {
    m.doc() = "GS-SDF Gaussian Splatting Renderer Bridge";

    py::class_<GaussianModel>(m, "GaussianModel")
        .def(py::init<>())
        .def("load_ply", &GaussianModel::load_ply,
             py::arg("ply_path"),
             py::arg("sh_degree") = 0,
             py::arg("device") = "cuda:0")
        .def("render", &GaussianModel::render,
             py::arg("pose_cam2world"),
             py::arg("fx"), py::arg("fy"),
             py::arg("cx"), py::arg("cy"),
             py::arg("width"), py::arg("height"),
             py::arg("near_plane") = 0.01,
             py::arg("far_plane") = 1000.0,
             py::arg("background") = 0,
             py::arg("packed") = true);
}
