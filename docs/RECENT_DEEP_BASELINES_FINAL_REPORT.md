# Recent Deep Baselines Final Report

Updated: 2026-05-19

## Completed Experiment

- Hardware: NVIDIA GeForce RTX 4080 SUPER on AutoDL
- Backend: PyTorch CUDA
- Added recent deep TSAD adapters: `CATCH`, `DCdetector`, `TranAD`
- Inputs: `results_autodl_final/` and `results_autodl_ros2_twoprocess_final/`
- Output: `results_recent_deep_baselines_final/`

## Key Default-Threshold Findings

- GPU onnx_cuda/mobilenet_v2: best selected recall is `DCdetector` with recall 0.545, false alarm 0.380, audit 0.378; CIRCA-RT recall 0.485, audit 0.013, overhead 0.132 ms.
- GPU onnx_cuda/resnet18: best selected recall is `TranAD` with recall 0.454, false alarm 0.295, audit 0.296; CIRCA-RT recall 0.449, audit 0.015, overhead 0.139 ms.
- GPU onnx_cuda/squeezenet1_1: best selected recall is `CATCH` with recall 0.549, false alarm 0.367, audit 0.367; CIRCA-RT recall 0.472, audit 0.014, overhead 0.137 ms.
- GPU torch_cuda/mobilenet_v2: best selected recall is `CIRCA-RT` with recall 0.433, false alarm 0.059, audit 0.011; CIRCA-RT recall 0.433, audit 0.011, overhead 0.123 ms.
- GPU torch_cuda/resnet18: best selected recall is `CIRCA-RT` with recall 0.464, false alarm 0.063, audit 0.012; CIRCA-RT recall 0.464, audit 0.012, overhead 0.127 ms.
- GPU torch_cuda/squeezenet1_1: best selected recall is `CIRCA-RT` with recall 0.444, false alarm 0.061, audit 0.012; CIRCA-RT recall 0.444, audit 0.012, overhead 0.127 ms.

ROS2/DDS two-process selected methods:

- `ContextAwareConformal`: recall 0.589, false alarm 0.060, audit 0.123, p99 5.076 ms, overhead 0.541 ms.
- `TranAD`: recall 0.571, false alarm 0.045, audit 0.105, p99 4.127 ms, overhead 0.541 ms.
- `CIRCA-RT`: recall 0.429, false alarm 0.077, audit 0.014, p99 2.638 ms, overhead 0.138 ms.
- `CATCH`: recall 0.382, false alarm 0.045, audit 0.082, p99 4.154 ms, overhead 0.449 ms.
- `DCdetector`: recall 0.339, false alarm 0.039, audit 0.072, p99 4.045 ms, overhead 0.409 ms.

## Budget-Normalized Findings

At audit budget `0.10`:
- ROS2/DDS: ContextAwareConformal 0.394; TranAD 0.317; CATCH 0.257; DCdetector 0.242; LearnedFourierIndependence 0.208; ConditionalRFF-HSIC 0.199.
- ONNX MobileNetV2: ConditionalRFF-HSIC 0.400; CIRCA-RT 0.400; LearnedFourierIndependence 0.347; ContextAwareConformal 0.217; CATCH 0.133; DCdetector 0.076.

At audit budget `0.15`:
- ROS2/DDS: TranAD 0.526; ContextAwareConformal 0.519; CATCH 0.397; DCdetector 0.336; LearnedFourierIndependence 0.316; ConditionalRFF-HSIC 0.312.
- ONNX MobileNetV2: ConditionalRFF-HSIC 0.515; CIRCA-RT 0.515; LearnedFourierIndependence 0.456; ContextAwareConformal 0.274; CATCH 0.198; DCdetector 0.131.

## Interpretation

- The new deep baselines make the comparison much more current and harder.
- Deep methods often increase recall, but they usually spend much more audit budget or incur higher false alarms than CIRCA-RT under default thresholds.
- On ROS2/DDS, `TranAD` and `ContextAwareConformal` are strong recall competitors. CIRCA-RT remains the low-audit, low-overhead method.
- The defensible paper claim is budget-aware low-overhead detection, not universal recall dominance.

## Main Output Tables

- `results_recent_deep_baselines_final/tables/recent_deep_main_table.csv`
- `results_recent_deep_baselines_final/tables/recent_deep_budget_main_table.csv`
- `results_recent_deep_baselines_final/tables/combined_default_main_table.csv`
- `results_recent_deep_baselines_final/tables/combined_budget_main_table.csv`
- `results_recent_deep_baselines_final/tables/combined_budget_top8_table.csv`
- `results_recent_deep_baselines_final/tables/paper_core_default_table.csv`
- `results_recent_deep_baselines_final/tables/paper_core_budget_table.csv`
