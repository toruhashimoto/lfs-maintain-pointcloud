# C++ ネイティブ実装ロードマップ — 真の position_anchor_reg 損失

本プラグインの proximal 補正で実用上は十分だが、Adam のモーメントを経由する
「真の損失項」として実装する場合の LichtFeld Studio 本体への変更計画。
上流の Issue #243 (Customizable Training) / PR #515 (scale/opacity reg の手書き勾配)
の路線に沿う。行番号は 2026-07 時点の master (v0.5.3-144-gdee66c79)。

## 損失定義

```
L_anchor = w · (1/N) Σ_i ρ_δ( max(0, ‖μ_i − a_i‖ − r_free) )
∂L/∂μ_i = (w/N) · clamp_δ'( … ) · (μ_i − a_i)/‖μ_i − a_i‖   （dead zone 内は 0）
```

## 変更点一覧

1. **パラメータ** — `src/core/include/core/parameters.hpp` (~:109) に
   `float position_anchor_reg = 0.0f;`（+ `position_anchor_free_radius`,
   `position_anchor_huber_delta`）を追加。`src/core/parameters.cpp` の
   to_json (~:115) / from_json contains() ガード (~:500) / validate() の
   nonnegative 配列 (~:267) に配線。既定 0.0 でオプトイン。

2. **カーネル** — `src/training/kernels/regularization.cu` に
   `launch_fused_position_anchor_regularization(means, anchors, mean_grads,
   loss_out, n, weight, r_free, delta, stream)` を追加。
   既存 `fused_scale_regularization_kernel` (:21-107) の
   atomicAdd + warp-reduce パターンを 3 要素/行に拡張。
   宣言: `src/training/include/lfs/kernels/regularization.cuh`。

3. **損失クラス** — `src/training/losses/regularization.hpp/.cpp` に
   `PositionAnchorRegularization`（`forward` / `forward_loss_only`）を
   `ScaleRegularization` (regularization.hpp:22-43) と同型で追加。

4. **Trainer** — `trainer.hpp/.cpp`:
   - メンバ `lfs::core::Tensor anchor_positions_;` を追加し、
     `Trainer::train()` 冒頭（`fitDepthAnchors` と同じ場所, trainer.cpp:5738 付近）で
     `strategy_->get_model().means().clone()` からキャプチャ
     （max_cap 容量で確保、MCMC 成長に追従）
   - 非融合 (gut) パス: `compute_scale_reg_loss` (trainer.cpp:1890) と同型の
     `compute_position_anchor_loss` を scale_reg ブロック (trainer.cpp:5167-5189)
     の隣に追加。勾配は `optimizer.get_grad(ParamType::Means)` へ

5. **fastgs 融合パス（既定・必須）** — means の Adam 更新はラスタライザ
   backward 内で融合されるため、`optimizer.get_grad(Means)` への書込みは無視される:
   - `FastGSFusedExtraGradients` (fast_rasterizer.hpp:136-146) に
     `float position_anchor_weight; const float* anchor_positions;` を追加
   - `FusedAdamSettings` (fused_adam_types.h:37 付近) / `FastGSFusedAdamState`
     (adam_optimizer.hpp:84-103) に同フィールド、
     `fast_rasterizer.cpp:687-699` でコピースルー
   - `kernels_backward.cuh`: 可視行の means 勾配確定部 (:288-291) と
     不可視行の fold (:68-71, means は現在 momentum decay のみ) に
     `(w/N)·dir·clamp(...)` を加算。sparsity ADMM の `sparsity_z/u`
     ポインタ渡しが device バッファ追加の前例

6. **MCMC アンカー整合** — `mcmc.cpp` `relocate_gs` の
   `launch_copy_gaussian_params` (:328-340) 後に
   `anchor[dead_indices[i]] = means[dead_indices[i]]`（誕生位置方式）、
   `add_new_gs` の `add_new_params_gather` (:517-522) 後に
   `anchor.append(means[new rows])`。`improved_gs_plus.cpp` / `mrnf.cpp` にも同フック

7. **公開** — `src/python/lfs/py_params.cpp:94` の float_prop パターンで
   Python/GUI プロパティ、`training.rml` + locale JSON に UI 行（任意）

## 工数見積り

コア（1–4, 6）: 数百行。fastgs 融合 (5) が最大の作業＋検証点。
検証は本プラグインの `LFS_MPC_STATS_OUT` ドリフト統計と同一プロトコルで、
proximal 版と native 版の等価性を同一データセットで比較すればよい。
