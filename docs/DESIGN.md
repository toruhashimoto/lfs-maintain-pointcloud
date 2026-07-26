# 設計ドキュメント — maintain_pointcloud

## 目的

事前に配置（撮影・クリーンアップ・整列済み）した初期点群の位置を、3DGS 学習を通して
「極力維持するが完全拘束はしない」。フローター抑制・幾何忠実性の観点で、初期点群が
信頼できる場合に学習後モデルの位置ドリフトを制御する。

## 前提: LichtFeld Studio 内部構造（調査結果）

実装方式はコードベースの実測調査に基づく。バージョン: master (2026-07 時点)。

1. **独自テンソル・autograd なし** — 学習は `lfs::core::Tensor`（LibTorch 非依存）で行われ、
   各損失は解析的勾配を `AdamOptimizer` の勾配バッファに直接書き込む
   (`src/training/optimizer/adam_optimizer.hpp` の `get_grad(ParamType)`)。
2. **既定パスは fused Adam** — 既定の fastgs パスではラスタライザ backward 内で
   Adam 更新までカーネル融合されており (`kernels_backward.cuh`)、勾配バッファへの
   外部書き込みは means には反映されない。
3. **Python フックは全て step 後** — `on_pre_optimizer_step` を含む全 per-iteration フックは
   `ControlBoundary::notify()` でキューされ、`trainer.cpp` の SafeControl 窓
   （train step 完了後）で一括実行される。名前に反して「ステップ前」には走らない。
4. **勾配バッファは Python 非公開** — したがって純プラグインでの損失注入は不可能。
5. **`means_raw` は書き込み可能なライブ CUDA ビュー** — `SplatData` は scene 所有で
   strategy と共有されており、Python の in-place 演算はオプティマイザの実パラメータ
   メモリを直接書き換える（公式ドキュメントが GPU 直書きを明示的に認めている）。

→ 純 Python で実現可能な唯一の正則化形態は **step 後の位置補正（proximal step）**。

## 数式

アンカー損失（dead-zone Huber、LI-GS / GeomGS / EnerGS 系）:

```
L_anchor = Σ_i gate_i · ρ_δ( max(0, ‖μ_i − a_i‖ − r_free) )
```

- `ρ_δ` : Huber カーネル（二次領域→線形領域、δ で影響を有界化）
- `r_free` : 不感帯半径。初期点群の点間隔程度を許容し、表面への微調整を妨げない
- `gate_i` : オプションで `sigmoid(opacity_raw_i)`（MCMC 再配置直後の低不透明度行を保護）

毎イテレーション、この損失の劣勾配方向へ明示的に 1 ステップ進める:

```
d_i      = μ_i − a_i
r_i      = ‖d_i‖
excess_i = max(0, r_i − r_free)
pull_i   = s(t) · min(excess_i, δ) · gate_i · [r_i ≤ d_max]
μ_i     ← μ_i − (d_i / r_i) · pull_i
```

### 性質

- `s = 1, δ = ∞` : 不感帯球面への射影（ハードリーシュ）。「完全拘束」の上限に相当
- `s < 1` : 幾何級数的に超過変位が減衰。光度勾配 g（毎 iter ~lr 相当の押し出し）との
  平衡点は `excess* ≈ (lr 由来の押し出し量) / s`。s を上げるほど平衡ドリフトが縮む
- Adam のモーメントを経由しないため、補正自体が振動やモーメント汚染を起こさない
- MCMC のノイズ注入（毎 iter、means に付加）とは拮抗し、平衡に `σ_noise / s` 程度の
  ゆらぎが残る（これが「完全拘束しない」性質そのもの）

### 損失加算との等価性について

真の損失加算（backward への注入）は勾配が Adam の適応的スケーリングを受けるのに対し、
本方式は生ステップを直接加える。正則化「圧」としての作用は同等だが、係数の意味が
異なる（本方式の `strength` は変位比率で直接解釈できるため、むしろ調整しやすい）。
厳密な損失注入が必要な場合は C++ 側の対応が必要（[cpp-native-loss.md](cpp-native-loss.md)）。

## MCMC トポロジー対応

LichtFeld の MCMC (`src/training/strategies/mcmc.cpp`) の実測挙動:

| 操作 | 挙動 | 本プラグインの対応 |
|---|---|---|
| relocate | 死行に生行のパラメータを **in-place コピー**（行番号不変） | ジャンプ検出→移動先で再アンカー |
| add | 末尾に最大 5%/iter **append** | 誕生位置でアンカー（またはマスク 0） |
| remove | ソフト削除（行は残る） | 何もしない（無害） |
| noise | 毎 iter means に摂動 | strength との平衡で吸収 |

行の並べ替え・圧縮は発生しないため、`[N,3]` のアンカーバッファ＋
「行数増加時 append」「テレポート時再アンカー」の 2 フックで恒等性が保てる。

### 再配置（テレポート）検出の詳細

素朴な「変位 > 閾値」だけでは 2 つの失敗モードがある:

1. **偽陽性**: MCMC は毎 iter means にノイズを注入し、その大きさは
   `lr · 5e5 · op_sigmoid(不透明度) · Σ` に比例する。瀕死（半透明）でスケールの大きい
   スプラットは閾値を超えるキックを受け得る → 生存ゲート
   `sigmoid(opacity) ≥ 0.02` を満たす行だけをジャンプ判定の対象にする。
   瀕死行はそもそも `min_pull_opacity` で引かないため、アンカーの鮮度は問題にならない。
2. **偽陰性**: 再配置先が偶然 1% 対角以内に落ちるケース。relocate は生存行の不透明度を
   コピーするため「死 (sigmoid<0.02) → 生 (sigmoid>0.1) の 1 iter 不連続」が同時に発生
   する（Adam の不透明度ステップでは 1 iter で作れないジャンプ）→ これを第二の検出器
   にする。両方をすり抜ける組合せ（閾値未満 + min_opacity クランプ）は稀で、
   `huber_delta > 0` により影響が有界になる（README 制限事項に記載）。

さらに、1 iter で全行の 5% 超が「テレポート」した場合はモデル全体イベント
（Bake Transform / その undo / 外部からの means 書込み）とみなし、行別の再アンカー
ではなく警告付き全再キャプチャを行う（初期基準は `baseline_valid: false` で無効化を明示）。

### nn モード

`mode="nn"` は恒等性追跡を放棄し、固定参照点群（初期スナップショットまたは外部 PLY）への
**最近傍点**を毎 `nn_refresh` iter で再割当てする（片側 chamfer 型）。
文献では静的 GMM (LI-GS)・SDF (GS-SDF)・NN エネルギー (EnerGS, GeomGS) に相当する
恒等性フリー方式で、密度制御に構造的に頑健。コストは CPU KD-tree クエリ
（100 万行で ~1 秒 / refresh、既定 100 iter ごと）。

## パラメータ設計指針

- `strength` : まず 0.1。維持を強めたければ 0.3、ハード寄りなら 0.5–1.0
- `free_radius` : 初期点群の中央値点間隔の 1–2 倍。0 でも動作（純粋なバネ）
- `huber_delta` : `2–4 × free_radius`。遠方に飛んだ行を一定速度でしか引き戻さない
  （有界影響）。0 = 無効
- `max_distance` : `~10 × free_radius`。参照に存在しない正当な新規構造
  （空・未スキャン領域）を引っ張らないための切り離し半径。0 = ∞
- `warmup_iters` : 300–1000。光度整合が落ち着く前に引くと初期整列を妨げる
- スケジュール: 文献のアンカー重みはすべて定数（LI-GS λ=1.0, GeomGS 0.1, EnerGS w=1.0）。
  「維持」が目的なら decay しない。`stop_iter` は品質優先で後半を自由にしたい場合のみ

## 検証プロトコル

MCMC は wall-clock シードのため再現不可。**ベースライン 2 本でノイズフロアを測ってから**
効果量と比較すること（`3dgs-floater-cleaner` の FINDINGS.md の測定規律に準拠）。

1. `LFS_MPC_ENABLED=0` で 2 回学習 → `drift.json` の mean/median ドリフトのばらつき = ノイズフロア
2. `LFS_MPC_ENABLED=1`（既定 strength=0.1）で学習 → ドリフトがフロアを超えて減っていること
3. 品質（PSNR/SSIM）を eval で比較 → 劣化がノイズフロア内であること

`mpc_tests/test_reference_math.py` が補正数式の numpy リファレンスを固定する
（不感帯・Huber 上限・切り離し・ゲート・射影収束・再配置検出の陽性/偽陽性/偽陰性）。

ドリフト統計 (`init_drift_stats`) は MCMC 再配置でスロットが再利用された行を
`_orig` マスクで除外して計測する（除外数は `rows_excluded` として報告）。
除外しないと max/p95 が「再配置距離」を「ドリフト」として誤報告する。

## 既知の制限

- チェックポイント再開時はアンカーを再開位置で再取得（v0.1）。将来: サイドカー保存
- `--freeze` 行にも補正が届くが、フリーズ行は光度側でも動かないため実害なし
- パネル操作と学習スレッドの共有状態はスカラー属性のみ（GIL でアトミック）
- stats 計算（既定 10 iter ごと）は GPU→CPU 同期を伴う。オーバーヘッドは数 ms 未満
