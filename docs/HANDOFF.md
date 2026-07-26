# 引き継ぎメモ（2026-07-26 時点）

別データセット・別マシンで検証を続けるときに、ここだけ読めば再開できるようにまとめたもの。

## プラグインの状態

v0.2.0 公開済み: https://github.com/toruhashimoto/lfs-maintain-pointcloud

```python
import lichtfeld as lf
lf.plugins.install("toruhashimoto/lfs-maintain-pointcloud")
```

**既定値のまま使ってよい。** v0.2.0 の既定（`strength=0.01` / `free_radius=0`=自動 /
`anchor_new_splats=False`）は実データ A/B で検証済み。v0.1.0 の既定は目的を反転させて
いたので、もし古い版が入っていたら必ず更新すること。

### 新しいデータセットで最初に確認する3点

1. **有効になっているか。** `enabled` の既定は False。GUI パネルの `Enabled` を入れるか、
   起動前に `LFS_MPC_ENABLED=1`。判定は `applied_iters`（`LFS_MPC_STATS_OUT` の JSON か
   ログ末尾）。**0 なら何も起きていない。** 一度これで2本とも無効のまま比較して丸一日
   分の実行を無駄にしている。
2. **自動不感帯が妥当か。** 起動ログに
   `anchors captured: N rows, scene diag ~X, nn spacing ~Y, free radius Z (auto)` が出る。
   Z が被写体スケールに対して極端でないか目視する。Z は点群の中央値最近傍間隔 × 2。
3. **テレポート閾値が妥当か。** 同じログ行の末尾。頑健シーン対角（p1–p99）の 1%。
   これが自由ドリフトの p95 より小さいと誤検出、被写体スケールより大きいと検出漏れ。

## 検証の作法（ここが本体）

MCMC/MRNF はシード固定不可。**必ずベースラインを2本回してノイズフロアを取ってから
効果量と比べる。** n=1 対 n=1 は何も言わない。実測ノイズフロア（RAMESSES/Chest, 44枚）:

- hold-out PSNR: **0.26 dB**（容量 5M でも 2M でも同じ）
- hold-out SSIM: 0.003
- 初期行ドリフト中央値: 0.1%

### 信用してはいけない指標（すべて実際に誤結論を出した）

| 指標 | なぜ駄目か |
|---|---|
| 学習 loss | hold-out がないと品質を反映しない。構造指標が 0.1% 一致していてもランごとに 22.7% 振れた |
| ドリフトの中央値だけ | 分布が二峰性になる。83% 完全固定・15.8% 消失で、平均はむしろ +57% 悪化していた例がある。必ず p50/p75/p90/p95 と `rows_excluded` を併せて見る |
| 初期点→最近傍スプラット距離（被覆） | アンカーが効いていると「スプラットが自分のアンカーの上に乗っている」ことを測るだけになる |
| 静的なフローター代理指標 | カメラ群から9倍遠い遠景を拾う（44視点中7-8視点にしか写らない）。フローターを主張するには novel view の描画が要る |

hold-out は `--eval --test-every 8` で作る。**付けないと全画像が学習に使われ、
PSNR は一切測れない。**

## 環境の罠（LichtFeld Studio 側）

- **headless + `--python-script` は動かない。** GUI モード + `--train` を使う。
  → Issue #1439
- **`plugin check` が日本語 Windows でクラッシュ。** `PYTHONUTF8=1` で回避。
  → Issue #1440
- **部分 config JSON は使えない。** `OptimizationParameters::from_json`
  (parameters.cpp:470-494) が13キーをガードなしで必須読み取りしており、
  `type must be number, but is number` という無関係なエラーで落ちる。
  `eval_steps` / `save_steps` / `scale_reg` だけ変えたい場合に詰む。
  回避: CLI フラグで済ませるか、完全な config を生成する。**未起票**
- プラグインの `settings.json` に `{"load_on_startup": true}` を書く際、**BOM を付けない**
  （PowerShell 5.1 の `Set-Content -Encoding utf8` は BOM 付きになり無視される）
- venv 手動作成: `build/bin/uv.exe venv .venv --python <vcpkg python.exe>` →
  `uv pip install numpy scipy` → `.venv\.deps_installed` を touch

## 未解決の問い

**イテレーション予算の最適化。** これが現時点で判明している唯一の有効な品質レバー。

RAMESSES/Chest では hold-out PSNR が 7,000 iter でピーク（17.38）、30,000 で 16.52 まで
落ちる（−0.86 dB）。原因は容量でも SH 次数でもない:

- `--max-cap` 500万 → 200万: 劣化は −0.99 → −0.86 とほぼ不変（ただし最終品質は +0.20 改善）
- `--sh-degree` 3 → 1 / 0: **悪化**（次数を下げるほどピークも最終も単調に低下）

容量も表現力も固定した状態で劣化する以上、後半イテレーションの学習ビューへの特化が
主因と考えられる。

スイープ用スクリプトを用意済み（未実行）:
`scratchpad/run_itersweep.ps1` — `--steps-scaler` で 3,000 / 5,000 / 7,500 / 10,000 /
15,000 を各2本。

**`--iter` ではなく `--steps-scaler` を使うこと。** `scale_steps()`
(parameters.cpp:44-63) が `iterations` / `start_refine` / `stop_refine` / `refine_every` /
`sh_degree_interval` / `grow_until_iter` / `eval_steps` / `save_steps` を一括でスケール
するため、全スケジュールが比例したまま短縮される。`--iter` だけ変えると密度制御や
SH ランプが絶対ステップ数のまま残り、比較にならない。また LR 減衰は
`gamma = (lr_end/lr_start)^(1/iterations)` なので、長い実行の途中経過を読むのと
最初から短く回すのは別物（後者は焼きなましきる）。

## この調査で分かったこと（結論だけ）

プラグインは目的を達成している。初期点群の位置を「大多数のスプラットは自由なまま、
外れた尾だけを不感帯の縁で捕まえる」形で維持し、代償は測光的に検出限界以下（<0.26 dB）。
容量 5M / 2M の2条件で再現済み。

ただし**フローター低減の道具ではない**。アンカーが支配するのはモデルの 12% 程度
（初期行のみ）で、フローターを構成する MCMC 生成分は管轄外。被写体近傍フローター代理
指標を閾値27通りで測っても効果/ノイズが 1.5 を超えた組合せはゼロだった。
フローターが目的なら検証済みの `scale_reg=0.02` が別途の正解（ただし MCMC 戦略での
結果であり、MRNF では未検証。かつ config バグで設定しにくい）。
