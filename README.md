# maintain_pointcloud — LichtFeld Studio 位置アンカープラグイン

**事前に配置した初期点群の位置を極力維持しながら 3D Gaussian Splatting を学習する**ための
[LichtFeld Studio](https://github.com/MrNeRF/LichtFeld-Studio) プラグインです。
スプラットを初期点群位置に完全拘束（フリーズ）するのではなく、ソフトなアンカー（バネ／リーシュ）で
引き戻します。強度・不感帯半径・Huber 上限などを独自パラメータとして調整できます。

LI-GS / Structured-Li-GS / GTLR-GS / GeomGS / EnerGS 系の
「初期点群への位置正則化」研究を、LichtFeld Studio の Python プラグイン機構だけで
（C++ の改造・再ビルドなしに）実用化したものです。

## 何をするのか

各イテレーションの最適化ステップ後に、以下の補正をスプラット位置 μ に適用します。

```
d_i      = μ_i − a_i                      # アンカーからの変位
r_i      = ‖d_i‖
excess_i = max(0, r_i − free_radius)      # 不感帯（この中は自由）
pull_i   = s(t) · min(excess_i, huber_delta)
μ_i     ← μ_i − (d_i / r_i) · pull_i · gate_i
```

これは次のアンカー損失に対する 1 ステップの proximal（近接勾配）更新と等価です。

```
L_anchor = Σ_i gate_i · ρ_δ( max(0, ‖μ_i − a_i‖ − r_free) )
```

- `strength = 1`・`huber_delta = 0(無効)` → 各ステップで不感帯境界へ完全射影（ハードリーシュ）
- `strength` 小 → ソフトなバネ。光度勾配とアンカー引力の平衡点に落ち着く
- `strength = 0` / `Enabled` オフ → 通常学習と完全に同一（トポロジー追跡のみ継続）

### なぜ「損失に加算」ではなく「ステップ後補正」なのか

LichtFeld Studio は LibTorch 非依存の独自 CUDA テンソル＋解析的勾配で動いており、
Python フックは全て各イテレーションの train step **完了後**（SafeControl 窓）に実行されます。
勾配バッファは Python に公開されていないため、純プラグインで backward に損失を注入することは
できません。一方 `means_raw` は最適化器のパラメータメモリへの**書き込み可能な CUDA ビュー**
として公開されており（公式ドキュメント公認）、ステップ後の位置補正はアンカー損失の
明示的 SGD ステップと数学的に等価です。詳細は [docs/DESIGN.md](docs/DESIGN.md) を、
将来の C++ ネイティブ実装計画は [docs/cpp-native-loss.md](docs/cpp-native-loss.md) を参照。

### MCMC 密度制御への対応

LichtFeld の MCMC 戦略はスプラットを毎リファインで再配置（relocate）・追加（append）しますが、
行インデックスは並べ替えられません。本プラグインは:

- **追加された行** → 既定では非アンカー（`anchor_new_splats = True` にすると誕生位置で固定）
- **再配置された行** → 「生存行の閾値超ジャンプ」または「死→生の不透明度不連続」で検出し、
  移動先を新しいアンカーとして再登録（MCMC ノイズで飛ぶ瀕死行は誤検出しない）
- **瀕死の行** → `min_pull_opacity` 未満の行は引かない（不可視でノイズに支配されるため）
- **ソフト削除された行** → 影響なし
- **1 イテレーションで 5% 超が跳んだ場合** → モデル全体イベント（Bake Transform 等）と
  みなし、警告を出して全再キャプチャ

さらに `mode = "nn"` では、固定参照点群（初期スナップショットまたは外部 PLY）への
最近傍点をアンカーとして定期再割当てします（要 scipy、密度制御に構造的に強い方式）。

## インストール

LichtFeld Studio 内の Python コンソールで:

```python
import lichtfeld as lf
lf.plugins.install("toruhashimoto/lfs-maintain-pointcloud")
```

または手動で `~/.lichtfeld/plugins/maintain_pointcloud/` に本リポジトリの内容を配置します。
初回ロード時に numpy / scipy 入りの専用 venv が自動作成されます。

## 使い方（GUI）

メインパネルの **PointCloud Anchor** タブを開き、`Enabled` をオンにして学習を開始するだけです。
アンカーは学習開始時のスプラット位置（= 初期点群）から自動キャプチャされます。

## 使い方（ヘッドレス CLI）

```bash
LichtFeld-Studio.exe --headless --train -d <dataset> -o <out> --iter 30000 \
    --python-script <repo>/headless_anchor.py
```

環境変数で設定します（抜粋）:

| 環境変数 | 意味 | 既定値 |
|---|---|---|
| `LFS_MPC_ENABLED` | 有効化 | `1` |
| `LFS_MPC_STRENGTH` | 引き戻し強度 (0–1) | `0.01` |
| `LFS_MPC_FREE_RADIUS` | 不感帯半径（ワールド単位） | `0`（= 自動） |
| `LFS_MPC_FREE_RADIUS_SPACING` | 自動不感帯 = 点群の中央値点間隔 × この倍率 | `2.0` |
| `LFS_MPC_HUBER_DELTA` | 1 イテレーションの最大引き戻し距離 | `0`（無制限） |
| `LFS_MPC_MAX_DISTANCE` | これ以遠は切り離し（引力ゼロ） | `0`（∞） |
| `LFS_MPC_OPACITY_GATE` | 不透明度ゲート | `0` |
| `LFS_MPC_MIN_PULL_OPACITY` | この不透明度未満の行は引かない | `0.01` |
| `LFS_MPC_WARMUP` | ウォームアップ iter 数 | `300` |
| `LFS_MPC_STOP` | この iter 以降停止 | `0`（停止しない） |
| `LFS_MPC_ANCHOR_NEW` | 新規/再配置行を誕生位置でアンカー | `0` |
| `LFS_MPC_TELEPORT` | テレポート判定距離 | `0`（自動: 頑健シーン対角の 1%） |
| `LFS_MPC_MODE` | `index` / `nn` | `index` |
| `LFS_MPC_REF_PLY` | nn モードの参照 PLY | （初期スナップショット） |
| `LFS_MPC_STATS_OUT` | 終了時ドリフト統計 JSON の出力先 | （出力しない） |
| `LFS_MPC_CONFIG` | 全パラメータをまとめた JSON | — |

## 独自パラメータ一覧

| パラメータ | 意味 | 推奨 |
|---|---|---|
| `strength` | 不感帯を超えた分を毎 iter 引き戻す割合 | 0.01（既定）。0.1 以上は事実上の凍結 |
| `free_radius` | アンカー周りの自由半径。この内側は引力ゼロ | 0 のまま（点間隔から自動算出） |
| `free_radius_spacing` | 自動不感帯の倍率（× 中央値点間隔） | 2.0。狭めるなら 1.0、緩めるなら 3.0 |
| `huber_delta` | 引き戻し距離の上限（有界影響 / Huber 微分） | free_radius の 2–4 倍 |
| `max_distance` | 参照から離れすぎた行は正当な新規構造とみなし放置 | free_radius の ~10 倍 |
| `opacity_gate` | sigmoid(不透明度) を引力に乗算。低不透明度の再配置直後行を保護 | nn モードで推奨 |
| `min_pull_opacity` | 不可視行（この不透明度未満）を引かない。MCMC ノイズとの無駄な綱引きを回避 | 既定 0.01 のまま |
| `warmup_iters` | 光度整合が落ち着くまで強度をランプアップ | 300–1000 |
| `stop_iter` | 後半を自由にしたい場合のみ設定 | 0（維持目的なら切らない） |
| `anchor_new_splats` | MCMC が追加/再配置した行も誕生位置で固定するか | **False（既定）**。True は「初期点群維持」ではなく「全体凍結」になる |
| `mode` | `index`（誕生位置）/ `nn`（参照点群への最近傍） | まず `index` |

## 検証のしかた

`LFS_MPC_STATS_OUT` を指定すると、学習終了時に「初期行の初期位置からのドリフト統計」
（mean / median / p95 / max、シーン対角付き）が JSON で出力されます。
MCMC 再配置でスロットが再利用された行は自動除外され（`rows_measured` / `rows_excluded`）、
途中で全再キャプチャが起きた場合は `baseline_valid: false` が付きます。
アンカー無効・有効の 2 回を回して比較してください。MCMC はシード固定不可のため、
厳密な評価ではベースラインを 2 本回してノイズフロアを見ることを推奨します
（テストは `python mpc_tests/test_reference_math.py` で数式のリファレンス実装を検証できます）。

## 実測（実データ 30000 iter A/B、検証済み）

RAMESSES/Chest（44枚、初期点群 263,782 点、5,000,000 splat 到達、strategy=mrnf）。
**アンカー無効の独立2本**をノイズフロアとして取得したうえで、有効1本と比較。
2本のベースラインは全指標で 0.1% 以内に一致したため、下記の差はすべて実効果。

なお以下は **v0.1.0 の既定値（`strength=0.1` / `free_radius=0` / `anchor_new_splats=True`）**
での結果であり、v0.2.0 が既定値を変えた理由そのものです。

| 指標 | ベースライン | アンカー有効 | 評価 |
|---|---|---|---|
| 初期行ドリフト 中央値 | 0.0889 | 9.65e-05 | 921分の1（ただし下記の但し書き） |
| 初期行ドリフト **平均** | 10.38 | 16.34 | **+57% 悪化** |
| 初期スロット生存率 | 93.7% | 84.2% | 消失 2.5 倍 |
| スプラット最大軸スケール 中央値 | 0.0628 | 0.1375 | **+119%**（被写体領域で +97%） |
| 占有体積プロキシ | 1.45e6 | 2.62e6 | +81% |
| 再配置回数 | 2,045,304 | 3,030,627 | +48% |

**ドリフト中央値は二峰性を隠している**：初期スロットの 83% はほぼ完全に固定される
（drift < 1e-3、ベースラインは 0%）一方、15.8% が失われる（ベースライン 6.3%）。
p90 では逆転し、アンカー有効側が 31 倍悪い。平均変位で見ると全体としては悪化する。

さらに `anchor_new_splats=True` だったため、**最終モデルのうち初期点にアンカーされていたのは
219,198 / 5,000,000 = 4.4% だけ**。残り 95.6% は MCMC の誕生位置・再配置着地点という
幾何的意味を持たない座標に固定されていた。これが再配置 +48% とスケール膨張の主因。

計測していないもの：**測光品質は一切評価していない**（全 44 枚を学習に使用、hold-out なし）。
学習 loss（0.0361 対 0.0331/0.0406）からは何も結論できない（n=2 対 n=1 では検出力ゼロ）。

### v0.2.0 の既定値での挙動（Sample_COLMAP, 1000 iter）

不感帯を自動算出（点間隔 0.0431 × 2 = 0.0862）し、初期行だけをアンカーした場合:

| 初期行ドリフト | アンカー無効 | v0.1.0 既定（凍結） | **v0.2.0 既定** |
|---|---|---|---|
| 中央値 | 0.0297 | 1.7e-05 | **0.0294**（自由運動を維持）|
| p95 | 0.110 | 5.8e-05 | **0.0864**（−21%）|
| 最大 | 23.5※ | 2.62 | **0.602** |

※ ベースラインの最大値は再配置スロットを含む（当時は除外処理なし）。

中央値がベースラインとほぼ同一 = **大多数のスプラットは従来どおり自由に動く**。
一方で外れて飛んでいく尾だけが引き戻される。これが本来意図した
「極力維持するが完全拘束しない」挙動。

## 制限事項

- 補正は Adam のモーメントを経由しない明示的ステップです（正則化圧としては等価、
  厳密な「損失加算」は C++ 側対応が必要 — [docs/cpp-native-loss.md](docs/cpp-native-loss.md)）
- チェックポイント再開時、アンカーは再開時点の位置で再キャプチャされます
- `--freeze` された行にも補正が及びます（フリーズ行はそもそも動かないため実害なし）
- MCMC のノイズ注入と拮抗するため、`strength` が極端に小さいと平衡変位が残ります
- `free_radius = 0`（不感帯なし）で固定 `strength` を使うと、学習が進んで means の
  実効学習率が下がるにつれてリーシュが単調に締まり、**終盤は必ず完全凍結になります**。
  不感帯はこれを構造的に防ぐためのものなので、無効化しないことを推奨します
- テレポート閾値未満の変換（小さな Bake Transform 等）は検出できず、アンカーが
  変換前の姿勢へ引き戻します。編集後はパネルの「Re-capture anchors now」を推奨
- ごく稀に閾値未満ジャンプ + 不透明度クランプの再配置が検出を逃れ、古いアンカーが
  残ることがあります。`huber_delta > 0` を設定すればその影響は有界になります

## 研究背景

- **LI-GS** (arXiv:2409.12899) — LiDAR GMM への point-to-plane ソフトアンカー
- **Structured-Li-GS** (arXiv:2606.27509) — LiDAR ボクセルアンカー + 異方性 L1 オフセット罰則
- **GTLR-GS** (arXiv:2603.23192) — 曲率適応配置 + 信頼度重み付き深度正則化
- **GeomGS** (arXiv:2501.13417) — 最近傍距離の確率的アンカー（信頼度学習）
- **EnerGS** (arXiv:2604.26238) — Welsch カーネルによる有界影響アンカー
- **GS-SDF** (arXiv:2503.10170) — ニューラル SDF への恒等性フリーなアンカー
- **3DGS-MCMC** (arXiv:2404.09591) — 本プラグインが対応する密度制御レジーム

## ライセンス

GPL-3.0-or-later（LichtFeld Studio 本体に準拠）

---

# English summary

A LichtFeld Studio plugin that **keeps 3DGS splats close to the pre-placed initial
point cloud during training** — a soft position anchor (spring/leash), not a hard freeze.
Implemented purely with the official Python plugin API (no C++ rebuild): each iteration,
after the optimizer step, splat means are pulled back toward their anchors by
`strength · min(max(0, ‖μ−a‖ − free_radius), huber_delta)` along `−(μ−a)/‖μ−a‖`,
which is the proximal step of a dead-zone Huber anchor loss (LI-GS / GeomGS / EnerGS family).
MCMC relocation/growth is handled by index-stable birth anchoring with teleport detection,
or by nearest-neighbor re-targeting to a fixed reference cloud (`mode="nn"`).
Install: `lf.plugins.install("toruhashimoto/lfs-maintain-pointcloud")`. GPL-3.0-or-later.
