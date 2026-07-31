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

## 推奨設定（結論から）

3 データセット・のべ 60 本以上の A/B で得た結論。根拠は
[docs/RESULTS.md](docs/RESULTS.md)。

### このプラグインの位置づけ（2026-07-30 確定）

対象の 3DGS モデルは**フォトグラメトリによる実測 3D モデルの補完**であり、将来
**計測・重ね合わせに使われる可能性が高い**。したがって**位置忠実性が第一優先**で、
それがこのプラグイン（位置アンカー）の担当分野です。実測では、測量点からの
ドリフト裾（p95）を −56〜−58%（効果/ノイズ 112〜345、2 データセット）、
逸脱行を 323,295 → 2〜6 行まで抑え、
測光コストは全条件で検出されていません。

- **計測・重ね合わせが要件の納品**では下記の設定でアンカーを有効にする
- **既定はオフを維持**する（汎用性のため。見た目だけの用途に強制しない）
- **手作業クリーンアップの削減はアンカーの担当ではない** — 学習前の入力クロップ
  （下記）が 6 倍効く。両者は直交し、クロップは「無いはずの場所に作らない」、
  アンカーは「測った場所から動かさない」を保証する

**幾何忠実性を上げたい（このプラグインの目的）**

```bash
LFS_MPC_ENABLED=1
LFS_MPC_CALIBRATE=1      # 不感帯をドリフト分布から自動決定
LFS_MPC_CALIB_Q=60       # 2026-07-31 から既定（それまで 70）
```

`LFS_MPC_FREE_RADIUS` は**指定しないでください。**外部仕様としての公差が無い限り、
データから決まる値のほうが根拠のない固定値より確実に妥当です。実測でも、校正は
手動掃引が見つけた最良値を 2% 以内で再現し、上回りました（p95 −44.9%、
効果/ノイズ 72.3）。**測光コストは検出されません。**

`--max-cap` は**初期点群の点数の 2 倍以上**にしてください。点数ちょうどに設定すると
密度制御が停止します（下記「初期点群が `--max-cap` を超える場合」）。

**手作業クリーンアップを減らしたい**

**このプラグインの学習時機能ではなく、学習前に `points3D.txt` を納品領域の箱で
クロップしてください。** プラグインのどの機能より効きます — **2 データセットで再現済み**:
箱外スプラット −56.0%（C、効果/ノイズ 15.5）/ **−59.1%（D、51.1）**、
納品領域の PSNR は −0.24 dB 〜 **+0.11 dB**（コストが出ないデータセットもある）。同梱の
[`crop_input.py`](crop_input.py) がこれを行います。

```bash
.venv\Scripts\python.exe crop_input.py --data <colmap> --box "-1,-1.5,-1:3,0.5,1" --dry-run
```

`--dry-run` は 1 点も書かずに「その箱が何点落とすか」だけを表示します。**箱を決める
のはこれで、決めてから `--dry-run` を外してください**（4.1M 点で 14 秒、
`<colmap>_cropped` ができます）。`cameras.txt` / `images.txt` はハードリンク、
`images/` はジャンクションで共有するので、追加ディスクは `points3D.txt` の分だけ
（実測 415 MB）。使った箱は出力先の `crop_input.json` に残るので、複数の箱を
試したあとでもどの学習がどの箱だったか判別できます。

理由は、撮影された点群自体が汚れているからです。実測では 4,302,501 点のうち
29,175 点（0.68%）が納品箱の外にあり、学習はそこにスプラットを育てていました。
**アンカーはむしろその点の上に行を留める**ので、`index` モードの効果は −0.3%
（効果/ノイズ 0.1）しかありません。余白は既定 5%（`--pad`）です
（納品後の箱は事前に知り得ないため）。

### 公称箱の座標（2 データセットの納品モデルで確定）

COLMAP フレームでの公称箱は **`-1,-1.5,-1:3,0.5,1`**（中心 `[1,-0.5,0]`・
サイズ `[4,2,2]`）です。これを `--box` に渡してください。

以前「中心 `[-1,0.5,0]`」と記録されていましたが、**x と y の符号が逆**でした。
そのまま当てると データセット D の点群の **27.1%** が落ち、被写体が箱に
入りません。符号を直すと 7.7% になります。

符号反転が正しいことは 2 データセットの納品モデルの実測箱で確認できます。上の公称箱は
どちらの実測箱も内包します。

| | C | D |
|---|---|---|
| 実測箱 lo | `-0.8777, -1.2579, -1.0000` | `-0.9992, -1.4594, -1.0000` |
| 実測箱 hi | `3.1204, 0.3682, 1.0000` | `2.1650, 0.3634, 0.9999` |
| 公称箱との lo 差 | `0.122, 0.242, 0.000` | `0.001, 0.041, 0.000` |
| 公称箱との hi 差 | `-0.120, 0.132, 0.000` | `0.835, 0.137, 0.000` |

**z の差は 4 面中 3 面で 0.000、残る 1 面も 0.0001** — 納品時の z 制限は
実質ちょうど ±1 です。
x の下限も D では 0.001 差で一致します。つまり公称箱は実在する規約であって
おおよその目安ではありません。ただし **C は hi 側で公称箱を 0.120 超えている**
ので、余白 5%（既定）はそのまま必要です。

**評価に使う箱と、クロップに使う箱を混同しないこと。** クロップには公称箱を
使ってください。そのデータセットの納品モデルから測った箱でクロップすると、**正解ラベルを
入力に混ぜたことになり効果が過大に出ます**（本番では納品箱はまだ存在しません）。

`LFS_MPC_MODE=nn` + `LFS_MPC_ANCHOR_NEW=1` は**未クロップ入力**では箱外を
−8.6%（C）/ −2.4%（D）動かしますが、**クロップ済み入力に重ねる意味は
確立できていません** — 箱外の追加効果は効果/ノイズ 1.76 / 1.3 で結論未満、
箱内 PSNR は −1.04 dB / −0.29 dB と 2 データセットで方向一致のコストが出ます。
足すのは箱外ではなく**撮影面への忠実性**（箱内 p99 距離が 3.6 → 1.0 点間隔に
吸着、2 データセットで再現）が目的の場合だけにしてください。実行時間は +10% ほど。

**やってはいけないこと**

- **`LFS_MPC_CROP_BOX`（学習中のクロップ箱）を使わない。** 箱外は 26% 減りますが
  **納品する領域そのものが 3.71 dB 悪化します。**実装はしてありますが、これは
  「試して駄目だった」記録として残しているだけです
- **`anchor_new_splats=1` を `index` モードで使わない。** 誕生位置に凍結され、
  再配置 +48%・スケール +119% になります（v0.1.0 の失敗）

### GUI から手動でクロップする（Input Crop パネル）

CLI を使わずに、RealityScan の再構成領域のようにビューア上で箱を編集して
同じクロップを実行できます。パネルは `Input Crop`（メインパネルタブ）。

1. データセットを開くと COLMAP パスは自動で入ります（手入力も可）
2. **Seed default region**（既定 ROI を配置）か **Fit to cloud**
   （点群の p0.5〜p99.5 にフィット）で箱をシーンに置く
3. ビューア標準の**クロップ箱ツール（ギズモ）**で箱を調整する —
   このパネル自身はギズモを持たず、エンジンの箱ノードをそのまま使います
4. **Count (sampled)** で「その箱が何 % 落とすか」を約 1 秒で確認
   （1/8 サンプリング。書き出しは常に全点の厳密パス）
5. **Export cropped dataset** で `<data>_cropped` を書き出し
   （ハードリンク／ジャンクション共有・`crop_input.json` 付き、
   CLI と同一の実装を呼びます）

制限（v1・意図的）: 書き出されるのは箱の**軸平行 min/max** です。箱ノードを
回転させても書き出しには反映されません（パネルに明記）。**inverse**（外側を
残す）設定の箱は書き出しを拒否します — 被写体を消して背景を残す事故防止です。

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
| `LFS_MPC_CALIBRATE` | 不感帯を対照行のドリフト分位点から自動決定 | `0` |
| `LFS_MPC_CONTROL_FRACTION` | 対照行（引かない行）の割合 | `0.02` |
| `LFS_MPC_CALIB_Q` | 不感帯に使う分位点 | `60`（2026-07-31 まで `70`） |
| `LFS_MPC_CALIB_EVERY` | 校正間隔（iter） | `500` |
| `LFS_MPC_CALIB_START` | この iter 以降で校正開始 | `1000` |
| `LFS_MPC_CALIB_SEED` | 対照行選択の乱数種（再現性） | `0` |
| `LFS_MPC_CALIB_MIN_SPACING` | 不感帯の下限（× 点間隔） | `0.25` |
| `LFS_MPC_REANCHOR_SPLIT` | 分割行を分割後の位置で再アンカー | `0`（**使わない**） |
| `LFS_MPC_CROP_BOX` | 学習中の ROI `x0,y0,z0:x1,y1,z1` | （空 = 無効。**使わない**） |
| `LFS_MPC_CROP_BOX_PAD` | ROI の各軸に加える余裕（割合） | `0.05` |
| `LFS_MPC_STATS_OUT` | ドリフト統計 JSON の出力先 | （出力しない） |
| `LFS_MPC_SNAPSHOT_EVERY` | 統計 JSON を書き直す間隔（iter）。0 で無効 | `1000` |
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
| `stats_snapshot_every` | 統計 JSON を書き直す間隔。最後に書けたものが最終結果になる | 1000。厳密に終端付近が欲しければ 100–200 |
| `calibrate` | 不感帯を対照行のドリフト分位点から毎 `calibrate_every` iter 決め直す | **True 推奨。**手動掃引の最良値を再現し上回る |
| `calibrate_quantile` | 不感帯に使う自由ドリフト分位点 | 70（既定）。下げると位置は良くなるがスケールが膨らむ |
| `control_fraction` | 対照行（自由ドリフトを測るため引かない行）の割合 | 0.02 のまま。上げると置いた点群が保持されなくなる |
| `calibrate_start` | 校正を開始する iter | 1000 のまま。5000 は分布の胴体が悪化する |
| `reanchor_on_split` | MRNF の長軸分割行を分割後位置で再アンカー | **False。**スケール膨張は減るが逸脱行数が桁で戻る |
| `crop_box` | 学習中の ROI | **空のまま。**納品領域の PSNR が 3.71 dB 落ちる |

## 検証のしかた

`LFS_MPC_STATS_OUT` を指定すると、「初期行の初期位置からのドリフト統計」
（mean / median / p50 / p75 / p90 / p95 / max、シーン対角付き）が JSON で
出力されます。分布は二峰性になるので、**中央値だけを見ないでください**
（`median` は `p50` の別名です）。不感帯の内側にいる大多数の行は自由に動くため
中央値はアンカー無効時とほぼ一致し、リーシュの作用は上位の裾にしか現れません。
MCMC 再配置でスロットが再利用された行は自動除外され（`rows_measured` / `rows_excluded`）、
途中で全再キャプチャが起きた場合は `baseline_valid: false` が付きます。
アンカー無効・有効の 2 回を回して比較してください。MCMC はシード固定不可のため、
厳密な評価ではベースラインを 2 本回してノイズフロアを見ることを推奨します
（テストは `python -m pytest mpc_tests/` で数式のリファレンス実装とスケジューリング
ロジックを検証できます）。

**`max` を効果量として読まないこと。** 単一行の値であり、平均と標準偏差で扱える
統計量ではない。実測ベースライン 6 本は 10.879 / 11.023 / 11.408 / 41.405 /
79.971 / 93.861 と二峰性で 9 倍振れ、アンカー側が全本ベースラインを下回っていても
効果/ノイズは 1.0 のままだった。**本数を増やしても収束しない。** 代わりに
`p99` / `p999` と、点間隔の 1/2/4 倍を超えた行数（`escaped_1x` / `escaped_2x` /
`escaped_4x`）を見る。どうしても `max` で判定するなら
`scripts/rank_test.py` の厳密な並べ替え検定を使う（同じデータで p = 0.0048）。

逸脱行数の閾値が不感帯ではなく**点群の中央値点間隔**なのは意図的である。不感帯は
腕ごとに違う（校正するのだから当然）ので、それを閾値にすると腕ごとに違う基準で
数えることになり比較できない。点間隔はデータの性質なので全実行で同一になる。

この JSON は学習終了時だけでなく `stats_snapshot_every` iter ごとに上書きされ、
**最後に書けたものが最終結果**になります。ファイル自身が `iter`（どの iteration の
状態か）と `final`（学習終了フックから書かれたか）を持つので、終端との差は判別できます。
定期書き出しにしているのは利便性のためではありません。LichtFeld Studio v0.5.1 の
ヘッドレス実行は training_end フックを登録はするものの一度も発火させず、埋め込み
Python は `atexit` も実行しないため、終了時にだけ書く実装では**ファイルが 1 つも
生成されません**。

まず最初に見るべきは `applied_iters` です。**0 なら補正は一度も適用されていません**
（`enabled` の既定は False）。

次に見るべきは **`init_drift.p95 ÷ config.free_radius_effective`** です。
アンカー無効の 1 本でこれを計算してください。**1 を下回っていたら、初期行の
95% 以上が不感帯の内側にいるということで、リーシュには引く対象がありません。**
そのまま A/B を回しても全指標がノイズ以下になります（データセット B で実際に
そうなりました。比 0.81 で効果ゼロ、`free_radius_spacing` を 1.0 にして
比 1.61 にすると p95 −38%）。この場合は `free_radius_spacing` を下げて
比を 1 より大きくしてから比較してください。詳細は [docs/RESULTS.md](docs/RESULTS.md) の「別データセットでの検証」以下。

### 初期点群が `--max-cap` を超える場合（重要）

**このプラグインの本来の用途で最も踏みやすい罠です。** LichtFeld Studio の
`max_cap` 既定は 5,000,000 で、初期点群がこれを超えると起動時に警告一行を出して
**ランダムに間引きます**。

```
[warning] training_setup.cpp:356 Max cap (5000000) is less than initial splat
          count (7886332), randomly selecting 5000000 splats
```

このとき二つのことが同時に起きます。

1. **置いた点の一部がアンカー対象から消えます。** 上の実測例では 36.6%
   （2,886,332 点）が学習開始前に捨てられました。維持したい点群そのものが
   減るので、プラグインの目的が根本から損なわれます。
2. **密度制御が事実上停止します。** 初期点数が上限に張り付くため MRNF が
   `utilization: 100.0%` で開始し、再配置・追加に使える空きスロットがありません。
   実測（150 iter 時点の teleports）では `--max-cap 5000000` で 27 回、
   `--max-cap 8000000` で 27,744 回と 1000 倍の差が出ました。上限に張り付いた
   条件は、密度制御が動く通常の学習とは別物です。

**初期点群の点数の 2 倍以上の `--max-cap` を明示してください。** 点数は COLMAP の
`points3D.txt` 先頭 3 行目（`# Number of points:`）で確認できます。

VRAM は engine のバージョンで大きく違います。v0.5.1 では 7,886,332 点・SH 次数 3 で
17.1 GB でしたが、**v0.5.3 では同条件 7.3 GB**（quantized Adam state のため）。
16M cap でも 12.5 GB です。**古い前提で `--max-cap` を絞らないこと。**実測では
8M cap（初期点群の 1.01 倍）から 16M（2.03 倍）に上げるだけで hold-out PSNR が
@7,000 で 16.76 → 22.20、@30,000 で 23.67 → 24.91 と改善しました。

### ヘッドレス実行時の注意

- `--headless` と `--python-script` は併用できます。ただし **`--python-script` を
  渡さないと Python ランタイム自体が起動せず、プラグインも読み込まれません。**
- `--python-script` で `headless_anchor.py` を使う場合、LichtFeld Studio が
  有効化するのは共有 venv（`~/.lichtfeld/venv`）であって、プラグイン自身の
  `.venv` ではありません。`headless_anchor.py` は隣接する `.venv` を自動で
  `sys.path` に追加しますが、numpy / scipy がどこにも無い場合は起動時に
  エラーログを出します。**この警告を無視しないでください**: 依存が欠けると
  自動不感帯が 0 に落ち、固定 strength と組み合わさって終盤の完全凍結に至ります。
- `--steps-scaler` を極端に小さくすると（実測: 0.002）密度制御のスケジュール値が
  0 に切り捨てられ、ゼロ除算で異常終了します。短縮する場合も 0.003 以上を推奨。
- **上記は v0.5.1 での実測です。v0.5.3-155 では `training_end` フックは発火し、
  `plugin check` も正常終了します**（Issue #1439 / #1440 は解消）。ただし定期
  スナップショットはフックに依存しない保険として残してあります。
- テストは **`.venv` の python** で走らせてください（システム python には numpy が
  入っていません）。`python -m pytest mpc_tests/` で 119 本。
## 実測結果の要約

詳細は **[docs/RESULTS.md](docs/RESULTS.md)**（3 データセット・のべ 60 本以上の A/B）。
ここでは結論だけ。

**効いたもの**

| | 効果 | 効果/ノイズ |
|---|---|---|
| 自己校正不感帯（`calibrate=1`, q70） — ドリフト p95 | −44.9% | 72.3 |
| 同 — 逸脱行数（可視行、点間隔の 2 倍超） | 204,743 行 → **0** | — |
| q を 60 / 50 に下げる — ドリフト p95 | −52.4% / −57.9% | 320 / 112 |
| **入力点群を箱でクロップ** — 箱外スプラット（C） | **−56.0%** | 15.5 |
| 同 — **2 つ目の被写体（D）で再現** | **−59.1%** | **51.1** |
| `nn` + `anchor_new` — 箱外スプラット（C / D） | −8.6% / −2.4% | 3.2 / 4.3 |

入力クロップの箱内 PSNR コストは **−0.24 dB（C）〜 +0.11 dB（D）**。
2 データセットとも小さく、D ではむしろ改善だった。

**測光コストは一貫して検出されない。** hold-out PSNR の効果/ノイズは 0.1〜0.4。
幾何を p95 で 45% 締めても品質は落ちない。

**効かなかった／やってはいけないもの**

| | 結果 |
|---|---|
| `index` モードのアンカー（箱外スプラットに対して） | −0.3%（効果/ノイズ 0.1）。撮影点自体が箱外にあり、アンカーはそこに留める |
| 学習中クロップ箱（`LFS_MPC_CROP_BOX`） | 箱外 −26% だが**納品領域が −3.71 dB 悪化**。実装したが使わない |
| **クロップ済み入力への** `nn` + `anchor_new` | 箱内 PSNR −1.04 dB / −0.29 dB（2 データセットで方向一致）。箱外の追加効果は効果/ノイズ 1.76 / 1.3 で未確立 |
| `reanchor_on_split=1` | スケール膨張は 2/3 消えるが逸脱行数が 25 → 106,080 に戻る |
| `control_fraction` を 0.02 → 0.10 | バイアスは減るが置いた点群の 10% が保持されなくなる |

**測定について分かったこと（他のプロジェクトでも効く）**

- **`max` を効果量として読まない。** 同じ現象を `p999` で測ると効果/ノイズ 199.7、
  `escaped_2x` で 100.1、`max` では 1.0。問題はデータではなく統計量の選び方だった
- **領域を絞って学習するなら全画面 PSNR は使えない。** 箱の内側だけを測って初めて
  クロップ箱の害が見えた（`scripts/inbox_psnr.py`）
- **不可視スプラットは統計を汚す。** 最も漂った 32 行の 31〜32 行が
  `min_pull_opacity` 未満で、設計上引かない行だった

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
- **`min_pull_opacity` 未満の不可視行は一切引かれません**（設計どおり）。極端に
  漂う行はほぼ全てこれで、`max` や `escaped_*` を全行で見ると汚染されます。
  可視行だけの系列 `visible_drift` を併せて見てください
- **手作業クリーンアップの主要因（納品箱の外に育つスプラット）は解決しません。**
  撮影点自体が箱の外にあるとアンカーはそこに留めるためです。学習前に
  `points3D.txt` をクロップしてください（上記「推奨設定」）
- 統計を全部有効にすると 1 スナップショットあたりの GPU→CPU 転送が増え、
  8M 行・30,000 iter で実行時間が 26.7 分 → 32.7 分になります。常用するなら
  `stats_snapshot_every` を上げてください

## 引き継ぎ

別データセット・別マシンで検証を再開するときは [docs/HANDOFF.md](docs/HANDOFF.md) を先に読む
（検証の作法、環境の罠、未解決の問い）。

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
Since v0.3.0 the dead zone can calibrate itself: a small fraction of the initial
rows is held out from the pull entirely, their drift quantile is read as a live
estimate of the *free* drift distribution, and the dead zone is re-derived from it
every few hundred iterations. Measured on a 1,749-image crash capture, this
reproduced and beat a value that had taken a manual sweep to find (drift p95
−44.9%, effect/noise 72.3) at no detectable photometric cost, and it tracked a
21% wider zone when raising `--max-cap` widened the free drift by 26% -- something
a spacing-derived constant cannot do.

Two things measured and NOT recommended, kept in the code as records: a
training-time crop box (cuts out-of-box splats 26% but costs 3.71 dB *inside the
delivered region*), and re-anchoring MRNF's long-axis splits (removes two thirds
of the scale inflation but returns the drift tail to half the unanchored
baseline). And one finding that is not about this plugin at all: if the goal is
less manual cleanup, crop the input `points3D.txt` to the region of interest
before training -- a 7-second text filter beat every plugin feature by 6x.

Install: `lf.plugins.install("toruhashimoto/lfs-maintain-pointcloud")`. GPL-3.0-or-later.
