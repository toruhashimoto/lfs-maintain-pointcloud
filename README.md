# maintain_pointcloud — LichtFeld Studio 位置アンカープラグイン

**事前に配置した高精度な初期点群を「正」として扱いながら 3D Gaussian Splatting を
学習する**ための [LichtFeld Studio](https://github.com/MrNeRF/LichtFeld-Studio)
プラグインです。

フォトグラメトリや LiDAR の実測点群から 3DGS を作るとき、通常の学習は点群の位置を
何も保証しません。スプラットは測光的に都合のよい場所へ漂い、測った覚えのない場所にも
育ちます。本プラグインはこの 2 つの問題に、それぞれ専用の道具で答えます。

| 道具 | 何を保証するか | 実測効果 |
|---|---|---|
| **位置アンカー**（学習中） | 測った場所から動かさない | 初期行ドリフト p95 **−56〜−58%**（効果/ノイズ 112〜345、2 データセット）、逸脱行 323,295 → 2〜6。測光コストは全条件で検出されず |
| **入力クロップ**（学習前） | 無いはずの場所に作らせない | 対象領域外のスプラット **−56% / −59%**（2 データセットで再現）。学習時のどのレバーよりおよそ 6 倍効く |

二つは直交していて、併用できます。位置アンカーはソフトなバネ（リーシュ）で
スプラットを初期点群位置へ引き戻すもので、完全拘束（フリーズ）はしません。
入力クロップは学習前に `points3D.txt` を対象領域の箱でフィルタし、クロップ済みの
COLMAP データセットを作ります。

LI-GS / Structured-Li-GS / GTLR-GS / GeomGS / EnerGS 系の「初期点群への位置正則化」
研究を、LichtFeld Studio の Python プラグイン機構だけで（C++ の改造・再ビルドなしに）
実用化しました。本 README の数値はすべて実データの A/B 測定です — 4 データセット・
のべ 60 本以上、各アーム 2 本以上。一次資料は [docs/RESULTS.md](docs/RESULTS.md)。

## インストール

LichtFeld Studio 内の Python コンソールで:

```python
import lichtfeld as lf
lf.plugins.install("toruhashimoto/lfs-maintain-pointcloud")
```

または本リポジトリの内容を `~/.lichtfeld/plugins/maintain_pointcloud/` に配置します。
初回ロード時に numpy / scipy 入りの専用 venv が自動作成されます。

動作確認済みのエンジンは v0.5.1 プレビルドと v0.5.3 系ソースビルドです。
テストは `.venv\Scripts\python.exe -m pytest mpc_tests/` で 130 本
（エンジンの埋め込みインタプリタなしで走ります）。

## クイックスタート

### 位置アンカー（GUI）

メインパネルの **PointCloud Anchor** タブで `Enabled` と `Calibrate` をオンにして
学習を開始するだけです。アンカーは学習開始時のスプラット位置（= 初期点群）から
自動キャプチャされ、不感帯はドリフト分布の実測から自動決定されます。

### 位置アンカー（ヘッドレス CLI）

プラグインとしてインストールしなくても、`--python-script` で単体実行できます:

```bash
LichtFeld-Studio.exe --headless --train -d <dataset> -o <out> --iter 30000 \
    --python-script <repo>/headless_anchor.py
```

この経路では `LFS_MPC_ENABLED` の既定が `1` になります（GUI プラグインの既定は
オフ）。その他の設定は `LFS_MPC_*` 環境変数で渡します（[一覧](#パラメータ一覧)）。
たとえば自己校正を有効にするなら、起動前に PowerShell では
`$env:LFS_MPC_CALIBRATE = "1"`、bash では `export LFS_MPC_CALIBRATE=1` を設定します。

### 入力クロップ（CLI）

学習の前に、対象領域の箱で `points3D.txt` をフィルタした `<colmap>_cropped` を
作ります。まず `--dry-run` で「その箱が何点落とすか」だけを確認し、箱が決まったら
外して本実行します:

```bash
.venv\Scripts\python.exe crop_input.py --data <colmap> --box "x0,y0,z0:x1,y1,z1" --dry-run
.venv\Scripts\python.exe crop_input.py --data <colmap> --box "x0,y0,z0:x1,y1,z1"
```

`cameras.txt` / `images.txt` はハードリンク、`images/` はジャンクションで共有する
ため、追加ディスクは `points3D.txt` の分だけです（実測: 4.1M 点で 14 秒・415 MB）。
箱には既定で 5% の余白が付きます（`--pad`）。使った箱は出力先の `crop_input.json`
に記録されるので、複数の箱を試しても後からどの学習がどの箱か判別できます。

なぜ学習前のクロップなのか: 実測では撮影点群の 0.68% がすでに対象領域の外にあり、
学習はその点の上にスプラットを育てていました。アンカーはむしろ点の上に行を
留める道具なので、この問題には構造的に効きません（実測 −0.3%、効果/ノイズ 0.1）。
入力から点を消すのが唯一実用的な手でした。

### 入力クロップ（GUI — Input Crop パネル）

RealityScan の再構成領域に相当する操作をビューア上で行えます。

1. データセットを開くと COLMAP パスは自動で入ります（手入力も可）
2. **Seed default region** か **Fit to cloud**（点群の p0.5〜p99.5）で箱を配置
3. ビューア標準のクロップ箱ギズモで調整（パネル独自のギズモはありません）
4. **Count (sampled)** で落ちる点の割合を約 1 秒で確認（1/8 サンプリング）
5. **Export cropped dataset** で書き出し（CLI と同一実装・全点の厳密パス）

書き出されるのは箱の軸平行 min/max です。箱を回転させても書き出しには反映されず、
**inverse**（外側を残す）設定の箱は書き出しを拒否します — 被写体を消して背景を残す
事故の防止です。

## 仕組み（位置アンカー）

各イテレーションの最適化ステップ後に、スプラット位置 μ へ次の補正を適用します。

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

「損失に加算」ではなく「ステップ後補正」なのは、LichtFeld Studio の Python フックが
train step 完了後にしか走らず、勾配バッファが Python に公開されていないためです。
一方 `means_raw` は書き込み可能な CUDA ビューとして公開されており（公式ドキュメント
公認）、ステップ後補正はアンカー損失の明示的 SGD ステップと数学的に等価です。
詳細は [docs/DESIGN.md](docs/DESIGN.md)、C++ ネイティブ実装の計画は
[docs/cpp-native-loss.md](docs/cpp-native-loss.md)。

### MCMC 密度制御への対応

LichtFeld の MCMC 戦略はスプラットを毎リファインで再配置・追加しますが、
行インデックスは並べ替えません。本プラグインは:

- **追加された行** → 既定では非アンカー（維持したいのは事前配置した点群であって、
  MCMC の誕生位置に幾何的な意味はないため）
- **再配置された行** → 「生存行の閾値超ジャンプ」または「死→生の不透明度不連続」で
  検出し、移動先を新しいアンカーとして再登録
- **瀕死の行**（`min_pull_opacity` 未満） → 引かない。不可視で MCMC ノイズに
  支配されるため
- **1 イテレーションで 5% 超が跳んだ場合** → モデル全体イベント（Bake Transform 等）
  とみなし、警告を出して全再キャプチャ

さらに `mode = "nn"` では、固定参照点群（初期スナップショットまたは外部 PLY）への
最近傍点をアンカーとして定期再割当てします（要 scipy。密度制御に構造的に強い方式）。

## 推奨設定

**位置アンカー** — 自己校正を有効にする。これだけです:

```bash
LFS_MPC_ENABLED=1
LFS_MPC_CALIBRATE=1
```

`LFS_MPC_FREE_RADIUS` は指定しないでください。外部仕様としての公差が無い限り、
ドリフト分布の実測から決まる値のほうが固定値より確実に妥当です。実測でも自己校正は
手動掃引が見つけた最良値を 2% 以内で再現し、上回りました。校正の分位点
`calibrate_quantile` の既定は 60 です（v0.4.0 で 70 から変更。下げるほど位置は
締まるがスプラット形状が代償になるトレードオフで、60 は 2 データセットの掃引が
一致した点です）。

**入力クロップ** — 納品・利用範囲が箱で決まっているワークフローなら、学習前に
必ず通してください。余白は既定の 5% のままでよいです（確定した境界を事前に
知り得ないため）。

**`--max-cap` は初期点群の点数の 2 倍以上**にしてください（下記「落とし穴」）。

### やってはいけないこと

| 操作 | 何が起きるか |
|---|---|
| `LFS_MPC_CROP_BOX`（学習中の ROI クロップ） | 領域外は 26% 減るが、**納品する領域そのものが 3.71 dB 悪化**。実装は「試して駄目だった」記録として残してあるだけ |
| `anchor_new_splats=1` を `index` モードで使う | 全スプラットが誕生位置に凍結され、再配置 +48%・スケール +119%。「初期点群の維持」ではなく「全体凍結」という別物になる |
| `reanchor_on_split=1` | スケール膨張は 2/3 消えるが、逸脱行数が 25 → 106,080 に戻る |

## パラメータ一覧

GUI パネルの各項目と環境変数は同じパラメータに対応します。

| パラメータ | 環境変数 | 既定 | 意味・推奨 |
|---|---|---|---|
| `enabled` | `LFS_MPC_ENABLED` | `False`（headless 経路では `True`） | 有効化 |
| `strength` | `LFS_MPC_STRENGTH` | `0.01` | 不感帯を超えた分を毎 iter 引き戻す割合。0.1 以上は事実上の凍結 |
| `free_radius` | `LFS_MPC_FREE_RADIUS` | `0`（自動） | 不感帯半径（ワールド単位）。0 のまま推奨 |
| `free_radius_spacing` | `LFS_MPC_FREE_RADIUS_SPACING` | `2.0` | 自動不感帯 = 点群の中央値点間隔 × この倍率（`calibrate` 無効時の静的規則） |
| `calibrate` | `LFS_MPC_CALIBRATE` | `False` | 不感帯を対照行のドリフト分位点から自動決定。**True 推奨** |
| `calibrate_quantile` | `LFS_MPC_CALIB_Q` | `60` | 校正に使う分位点。下げると位置が締まり、スケールが膨らむ |
| `control_fraction` | `LFS_MPC_CONTROL_FRACTION` | `0.02` | 対照行（引かずに自由ドリフトを測る行）の割合。上げると置いた点群が保持されなくなる |
| `calibrate_every` | `LFS_MPC_CALIB_EVERY` | `500` | 校正間隔（iter） |
| `calibrate_start` | `LFS_MPC_CALIB_START` | `1000` | 校正開始 iter。遅らせると分布の胴体が悪化する |
| `calibrate_seed` | `LFS_MPC_CALIB_SEED` | `0` | 対照行選択の乱数種（再現用） |
| `calibrate_min_spacing` | `LFS_MPC_CALIB_MIN_SPACING` | `0.25` | 不感帯の下限（× 点間隔） |
| `huber_delta` | `LFS_MPC_HUBER_DELTA` | `0`（無効） | 1 iter の引き戻し距離の上限（有界影響）。使うなら free_radius の 2〜4 倍 |
| `max_distance` | `LFS_MPC_MAX_DISTANCE` | `0`（∞） | これ以遠は正当な新規構造とみなし引かない |
| `opacity_gate` | `LFS_MPC_OPACITY_GATE` | `False` | sigmoid(不透明度) を引力に乗算。`nn` モードで推奨 |
| `min_pull_opacity` | `LFS_MPC_MIN_PULL_OPACITY` | `0.01` | この不透明度未満の不可視行は引かない |
| `warmup_iters` | `LFS_MPC_WARMUP` | `300` | 強度のランプアップ iter 数 |
| `start_iter` / `stop_iter` | `LFS_MPC_START` / `LFS_MPC_STOP` | `0` | 適用区間。維持が目的なら切らない |
| `anchor_new_splats` | `LFS_MPC_ANCHOR_NEW` | `False` | MCMC の追加・再配置行を誕生位置でアンカー。**False のまま** |
| `teleport_threshold` | `LFS_MPC_TELEPORT` | `0`（自動） | 再配置の判定距離。自動 = 頑健シーン対角の 1% |
| `mode` | `LFS_MPC_MODE` | `index` | `index`（誕生位置）/ `nn`（参照点群への最近傍） |
| `reference_ply` | `LFS_MPC_REF_PLY` | 初期スナップショット | `nn` モードの参照点群 |
| `nn_refresh` | `LFS_MPC_NN_REFRESH` | `100` | `nn` の再割当て間隔（iter） |
| `reanchor_on_split` | `LFS_MPC_REANCHOR_SPLIT` | `False` | 分割行の再アンカー。**False のまま** |
| `stats_out` | `LFS_MPC_STATS_OUT` | 出力しない | ドリフト統計 JSON の出力先 |
| `stats_snapshot_every` | `LFS_MPC_SNAPSHOT_EVERY` | `1000` | 統計 JSON を書き直す間隔。最後に書けたものが最終結果 |
| `log_every` | `LFS_MPC_LOG_EVERY` | `0` | 定期ログ間隔 |
| — | `LFS_MPC_CONFIG` | — | 全パラメータをまとめた JSON ファイルのパス |

## 効きを自分のデータで確かめる

`LFS_MPC_STATS_OUT` を指定すると、初期行のドリフト統計（p50/p75/p90/p95/max、
逸脱行数、シーン対角付き）が JSON で出力されます。アンカー無効・有効を回して
比較します。最初に見る値は 2 つ:

1. **`applied_iters`** — 0 なら補正は一度も適用されていません（`enabled` の既定は
   オフです。まずここを疑ってください）
2. **`init_drift.p95 ÷ config.free_radius_effective`**（アンカー無効の 1 本で計算）—
   1 を下回っていたら初期行の 95% 以上が不感帯の内側にいて、リーシュには引く対象が
   ありません。そのまま A/B を回しても全指標がノイズ以下になります。`calibrate`
   を使うか、`free_radius_spacing` を下げて比を 1 以上にしてから比較してください

測るときの注意（すべて実際に誤結論を出したパターンです）:

- **MCMC はシード固定不可**です。ベースラインを 2 本回してアーム内ばらつき
  （ノイズフロア）を先に測り、効果はその倍率で判定してください。1 本対 1 本の
  比較は何も言えません
- **`max` を効果量として読まない。** 単一行の値で、ベースライン同士でも 9 倍
  振れます。裾は `p99` / `p999` と逸脱行数（`escaped_1x/2x/4x` — 点間隔の
  1/2/4 倍を超えた行数）で見てください
- **中央値だけを見ない。** 不感帯の内側の大多数は自由に動くので、中央値は
  アンカー無効時とほぼ一致します。作用は上位の裾にだけ現れます
- **領域を絞る比較に全画面 PSNR を使わない。** 対象領域の内側だけを測って
  初めて見える害があります（`scripts/inbox_psnr.py`）

検証の作法・環境の罠の全記録は [docs/HANDOFF.md](docs/HANDOFF.md) にあります。

## 落とし穴

### 初期点群が `--max-cap` を超えると、目的が静かに壊れる

このプラグインの用途で最も踏みやすい罠です。LichtFeld Studio の `max_cap` 既定は
5,000,000 で、初期点群がこれを超えると警告 1 行を出して**ランダムに間引きます**:

```
[warning] training_setup.cpp:356 Max cap (5000000) is less than initial splat
          count (7886332), randomly selecting 5000000 splats
```

このとき二つのことが同時に起きます。

1. **維持したい点群そのものが減ります。** 上の例では 36.6% が学習開始前に
   捨てられました
2. **密度制御が事実上停止します。** 初期点数が上限に張り付き、MRNF が
   `utilization: 100%` で開始して再配置・追加の空きがなくなります。実測では
   密度制御の再配置回数に 1000 倍の差が出ました

**初期点群の点数の 2 倍以上を明示してください。** 点数は `points3D.txt` の 3 行目
（`# Number of points:`）にあります。VRAM の目安はエンジンで大きく違い、
v0.5.3 系では 7.9M 点・SH 次数 3 で 7.3 GB でした（古い版はこの 2 倍以上）。

### ヘッドレス実行

- `--python-script` を渡さないと Python ランタイム自体が起動せず、プラグインも
  読み込まれません（`--headless` との併用は可能です）
- `--python-script` 経路ではプラグイン専用 venv が有効化されません。
  `headless_anchor.py` が隣接 `.venv` を自動で `sys.path` に足しますが、numpy /
  scipy がどこにも無いと起動時にエラーログを出します。**この警告を無視しないで
  ください** — 依存が欠けると自動不感帯が 0 に落ち、終盤の完全凍結に至ります
- 統計 JSON は `stats_snapshot_every` ごとに上書きされ、最後に書けたものが最終結果に
  なります。学習終了フックが発火しないエンジン（v0.5.1 headless で実測）でも
  結果が残るようにするためです
- `--steps-scaler` は 0.003 以上にしてください（それ未満は密度制御のスケジュール値が
  0 に切り捨てられゼロ除算で落ちます）

## 実測結果の要約

詳細は [docs/RESULTS.md](docs/RESULTS.md)。ここでは結論だけ。

**効いたもの**

| | 効果 | 効果/ノイズ |
|---|---|---|
| 自己校正不感帯（q70）— ドリフト p95 | −44.9% | 72.3 |
| 同 — 逸脱行数（可視行、点間隔の 2 倍超） | 204,743 行 → **0** | — |
| q を 60 / 50 に下げる — ドリフト p95 | −52.4% / −57.9% | 320 / 112 |
| **入力クロップ** — 対象領域外のスプラット | **−56.0% / −59.1%**（2 データセット） | 15.5 / 51.1 |
| 入力クロップ — 孤立フローター（novel view 実測） | −91% | — |

入力クロップの対象領域内 PSNR コストは −0.24 dB 〜 +0.11 dB（改善する
データセットもあります）。**アンカーの測光コストは一貫して検出されず**、
hold-out PSNR の効果/ノイズは全条件で 0.1〜0.4 でした。幾何を p95 で半分近く
締めても、見えの品質は落ちていません。

**効かなかった／割に合わなかったもの**

| | 結果 |
|---|---|
| `index` アンカーで領域外スプラットを減らす | −0.3%（効果/ノイズ 0.1）。撮影点自体が領域外にあり、アンカーはそこに留める |
| 学習中クロップ箱 | 領域外 −26% だが対象領域が −3.71 dB 悪化 |
| クロップ済み入力への `nn` + `anchor_new` の重ね掛け | 追加効果は効果/ノイズ 1.3〜1.76 で未確立。対象領域 PSNR に −0.29〜−1.04 dB のコスト |
| `reanchor_on_split=1` | スケール膨張 2/3 減と引き換えに逸脱行数が桁で戻る |

## 制限事項

**位置アンカー**

- 補正は Adam のモーメントを経由しない明示的ステップです（正則化圧としては等価。
  厳密な「損失加算」は C++ 側対応が必要 — [docs/cpp-native-loss.md](docs/cpp-native-loss.md)）
- チェックポイント再開時、アンカーは再開時点の位置で再キャプチャされます
- テレポート閾値未満の変換（小さな Bake Transform 等）は検出できず、アンカーが
  変換前の姿勢へ引き戻します。編集後はパネルの「Re-capture anchors now」を推奨
- `min_pull_opacity` 未満の不可視行は引かれません（設計どおり）。極端に漂う行は
  ほぼ全てこれで、全行の `max` や `escaped_*` は汚染されます。可視行だけの系列
  `visible_drift` を併せて見てください
- 統計を全部有効にすると GPU→CPU 転送が増えます（8M 行・30,000 iter の実測で
  実行時間 26.7 分 → 32.7 分）。常用するなら `stats_snapshot_every` を上げてください
- **対象領域の外に育つスプラットは解決しません。** それは入力クロップの担当です

**入力クロップ**

- 書き出されるのは箱の軸平行 min/max のみ（箱の回転は反映されません）
- `images.txt` の `POINT3D_ID` は付け替えません。COLMAP のリーダは宙に浮いた参照を
  許容し、LichtFeld Studio でも問題は出ていませんが、厳密なリーダは弾く可能性が
  あります

## ドキュメント

| ファイル | 内容 |
|---|---|
| [docs/RESULTS.md](docs/RESULTS.md) | 全 A/B の実測結果（4 データセット・のべ 60 本以上）と生の集計出力 |
| [docs/DESIGN.md](docs/DESIGN.md) | 設計判断の記録 |
| [docs/HANDOFF.md](docs/HANDOFF.md) | 検証を再開・再現するための引き継ぎ（作法・環境の罠・未解決の問い） |
| [docs/cpp-native-loss.md](docs/cpp-native-loss.md) | C++ ネイティブ実装のロードマップ |
| [CHANGELOG.md](CHANGELOG.md) | 変更履歴（各既定値の変更理由を含む） |

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

Two tools for training 3D Gaussian Splatting against a trusted, pre-placed point
cloud in [LichtFeld Studio](https://github.com/MrNeRF/LichtFeld-Studio),
implemented purely with the official Python plugin API (no C++ rebuild):

**Position anchor** (during training) — a soft leash, not a hard freeze. Each
iteration, after the optimizer step, splat means are pulled back toward their
anchors by `strength · min(max(0, ‖μ−a‖ − free_radius), huber_delta)`, the
proximal step of a dead-zone Huber anchor loss (LI-GS / GeomGS / EnerGS family).
MCMC relocation/growth is handled by index-stable birth anchoring with teleport
detection, or by nearest-neighbor re-targeting to a fixed reference cloud
(`mode="nn"`). The dead zone calibrates itself at runtime: a small held-out
fraction of rows is never pulled, their live drift quantile becomes the radius —
splats keep the freedom they need to settle onto surface detail, and lose the
freedom to wander. Measured across datasets: init-row drift p95 −56…−58%
(effect/noise 112–345), escaped rows 323,295 → 2–6, with no detectable
photometric cost.

**Input crop** (before training) — `crop_input.py` and the Input Crop GUI panel
filter `points3D.txt` to an axis-aligned box (RealityScan-style region of
interest) and emit a `<colmap>_cropped` dataset sharing images via
hardlinks/junctions. Out-of-region splats −56% / −59% on two subjects, isolated
floaters −91% in novel-view renders, in-region PSNR −0.24…+0.11 dB. Roughly 6×
more effective against manual-cleanup load than any training-time lever we
measured, because 0.68% of surveyed points already lie outside the region and
anchoring would hold splats exactly there.

All numbers are paired A/B runs (≥2 per arm) on real captures; see
[docs/RESULTS.md](docs/RESULTS.md) (Japanese) for the full record.

## License

GPL-3.0-or-later, matching LichtFeld Studio.
