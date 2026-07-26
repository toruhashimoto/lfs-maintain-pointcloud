# Changelog

## 0.1.0 (2026-07-26)

初回リリース。

- dead-zone Huber 位置アンカーの proximal 補正実装（純 Python、C++ 再ビルド不要）
- 独自パラメータ: strength / free_radius / huber_delta / max_distance /
  opacity_gate / warmup_iters / start_iter / stop_iter / anchor_new_splats /
  teleport_threshold / mode(index|nn) / reference_ply / nn_refresh
- MCMC/MRNF トポロジー追跡（append 誕生アンカー・テレポート検出再アンカー）
- GUI パネル (PointCloud Anchor)、LFS_MPC_* 環境変数、--python-script 用
  ヘッドレスエントリ、終了時ドリフト統計 JSON 出力
- numpy リファレンス数式のユニットテスト 15 本
- 敵対的レビュー (16 エージェント) を経た堅牢化:
  再配置検出の不透明度二重化 (ノイズ偽陽性・閾値未満偽陰性対策)、
  min_pull_opacity (不可視行を引かない)、モデル全体変換の検出と全再キャプチャ、
  UI 操作のフラグ化 (スレッドレース根絶)、例外時の自己修復、
  ドリフト統計から再配置行を除外、nn 参照点群の不変スナップショット化と
  reference_ply 変更時の KD-tree 再構築
