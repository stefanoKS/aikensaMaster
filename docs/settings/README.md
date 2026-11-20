# 設備設定ドキュメント (AIKensaMaster)

本ディレクトリは、AIKensaMaster の**設備設定**を運用者が安全に扱うための日本語ドキュメントです。
Python や YAML を直接編集しなくても、日常的な設備パラメータは GUI から変更できます。

## 目次

| No | 内容 | ファイル |
|----|------|----------|
| 1 | インストールと起動 | [install_and_start.md](install_and_start.md) |
| 2 | 新しい設備の初期設定 | [install_and_start.md](install_and_start.md#新しい設備の初期設定) |
| 3 | カメラのシリアル番号割当 | [camera_assignment.md](camera_assignment.md) |
| 4 | カメラ① / カメラ② の校正 | [calibration.md](calibration.md) |
| 5 | ホモグラフィの算出 | [calibration.md](calibration.md#ホモグラフィの算出) |
| 6 | 全体微調整 | [fine_adjustment.md](fine_adjustment.md) |
| 7 | 品番別（ウィジェット別）調整 | [fine_adjustment.md](fine_adjustment.md#品番別ウィジェット別調整) |
| 8 | 設定の保存と復元 | [save_and_restore.md](save_and_restore.md) |
| 9 | 未設定 / 準備完了 / 異常 の状態 | [save_and_restore.md](save_and_restore.md#状態の意味) |
| 10 | `git pull` から設備設定を守る | [git_pull_protection.md](git_pull_protection.md) |
| 11 | バックアップからの復元 | [save_and_restore.md](save_and_restore.md#バックアップからの復元) |
| 12 | 新しい品番の追加 | [add_new_part.md](add_new_part.md) |
| 13 | 自動テストの実行 | [testing.md](testing.md) |
| 14 | カメラ・校正の異常時の対処 | [troubleshooting.md](troubleshooting.md) |
| 参考 | パラメータ一覧表 | [parameter_reference.md](parameter_reference.md) |
| 参考 | Qt Designer での .ui 編集 | [qt_designer.md](qt_designer.md) |
| 参考 | 実機検証チェックリスト | [hardware_checklist.md](hardware_checklist.md) |
| 参考 | Git 移行手順（エンジニア向け） | [git_migration_runbook.md](git_migration_runbook.md) |

## 最重要ルール

1. **設備固有のファイルは絶対にコミットしない**
   カメラのシリアル番号・校正行列・位置補正値は設備ごとに異なります。
   これらは `.gitignore` で除外されています。

2. **`git pull` の前に必ずバックアップする**
   `tools/backup_machine_config.cmd` を実行し、“BACKUP VERIFIED” を確認してください。
   詳細: [git_pull_protection.md](git_pull_protection.md)

3. **校正データが `null` のときは検査を開始しない**
   アプリは「未設定」と表示し、生産検査を開始しません（クラッシュしません）。

## 用語

| 用語 | 意味 |
|------|------|
| 論理カメラ番号 0 | 画面表記「カメラ①」。`Calibration_camera_0.yaml` を使用 |
| 論理カメラ番号 1 | 画面表記「カメラ②」。`Calibration_camera_1.yaml` を使用 |
| 内部校正 (intrinsic) | レンズ歪み・焦点距離の校正 |
| ホモグラフィ | カメラ画像を結合テンプレート平面へ投影する変換 |
| 全体微調整 | カメラ②をカメラ①に対して平行移動／回転する補正 |
| 品番別調整 | 特定ウィジェットのみに適用する補正 |
| 共有設定 | Git 管理される既定値（全設備共通） |
| 設備固有設定 | Git 管理外の実機値（この設備だけ） |
