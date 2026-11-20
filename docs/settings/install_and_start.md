# 1. インストールと起動 / 新しい設備の初期設定

## 動作環境

- Windows 10 / 11（生産設備の標準）または Linux（開発用）
- Python 3.11 系（本リポジトリの検証環境: 3.11.14）
- 依存パッケージ: PyQt5, numpy, opencv-python, pyyaml, ultralytics, sahi,
  pygame, mysql-connector-python, imagingcontrol4 (The Imaging Source IC4 SDK)
- pytest（自動テスト実行時のみ）

## 起動手順

```bat
:: 1. リポジトリを取得
git clone https://github.com/stefanoKS/aikensaMaster.git
cd aikensaMaster

:: 2. 仮想環境（例）
conda create -n aikensa python=3.11 -y
conda activate aikensa
pip install pyqt5 numpy opencv-python pyyaml ultralytics sahi pygame mysql-connector-python

:: 3. 起動
python main.py
```

起動すると `aikensa/logs/crash.log` に起動記録が追記されます。
異常終了時はこのファイルを確認してください。

## 起動時に何が起きるか

1. アプリケーションのルートが自動判定されます（`AIKENSA_ROOT` 環境変数でも指定可能）。
2. **設備固有ファイルが無い場合のみ**、Git 管理のテンプレートから作成されます。
   - 既存ファイルは絶対に上書きされません。
   - 新規作成されるテンプレートは「`null`（未設定）」です。
3. 設定は 1 か所（`SettingsService`）を通して検証付きで読み込まれます。
4. 校正が未設定でもアプリは起動し、メイン画面と設備設定画面を操作できます。
   ただし**生産検査は開始されません**。

作成されるファイル:

| パス | 役割 | 初期値 |
|------|------|--------|
| `aikensa/core/config/cam/cam_id.yaml` | 物理カメラのシリアル割当 | `0: null` / `1: null` |
| `aikensa/core/config/cam/cam_config.yaml` | 解像度・FPS・露光など | 3072x2048 / 15fps など |
| `aikensa/core/param/camera/Calibration_camera_0.yaml` | カメラ① 内部校正 | `null` |
| `aikensa/core/param/camera/Calibration_camera_1.yaml` | カメラ② 内部校正 | `null` |
| `aikensa/core/param/camera/homography_param_cam1.yaml` | カメラ① ホモグラフィ | `null` |
| `aikensa/core/param/camera/homography_param_cam2.yaml` | カメラ② ホモグラフィ | `null` |
| `aikensa/core/param/adjustments/global_alignment.yaml` | 全体微調整 | 0.0 |
| `aikensa/core/param/adjustments/widget_adjustments.yaml` | 品番別調整 | 空 |

`*_scaled.yaml`（プレビュー用の縮小版）と `planarizeTransform_*.yaml`（平面化変換）も
同名で作成されます。**ファイル名は変更しないでください**（下流の処理が依存しています）。

## 新しい設備の初期設定

新しい設備では、以下の順に設定します。

1. **カメラ設定** — メイン画面の「設備設定」→「カメラ設定」
   - 「カメラ検索 / 更新」で実機シリアルを取得
   - カメラ① / カメラ② にシリアル番号を割当
   - 解像度・FPS・露光・ゲイン・ホワイトバランスを設定
   - 「接続テスト」で画像が取得できることを確認
   - 「保存」
2. **校正** — 「設備設定」→「校正・位置調整」→「カメラ①を再校正」/「カメラ②を再校正」
   - 詳細: [calibration.md](calibration.md)
3. **全体微調整** — 必要な場合のみ。詳細: [fine_adjustment.md](fine_adjustment.md)
4. **品番別調整** — 必要な品番のみ。詳細: [fine_adjustment.md](fine_adjustment.md#品番別ウィジェット別調整)

すべて完了すると「設備設定」画面の上部に

```
検査準備: 完了
カメラ①: 校正済み (準備完了)
カメラ②: 校正済み (準備完了)
```

と表示されます。

## 環境変数

| 変数 | 用途 |
|------|------|
| `AIKENSA_ROOT` | アプリケーションルートを明示指定 |
| `AIKENSA_BACKUP_DIR` | バックアップ保存先（既定: `<repo>/../aikensa_backups`） |
| `AIKENSA_ALLOW_PLACEHOLDER_CAMERAS` | `1` でプレースホルダ映像を許可（開発専用。生産では絶対に設定しない） |
| `AIKENSA_DEBUG_ACCEPT_ANY_MARKING` | デバッグ用マーカーID検査スキップ。**生産では未設定のまま** |
